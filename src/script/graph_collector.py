import os
import re
import json

import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np

from src import project_path
from src.utils.custom_logging import setup_logging
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity
from env import Env
import networkx as nx

from time import time

env = Env()
log = setup_logging()


@dataclass
class GraphNodeCollector:
    path_to_data: str = os.path.join(project_path, env.__getattr__("DATA_PATH"))
    path_to_dir: str = os.path.join(project_path, env.__getattr__("DATA_PATH"))
    similarity_threshold: float = 0.65  # Порог сходства

    def __post_init__(self):
        self.text_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
        self.metadata = pd.read_csv(os.path.join(project_path, self.path_to_data, 'metadata.csv'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f'device: {self.device}')
        self.path_to_new_metadata = os.path.join(self.path_to_dir, 'metadata_newform.csv')
        self.path_to_graph = os.path.join(self.path_to_dir, 'graph.gexf')
        self.cat2idx = None
        self.idx2cat = None
        self.subcat2idx = None
        self.idx2subcat = None

    def text_pipeline(self, text: list[str]) -> np.ndarray:
        # Возвращает эмбеддинги для списка текстов
        return self.text_model.encode(text, max_length=1024, return_dense=True)

    def get_unique_categories_and_subcategories(self) -> Tuple[np.ndarray, np.ndarray]:
        unique_categories = self.metadata['category'].unique()
        self.cat2idx = {category: idx for idx, category in enumerate(unique_categories)}
        self.idx2cat = {idx: category for idx, category in enumerate(unique_categories)}
        unique_subcategories = self.metadata['tag'].unique()
        self.subcat2idx = {subcategory: idx for idx, subcategory in enumerate(unique_subcategories)}
        self.idx2subcat = {idx: subcategory for idx, subcategory in enumerate(unique_subcategories)}
        return unique_categories, unique_subcategories

    def get_cat_sub_lists(self) -> Dict[str, List[str]]:
        # Словарь для хранения категорий и их подкатегорий
        category_mapping = {}

        # Группировка данных по категориям
        grouped = self.metadata.groupby('category')

        # Итерация по каждой категории
        for category, group in grouped:
            # Уникальные подкатегории для данной категории
            unique_subcategories = group['tag'].unique().tolist()
            category_mapping[category] = unique_subcategories
        return category_mapping

    # def correct_subcategories(self, categories: List[str], subcategories: List[str],
    #                           category_embeds: np.ndarray, all_subcategory_embeds: np.ndarray) -> Dict[str, List[str]]:
    #     updated_mapping = {}
    #
    #     # Для каждой категории ищем подкатегории с косинусным сходством выше порога
    #     for i, category in enumerate(categories):
    #         category_embed = category_embeds[i:i + 1]  # Эмбеддинг текущей категории
    #         similarities = cosine_similarity(category_embed, all_subcategory_embeds)[0]  # Сходства с подкатегориями
    #
    #         # Отбираем подкатегории, у которых сходство выше порога
    #         close_subcategories = [
    #             subcategories[j] for j, sim in enumerate(similarities) if sim >= self.similarity_threshold
    #         ]
    #
    #         updated_mapping[category] = close_subcategories
    #
    #     return updated_mapping

    def save_new_metadata(self, updated_mapping: Dict[str, List[str]]):
        new_metadata = []
        for category, subcategories in updated_mapping.items():
            for sub in subcategories:
                new_metadata.append({'category': category, 'tag': sub})
        new_metadata_df = pd.DataFrame(new_metadata)
        new_metadata_df.to_csv(self.path_to_new_metadata, index=False)
        log.info(f'Новая метадата сохранена в {self.path_to_new_metadata}')

    def save_graph_gexf(self, G: nx.Graph):
        # Сохранение в GEXF
        path = self.path_to_graph
        nx.write_gexf(G, path)
        log.info(f"Граф сохранен в формате GEXF: {path}")

    def build_and_save_graph(self, categories: List[str], subcategories: List[str],
                             category_embeds: np.ndarray, subcategory_embeds: np.ndarray,
                             category_mapping):
        # Создаем граф
        G = nx.Graph()

        # Добавляем узлы подкатегорий
        for subcategory in subcategories:
            G.add_node(self.subcat2idx[subcategory], type='subcategory', label=subcategory)

        # Добавляем узлы категорий
        for category in categories:
            G.add_node(self.cat2idx[category], type='category', label=category)

        # Добавляем рёбра только от категорий к подкатегориям
        for i, category in enumerate(categories):
            category_embed = category_embeds[i:i + 1]  # Эмбеддинг категории
            similarities = cosine_similarity(category_embed, subcategory_embeds)[0]  # Сходство

            for j, subcategory in enumerate(subcategories):
                subcategory_idx = self.subcat2idx[subcategory]  # Индекс подкатегории
                sim = similarities[j]  # Сходство между категорией и подкатегорией

                # Добавляем ребра от категории к подкатегории
                G.add_edge(self.cat2idx[category], subcategory_idx, weight=sim)

        # Проверка количества узлов в графе
        log.info(f'Количество узлов в графе: {len(G.nodes)}')  # Ожидается 43 узла

        # Сохранение графа в файл в формате GEXF
        self.save_graph_gexf(G)

        # Вывод краткой статистики
        log.info(f'Граф создан: узлы={len(G.nodes)}, рёбра={len(G.edges)}')
        return G

    def run(self):
        start = time()
        log.info(f'Запуск процесса проверки и коррекции подкатегорий')

        # Уникальные категории и подкатегории
        unique_categories, unique_subcategories = self.get_unique_categories_and_subcategories()

        # Эмбеддинги всех подкатегорий
        log.info(f'Вычисление эмбеддингов подкатегорий...')
        all_subcategory_embeds = self.text_pipeline(unique_subcategories.tolist())
        all_subcategory_embeds = all_subcategory_embeds['dense_vecs']
        all_category_embeds = self.text_pipeline(unique_categories.tolist())
        all_category_embeds = all_category_embeds['dense_vecs']

        # Словарь текущих категорий и подкатегорий
        category_mapping = self.get_cat_sub_lists()
        log.info(f'category_mapping: {category_mapping}')

        # Коррекция подкатегорий
        # updated_mapping = self.correct_subcategories(unique_categories.tolist(), unique_subcategories.tolist(),
        #                                              all_category_embeds, all_subcategory_embeds)
        # log.info(updated_mapping)

        # Построение и сохранение графа
        log.info(f'Построение графа...')
        graph = self.build_and_save_graph(unique_categories.tolist(), unique_subcategories.tolist(),
                                          all_category_embeds, all_subcategory_embeds, category_mapping)

        # Сохранение новой метадаты
        # self.save_new_metadata(updated_mapping)

        log.info(f'Процесс завершен. Время выполнения: {time() - start:.2f} секунд')


if __name__ == '__main__':
    graph = GraphNodeCollector()
    graph.run()
