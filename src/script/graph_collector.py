import os
import re
import json
import concurrent.futures
import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import pandas as pd
from numpy.ma.extras import unique
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from src import project_path
from src.utils.custom_logging import setup_logging
from sklearn.metrics.pairwise import cosine_similarity
import torch_geometric
from torch_geometric.data import Data
from env import Env
import networkx as nx
import random
import torch
from time import time

env = Env()
log = setup_logging()


@dataclass
class GraphNodeCollector:
    data_folder: str
    similarity_threshold: float = 0.5

    def __post_init__(self):
        self.metadata = pd.read_csv(os.path.join(project_path, self.data_folder, 'metadata.csv'))
        self.path_to_txt_emb = os.path.join(project_path, self.data_folder, 'embeddings', 'texts')
        self.path_to_aud_emb = os.path.join(project_path, self.data_folder, 'embeddings', 'audios')
        self.path_to_img_emb = os.path.join(project_path, self.data_folder, 'embeddings', 'images')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f'device: {self.device}')
        self.path_to_new_metadata = os.path.join(self.data_folder, 'metadata_newform.csv')
        self.path_to_category_mapping = os.path.join(self.data_folder, 'category_mapping.json')
        self.path_to_category_list = os.path.join(self.data_folder, 'category_list.json')
        self.path_to_subcategory_list = os.path.join(self.data_folder, 'subcategory_list.json')
        self.path_to_graph = os.path.join(self.data_folder, 'graph.pt')
        self.cat2idx = None
        self.idx2cat = None
        self.subcat2idx = None
        self.idx2subcat = None

        self.embeddings = {
            "txt": None,
            "aud": None,
            "img": None
        }

    def get_unique_categories_and_subcategories(self) -> Tuple[np.ndarray, np.ndarray]:
        unique_categories = self.metadata['category'].unique()
        self.cat2idx = {category: idx for idx, category in enumerate(unique_categories)}
        self.idx2cat = {idx: category for idx, category in enumerate(unique_categories)}
        unique_subcategories = self.metadata['tag'].unique()
        self.subcat2idx = {subcategory: idx for idx, subcategory in enumerate(unique_subcategories)}
        self.idx2subcat = {idx: subcategory for idx, subcategory in enumerate(unique_subcategories)}
        return unique_categories, unique_subcategories

    def get_emb(
            self,
            modality: str,
    ) -> list:
        if modality == 'txt':
            path = self.path_to_txt_emb
        elif modality == 'aud':
            path = self.path_to_aud_emb
        elif modality == 'img':
            path = self.path_to_img_emb
        else:
            raise NotImplementedError("txt, aud, or img доступно")

        emb = []

        with tqdm(total=len(self.metadata)) as pbar:
            for index, row in self.metadata.iterrows():

                if modality == 'txt':
                    emb.append({
                        "video_id": row["video_id"],
                        "embedding": torch.tensor(torch.load(
                            os.path.join(path, f"{row['video_id']}.pt"),
                            weights_only=False, map_location=torch.device('cpu')
                        )['dense_vecs'], dtype=torch.float32).mean(dim=0)})
                elif modality == 'aud':
                    emb.append({
                        "video_id": row["video_id"],
                        "embedding": torch.load(
                            os.path.join(path, f"{row['video_id']}.pt"),
                            weights_only=False,
                            map_location=torch.device('cpu')).mean(dim=0).mean(dim=0)})
                elif modality == 'img':
                    emb.append({
                        "video_id": row["video_id"],
                        "embedding": torch.load(
                            os.path.join(path, f"{row['video_id']}.pt"),
                            weights_only=False, map_location=torch.device('cpu')
                        ).to(dtype=torch.float32).mean(dim=0)})
                pbar.update(1)
                pbar.set_description(f"Сбор эмбеддингов {modality}")

        return emb

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

    def save_graph_gexf(self, G: nx.Graph):
        # Сохранение в GEXF
        path = self.path_to_graph
        nx.write_gexf(G, path)
        log.info(f"Граф сохранен в формате GEXF: {path}")

    # Функция для кодирования видео
    def encode_video(self, row, num_categories, num_subcategories):
        category = row["category"]
        tag = row["tag"]

        # Создаем нулевой тензор размерности (43)
        tensor_cat = torch.zeros(num_categories, dtype=torch.int)
        # Создаем нулевой тензор размерности (1007)
        tensor_sub = torch.zeros(num_subcategories, dtype=torch.int)

        # Заполняем категорию
        tensor_cat[self.cat2idx[category]] = 1

        # Заполняем подкатегорию
        tensor_sub[self.subcat2idx[tag]] = 1

        return tensor_cat, tensor_sub

    @staticmethod
    def compute_cosine_distance(i, j, node_features):
        # Вычисляем косинусное расстояние для объединённого вектора
        dist = cosine_distances(
            node_features[i].reshape(1, -1), node_features[j].reshape(1, -1)
        )[0][0]

        # Возвращаем пару индексов и косинусное расстояние
        return i, j, dist

    def get_sampled_metadata(self, sample_ratio=0.1):
        """
        Отбирает первые sample_ratio процентов данных из self.metadata.
        """
        # Удаление дубликатов перед выборкой
        self.metadata = self.metadata.drop_duplicates(subset=["video_id"])
        # Расчет количества элементов для выборки
        total_videos = len(self.metadata)
        num_sample_videos = int(total_videos * sample_ratio)

        # Просто берем первые num_sample_videos строк из self.metadata
        sampled_metadata = self.metadata.iloc[:num_sample_videos]

        return sampled_metadata

    def build_and_save_graph(self, categories: List[str], subcategories: List[str],
                             embeddings, category_mapping, sample_ratio=0.1):
        # Получаем выборку данных (10%)
        sampled_metadata = self.get_sampled_metadata(sample_ratio=sample_ratio)

        # Дальше работаем с sampled_metadata, как обычно
        log.info(f'Количество видео в выборке: {len(sampled_metadata)}')

        # Размерности
        num_categories = len(categories)  # 43
        num_subcategories = len(subcategories)  # 1007

        # Создание словаря video_id -> индекс
        video_id_to_index = {video_id: idx for idx, video_id in enumerate(sampled_metadata['video_id'])}

        # Словарь video_id: tensor
        video_tensors = {}

        # Создание тензоров для каждого видео
        for _, row in sampled_metadata.iterrows():
            video_id = row["video_id"]
            tensor_cat, tensor_sub = self.encode_video(row, num_categories, num_subcategories)
            video_tensors[video_id] = (tensor_cat, tensor_sub)

        # Список для хранения фич и рёбер
        node_features = []
        edge_index = []
        edge_attr = []
        labels = []

        video_ids = list(sampled_metadata['video_id'])
        unique_video_ids = np.unique(video_ids)
        log.info(len(video_ids))
        log.info(len(unique_video_ids))

        # Множество для отслеживания добавленных video_id
        added_video_ids = set()

        # Добавляем фичи для каждой вершины
        for embedding in embeddings["txt"]:  # Пример для txt, предполагается, что другие модальности совпадают
            video_id = embedding["video_id"]

            # Пропускаем те video_id, которых нет в выборке или которые уже были добавлены
            if video_id not in sampled_metadata['video_id'].values or video_id in added_video_ids:
                continue

            # Отмечаем video_id как добавленный
            added_video_ids.add(video_id)

            # Получаем эмбеддинги для каждой модальности
            txt_emb = embedding["embedding"]
            aud_emb = next(e["embedding"] for e in embeddings["aud"] if e["video_id"] == video_id)
            img_emb = next(e["embedding"] for e in embeddings["img"] if e["video_id"] == video_id)

            # Объединяем все эмбеддинги в одну строку
            node_features.append(np.concatenate([txt_emb, aud_emb, img_emb]))

            # Получаем категорию и подкатегорию для этого видео
            tensor_cat, tensor_sub = video_tensors[video_id]

            # Добавляем метки (категория и подкатегория)
            labels.append(torch.cat((tensor_cat, tensor_sub), dim=0))  # Соединяем категорию и подкатегорию

        # Преобразуем список фич в tensor
        node_features = torch.tensor(np.array(node_features), dtype=torch.float32)

        # Список для рёбер и атрибутов
        edge_index = []
        edge_attr = []

        # # Используем ThreadPoolExecutor для распараллеливания расчёта расстояний
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []
        #
        #     # Создаем пул задач
        #     with tqdm(total=(len(video_ids) * (len(video_ids) - 1)) // 2, desc="Задачи в пуле", ncols=100) as task_bar:
        #         for i in range(len(video_ids)):
        #             for j in range(i + 1, len(video_ids)):
        #                 futures.append(executor.submit(self.compute_cosine_distance, i, j, node_features))
        #                 task_bar.update(1)
        #
        #     # Обрабатываем результаты
        #     with tqdm(total=len(futures), desc="Вычисление рёбер", ncols=100) as result_bar:
        #         for future in concurrent.futures.as_completed(futures):
        #             i, j, dist = future.result()
        #
        #             # Добавляем ребро только если косинусное расстояние меньше 0.5
        #             if dist < 0.5:
        #                 edge_index.append([i, j])  # Добавляем ребро
        #                 edge_attr.append(torch.tensor([dist], dtype=torch.float32))  # Добавляем атрибут рёбер
        #
        #                 # Для неориентированного графа добавляем симметричное ребро
        #                 edge_index.append([j, i])
        #                 edge_attr.append(torch.tensor([dist], dtype=torch.float32))  # Атрибут для симметричного рёбра
        #
        #             result_bar.update(1)

        # Создаем цикл с tqdm для отслеживания прогресса
        with tqdm(total=(len(video_ids) * (len(video_ids) - 1)) // 2, desc="Вычисление рёбер", ncols=100) as progress_bar:
            for i in range(len(video_ids)):
                for j in range(i + 1, len(video_ids)):
                    # Вычисляем косинусное расстояние
                    i, j, dist = self.compute_cosine_distance(i, j, node_features)

                    # Добавляем ребро только если косинусное расстояние меньше 0.5
                    if dist < 0.5:
                        edge_index.append([i, j])  # Добавляем ребро
                        edge_attr.append(torch.tensor([dist], dtype=torch.float32))  # Добавляем атрибут рёбер

                        # Для неориентированного графа добавляем симметричное ребро
                        edge_index.append([j, i])
                        edge_attr.append(torch.tensor([dist], dtype=torch.float32))  # Атрибут для симметричного рёбра

                    # Обновляем прогресс-бар
                    progress_bar.update(1)

        # Преобразуем список рёбер и атрибутов в tensor
        edge_index = torch.tensor(edge_index).t().contiguous()  # Размерность (2, num_edges)
        edge_attr = torch.stack(edge_attr)  # Размерность (num_edges, 1)

        # Преобразуем метки в tensor
        labels = torch.stack(labels)

        # Индексы для каждой вершины (вместо чисел используем video_id)
        index = torch.tensor([video_id_to_index[video_id] for video_id in video_ids], dtype=torch.long)

        # label_mask - маска для всех вершин (единицы для всех)
        label_mask = torch.ones(len(node_features), dtype=torch.bool)

        # Проверяем типы данных
        assert isinstance(node_features, torch.Tensor), f"node_features должен быть torch.Tensor, но получен {type(node_features)}"
        assert isinstance(edge_index, torch.Tensor), f"edge_index должен быть torch.Tensor, но получен {type(edge_index)}"
        assert isinstance(edge_attr, torch.Tensor), f"edge_attr должен быть torch.Tensor, но получен {type(edge_attr)}"
        assert isinstance(labels, torch.Tensor), f"labels должен быть torch.Tensor, но получен {type(labels)}"

        # Создаем объект Data
        data = Data(
            x=node_features,  # Признаки узлов
            edge_index=edge_index,  # Рёбра
            edge_attr=edge_attr,  # Атрибуты рёбер
            y=labels,  # Метки
            label_1=labels[:, 0],  # Метка для категории
            label_2=labels[:, 1],  # Метка для подкатегории
            label_mask=label_mask,  # Маска меток
            index=index  # Индексы узлов
        )

        # Проверка на количество узлов и рёбер
        log.info(f'Количество узлов в графе: {data.num_nodes}')
        log.info(f'Количество рёбер в графе: {data.num_edges}')

        # Сохранение графа в файл в формате GEXF (по желанию)
        # self.save_graph_gexf(data)

        # Вывод краткой статистики
        log.info(f'Граф создан: узлы={data.num_nodes}, рёбра={data.num_edges}')
        return data

    def run(self):
        start = time()
        log.info(f'Запуск процесса')

        # Уникальные категории и подкатегории
        unique_categories, unique_subcategories = self.get_unique_categories_and_subcategories()

        # Эмбеддинги всех подкатегорий
        log.info(f'Получаем все ембедды...')

        self.embeddings = {
            "txt": self.get_emb("txt"),
            "aud": self.get_emb("aud"),
            "img": self.get_emb("img")
        }

        # Словарь текущих категорий и подкатегорий
        category_mapping = self.get_cat_sub_lists()

        # Построение и сохранение графа
        log.info(f'Построение графа...')
        graph = self.build_and_save_graph(unique_categories.tolist(), unique_subcategories.tolist(),
                                          self.embeddings, category_mapping)

        log.info(graph)

        # Сохраняем граф в файл
        torch.save(graph, self.path_to_graph)

        log.info(f'Граф сохранен по пути: {self.path_to_graph}')

        log.info(f'Процесс завершен. Время выполнения: {time() - start:.2f} секунд')

