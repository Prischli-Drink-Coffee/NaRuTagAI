import os
import re
import json

import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import pandas as pd
from numpy import dtype
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src import project_path
from src.utils.custom_logging import setup_logging
from src.utils.create_dir import create_directories_if_not_exist
from mpl_toolkits.mplot3d import Axes3D
import random

from time import time

log = setup_logging()


@dataclass
class ClusterCollector:
    data_folder: str
    path_to_plots: str
    task: str
    loky_max_cpu_count: int

    def __post_init__(self):

        os.environ["LOKY_MAX_CPU_COUNT"] = f"{self.loky_max_cpu_count}"

        # Определяем путь к метадате
        self.metadata_path = os.path.join(project_path, self.data_folder, 'metadata.csv')
        self.metadata = pd.read_csv(self.metadata_path)
        self.data_path = os.path.join(project_path, self.data_folder)
        self.path_to_txt_emb = os.path.join(project_path, self.data_folder, 'embeddings', 'texts')
        self.path_to_aud_emb = os.path.join(project_path, self.data_folder, 'embeddings', 'audios')
        self.path_to_img_emb = os.path.join(project_path, self.data_folder, 'embeddings', 'images')
        self.path_to_plots = os.path.join(project_path, self.path_to_plots, 'clusters')
        create_directories_if_not_exist([self.path_to_plots])

    def run(self):
        start = time()
        log.info(f'Запуск процесса построения кластеров для всех модальностей')

        embeddings = {
            "txt": self.get_emb("txt"),
            "aud": self.get_emb("aud"),
            "img": self.get_emb("img")
        }

        if self.task == "cluster":
            self.cluster_and_plot(embeddings)

        log.info(f'Завершение процесса кластеризации')
        log.info(f'Время завершения: {time() - start} секунд')

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

    def cluster_and_plot(self, embeddings: dict):

        log.info(f'Запуск процесса кластеризации')

        # Словарь для цветов каждой модальности
        modality_colors = {
            'txt': 'blue',
            'aud': 'green',
            'img': 'red'
        }

        # Получение категорий для каждого video_id
        video_id_to_category = dict(zip(self.metadata['video_id'], self.metadata['category']))

        # Находим максимальную длину среди всех эмбеддингов
        max_length = max(
            emb["embedding"].shape[0] for emb_list in embeddings.values() for emb in emb_list
        )

        combined_embeddings = []
        combined_labels = []
        concat_embeddings = []
        concat_labels = []
        concat_categories = []  # Категории для сконкатенированных эмбеддингов

        for modality, emb_list in embeddings.items():
            for emb in emb_list:
                emb_vector = emb["embedding"].numpy()
                if emb_vector.shape[0] < max_length:
                    emb_vector = np.pad(emb_vector, (0, max_length - emb_vector.shape[0]), mode='constant')

                combined_embeddings.append(emb_vector)
                combined_labels.append(modality)

            modality_embeddings = [
                np.pad(e["embedding"].numpy(), (0, max_length - e["embedding"].shape[0]), mode='constant')
                if e["embedding"].shape[0] < max_length else e["embedding"].numpy()
                for e in emb_list
            ]
            concat_embeddings.append(np.vstack(modality_embeddings))
            concat_labels.extend([modality] * len(modality_embeddings))
            concat_categories.extend([
                video_id_to_category[emb["video_id"]] for emb in emb_list
            ])

        concat_embeddings_all = np.concatenate(concat_embeddings, axis=0)

        # Применяем t-SNE к отдельным эмбеддингам и сконкатенированным
        tsne = TSNE(n_components=2, random_state=42)
        reduced_combined = tsne.fit_transform(np.array(combined_embeddings))
        reduced_concat = tsne.fit_transform(concat_embeddings_all)

        # Преобразуем результат в DataFrame
        tsne_combined_df = pd.DataFrame({
            'x': reduced_combined[:, 0],
            'y': reduced_combined[:, 1],
            'modality': combined_labels
        })

        tsne_concat_df = pd.DataFrame({
            'x': reduced_concat[:, 0],
            'y': reduced_concat[:, 1],
            'category': concat_categories
        })

        # Построение графиков
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"Результаты кластеризации", fontsize=20)

        unique_categories = tsne_concat_df['category'].unique()
        palette = {cat: sns.color_palette("hsv", len(unique_categories))[i]
                   for i, cat in enumerate(unique_categories)}

        # 1. Histplot для отдельных модальностей
        for modality in embeddings.keys():
            subset = tsne_combined_df[tsne_combined_df['modality'] == modality]
            sns.histplot(
                data=subset,
                x='x',
                y='y',
                ax=axs[0, 0],
                color=modality_colors[modality],
                bins=30
            )
        axs[0, 0].set_title("Histplot три модальности", fontsize=16)
        axs[0, 0].set_xlabel('x', fontsize=16)
        axs[0, 0].set_ylabel('y', fontsize=16)

        # 2. KDE Plot для отдельных модальностей
        for modality in embeddings.keys():
            subset = tsne_combined_df[tsne_combined_df['modality'] == modality]
            sns.kdeplot(
                x=subset['x'],
                y=subset['y'],
                ax=axs[0, 1],
                color=modality_colors[modality]
            )
        axs[0, 1].set_title("KDE Plot три модальности", fontsize=16)
        axs[0, 1].set_xlabel('x', fontsize=16)
        axs[0, 1].set_ylabel('y', fontsize=16)

        # 3. Histplot для сконкатенированных эмбеддингов
        sns.histplot(
            data=tsne_concat_df,
            x='x',
            y='y',
            hue='category',
            palette=palette,
            bins=100,
            ax=axs[1, 0],
            legend=False
        )
        axs[1, 0].set_title("Histplot метки", fontsize=16)
        axs[1, 0].set_xlabel('x', fontsize=16)
        axs[1, 0].set_ylabel('y', fontsize=16)

        # 4. KDE Plot для сконкатенированных эмбеддингов
        sns.kdeplot(
            data=tsne_concat_df,
            x='x',
            y='y',
            hue=None,
            color='purple',
            ax=axs[1, 1]
        )
        axs[1, 1].set_title("KDE Plot метки", fontsize=16)
        axs[1, 1].set_xlabel('x', fontsize=16)
        axs[1, 1].set_ylabel('y', fontsize=16)

        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], color='blue', lw=5, label='txt'),
            Line2D([0], [0], color='green', lw=5, label='aud'),
            Line2D([0], [0], color='red', lw=5, label='img'),
            Line2D([0], [0], color='purple', lw=5, label='merge'),
        ]

        # Добавляем легенду на график
        fig.legend(
            handles=legend_handles,
            loc='upper center',  # Легенда будет располагаться относительно верхнего центра
            bbox_to_anchor=(0.5, 0.96),  # Опускаем её ниже фигуры
            ncol=5,  # Число колонок в легенде
            frameon=False,  # Убираем рамку вокруг легенды (опционально)
            fontsize=16
        )

        # Легенда
        legend_handles = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=palette[cat], markersize=14,
                   label=cat, markeredgecolor='black', markeredgewidth=1)
            for cat in unique_categories
        ]

        # Добавляем легенду на график
        fig.legend(
            handles=legend_handles,
            loc='upper center',  # Легенда будет располагаться относительно верхнего центра
            bbox_to_anchor=(0.5, 0.1),  # Опускаем её ниже фигуры
            ncol=6,  # Число колонок в легенде
            frameon=False,  # Убираем рамку вокруг легенды (опционально)
            fontsize=10
        )

        # Сохранение графиков
        fig.tight_layout(rect=[0.06, 0.11, 0.94, 0.94])
        save_path = os.path.join(self.path_to_plots, "cluster.png")
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        log.info(f'График кластеризации: {save_path}')

    # def cluster_and_plot(self, embeddings: dict):
    #     """
    #     Функция для кластеризации эмбеддингов по категориям и подкатегориям.
    #     """
    #     log.info(f'Запуск процесса кластеризации KMeans по категориям и подкатегориям')
    #
    #     # Категории и подкатегории из метаданных
    #     categories = self.metadata['category'].unique()
    #     subcategories = self.metadata['tag'].unique()
    #
    #     # Словарь для хранения эмбеддингов по категориям и подкатегориям
    #     embeddings_by_category = {cat: [] for cat in categories}
    #     embeddings_by_subcategory = {subcat: [] for subcat in subcategories}
    #
    #     # Кодируем три модальности в один вектор
    #     for (txt, aud, img) in zip(embeddings['txt'], embeddings['aud'], embeddings['img']):
    #         latent = self.model.get_latent_space({
    #             'audio': aud['embedding'].unsqueeze(0),
    #             'text': txt['embedding'].unsqueeze(0),
    #             'vision': img['embedding'].unsqueeze(0)
    #         })
    #         latent = latent.squeeze(0)
    #         video_id = txt['video_id']
    #         category = self.metadata[self.metadata['video_id'] == video_id]['category'].values[0]
    #         subcategory = self.metadata[self.metadata['video_id'] == video_id]['tag'].values[0]
    #         embeddings_by_category[category].append(latent.numpy())
    #         embeddings_by_subcategory[subcategory].append(latent.numpy())
    #
    #     def perform_kmeans_and_plot_3d(embeddings_dict, title, axs_row):
    #         """
    #         Вспомогательная функция для выполнения кластеризации и построения 3D-графиков.
    #         """
    #         combined_embeddings = []
    #         labels = []
    #
    #         # Объединяем эмбеддинги в один массив
    #         for label, emb_list in embeddings_dict.items():
    #             if emb_list:  # Пропускаем, если эмбеддингов для категории/подкатегории нет
    #                 # Уменьшаем количество точек с помощью случайной выборки
    #                 sample_size = min(len(emb_list), 300)  # Ограничиваем до 500 точек на категорию
    #                 sampled_emb_list = random.sample(emb_list, sample_size)  # Выборка с помощью random.sample
    #                 combined_embeddings.extend(sampled_emb_list)
    #                 labels.extend([label] * len(sampled_emb_list))
    #
    #         combined_embeddings = np.array(combined_embeddings)
    #
    #         # Понижение размерности для визуализации (PCA)
    #         pca = PCA(n_components=3, random_state=42)
    #         reduced_embeddings = pca.fit_transform(combined_embeddings)
    #
    #         # Выполнение KMeans кластеризации
    #         num_clusters = 43
    #         kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    #         cluster_labels = kmeans.fit_predict(combined_embeddings)
    #
    #         # Преобразуем результат в DataFrame для удобства
    #         cluster_df = pd.DataFrame({
    #             'x': reduced_embeddings[:, 0],
    #             'y': reduced_embeddings[:, 1],
    #             'z': reduced_embeddings[:, 2],
    #             'label': labels,
    #             'cluster': cluster_labels
    #         })
    #
    #         # Построение 3D-графиков
    #         ax_label = fig.add_subplot(2, 2, 2 * axs_row + 1, projection='3d')
    #         scatter_label = ax_label.scatter(
    #             cluster_df['x'], cluster_df['y'], cluster_df['z'], c=cluster_df['label'].factorize()[0], cmap='viridis'
    #         )
    #         ax_label.set_title(f'{title} (по меткам)')
    #         fig.colorbar(scatter_label, ax=ax_label, shrink=0.5)
    #
    #         ax_cluster = fig.add_subplot(2, 2, 2 * axs_row + 2, projection='3d')
    #         scatter_cluster = ax_cluster.scatter(
    #             cluster_df['x'], cluster_df['y'], cluster_df['z'], c=cluster_df['cluster'], cmap='viridis'
    #         )
    #         ax_cluster.set_title(f'{title} (по кластерам)')
    #         fig.colorbar(scatter_cluster, ax=ax_cluster, shrink=0.5)
    #
    #     # Построение графиков для категорий и подкатегорий
    #     fig = plt.figure(figsize=(16, 12))
    #     fig.suptitle('Кластеризация эмбеддингов KMeans (3D)', fontsize=16)
    #
    #     perform_kmeans_and_plot_3d(embeddings_by_category, 'Категории', 0)
    #     perform_kmeans_and_plot_3d(embeddings_by_subcategory, 'Подкатегории', 1)
    #
    #     # Сохранение графиков
    #     save_path = os.path.join(self.path_to_plots, "kmeans_categories_subcategories_3d.png")
    #     fig.savefig(save_path, dpi=300)
    #     plt.close(fig)
    #
    #     log.info(f'3D-графики KMeans: {save_path}')
