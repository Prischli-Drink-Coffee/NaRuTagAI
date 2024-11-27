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

from src import project_path
from src.utils.custom_logging import setup_logging
from src.utils.create_dir import create_directories_if_not_exist

from time import time

log = setup_logging()


@dataclass
class ClusterCollector:
    data_folder: str
    path_to_plots: str
    loky_max_cpu_count: int
    palette: str

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

        self.reduce_and_plot_tsne(embeddings)

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

                if index == 100:
                    break

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

    def reduce_and_plot_tsne(self, embeddings: Dict[str, List[Dict[str, torch.Tensor]]]) -> None:
        # Словарь для цветов каждой модальности
        modality_colors = {
            'txt': 'blue',
            'aud': 'green',
            'img': 'red'
        }

        # Находим максимальную длину среди всех эмбеддингов
        max_length = max(
            emb["embedding"].shape[0] for emb_list in embeddings.values() for emb in emb_list
        )

        # Собираем все эмбеддинги и метки для каждой модальности
        combined_embeddings = []
        combined_labels = []
        concat_embeddings = []  # Для хранения сконкатенированных эмбеддингов
        concat_labels = []  # Метка для всех сконкатенированных эмбеддингов

        for modality, emb_list in embeddings.items():
            for emb in emb_list:
                emb_vector = emb["embedding"].numpy()
                # Дополняем до максимальной длины нулями, если необходимо
                if emb_vector.shape[0] < max_length:
                    emb_vector = np.pad(emb_vector, (0, max_length - emb_vector.shape[0]), mode='constant')

                combined_embeddings.append(emb_vector)
                combined_labels.append(modality)

            # Конкатенация всех эмбеддингов для данной модальности
            modality_embeddings = [
                np.pad(e["embedding"].numpy(), (0, max_length - e["embedding"].shape[0]), mode='constant')
                if e["embedding"].shape[0] < max_length else e["embedding"].numpy()
                for e in emb_list
            ]
            concat_embeddings.append(np.vstack(modality_embeddings))
            concat_labels.append(modality)

        # Объединяем сконкатенированные эмбеддинги всех модальностей
        concat_embeddings_all = np.concatenate(concat_embeddings, axis=0)
        concat_labels_all = ['concat'] * concat_embeddings_all.shape[0]

        # Убедимся, что данные имеют правильную форму
        combined_embeddings = np.array(combined_embeddings)  # Форматируем в 2D
        concat_embeddings_all = np.array(concat_embeddings_all)  # Форматируем в 2D

        # Применяем t-SNE к отдельным эмбеддингам и сконкатенированным
        tsne = TSNE(n_components=2, random_state=42)
        reduced_combined = tsne.fit_transform(combined_embeddings)
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
            'modality': concat_labels_all
        })

        # Построение графиков
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Результаты t-SNE для модальностей и сконкатенированных эмбеддингов", fontsize=14)

        # 1. Histplot для отдельных модальностей
        for modality in embeddings.keys():
            subset = tsne_combined_df[tsne_combined_df['modality'] == modality]
            sns.histplot(
                data=subset,
                x='x',
                y='y',
                ax=axs[0, 0],
                color=modality_colors[modality],
                bins=20
            )
        axs[0, 0].set_title("Histplot для модальностей")

        # 2. KDE Plot для отдельных модальностей
        for modality in embeddings.keys():
            subset = tsne_combined_df[tsne_combined_df['modality'] == modality]
            sns.kdeplot(
                x=subset['x'],
                y=subset['y'],
                ax=axs[0, 1],
                color=modality_colors[modality],
            )
        axs[0, 1].set_title("KDE Plot для модальностей")

        # 3. Histplot для сконкатенированных эмбеддингов
        sns.histplot(
            data=tsne_concat_df,
            x='x',
            y='y',
            ax=axs[1, 0],
            color='purple',
            bins=20
        )
        axs[1, 0].set_title("Histplot для сконкатенированных эмбеддингов")

        # 4. KDE Plot для сконкатенированных эмбеддингов
        sns.kdeplot(
            x=tsne_concat_df['x'],
            y=tsne_concat_df['y'],
            ax=axs[1, 1],
            color='purple'
        )
        axs[1, 1].set_title("KDE Plot для сконкатенированных эмбеддингов")

        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], color='blue', lw=2, label='txt'),
            Line2D([0], [0], color='green', lw=2, label='aud'),
            Line2D([0], [0], color='red', lw=2, label='img'),
            Line2D([0], [0], color='purple', lw=2, label='merge'),
        ]

        # Добавляем легенду на график
        fig.legend(
            handles=legend_handles,
            loc='upper center',  # Легенда будет располагаться относительно верхнего центра
            bbox_to_anchor=(0.5, 0.95),  # Опускаем её ниже фигуры
            ncol=4,  # Число колонок в легенде
            frameon=False  # Убираем рамку вокруг легенды (опционально)
        )

        # Сохранение графиков
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.path_to_plots, "tsne_separate_and_concat.png")
        fig.savefig(save_path, dpi=300)
        fig.clear()
        log.info(f'График t-SNE для модальностей и сконкатенированных эмбеддингов сохранён по пути: {save_path}')

