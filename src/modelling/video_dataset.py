import os
import numpy as np
import pandas as pd
import librosa
from typing import Tuple, List, Callable
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from src import path_to_project
from src.utils.custom_logging import setup_logging
from src.utils.seed import seed_everything
from src.utils.config_parser import ConfigParser
from src import path_to_config
from FlagEmbedding import BGEM3FlagModel


log = setup_logging()
config = ConfigParser.parse(path_to_config())
train_config = config.get('TrainParam', {})


# Обновляем функцию collate_fn, добавляя возможность аугментации
def collate_fn(batch):

    process_batch = {
        "video_ids": [],
        "images": [],
        "audios": [],
        "texts": [],
        "titles": [],
        "descriptions": [],
        "categories": [],
        "category_ids": [],
        "subcategories": [],
        "subcategory_ids": []
    }

    for i, sample in enumerate(batch):
        video_id = sample['video_id']
        image = sample['image']
        audio = sample['audio']
        text = sample['text']
        title = sample['title']
        description = sample['description']
        category = sample['category']
        category_id = sample['category_id']
        subcategory = sample['subcategory']
        subcategory_id = sample['subcategory_id']

        # Здесь можно добавить аугментацию
        text = torch.tensor(text['dense_vecs'], dtype=torch.float32)
        audio = audio.to(dtype=torch.float32)
        image = image.to(dtype=torch.float32)
        # ...

        # Добавляем данные в batch
        process_batch['video_ids'].append(video_id)
        process_batch['images'].append(image)
        process_batch['audios'].append(audio)
        process_batch['texts'].append(text)
        process_batch['titles'].append(title)
        process_batch['descriptions'].append(description)
        process_batch['categories'].append(category)
        process_batch['category_ids'].append(category_id)
        process_batch['subcategories'].append(subcategory)
        process_batch['subcategory_ids'].append(subcategory_id)

    # Преобразуем category_id в один тензор после итерации
    process_batch['category_ids'] = torch.tensor(process_batch['category_ids'])
    process_batch['subcategory_ids'] = torch.tensor(process_batch['subcategory_ids'])
    process_batch['images'] = torch.stack(process_batch['images'])
    process_batch['audios'] = torch.stack(process_batch['audios'])
    process_batch['texts'] = torch.stack(process_batch['texts'])

    return process_batch


def get_datasets(data_folder: str,
                 val_size: float = 0,
                 test_size: float = 0,
                 separator: str = 'SEP',
                 seed: int = 17,
                 categories: List[str] = None,
                 subcategories: List[str] = None
                 ):
    """
    Создает тренировочный, валидационный и тестовый датасеты на основе указанного CSV файла

    Args:
        data_folder (str): Путь к директории с данными
        val_size (float): Доля данных для валидации, в диапазоне [0, 1)
        test_size (float): Доля данных для тестирования, в диапазоне [0, 1)
        separator (str): Разделитель для тегов, если несколько тегов
        categories (List[str]): Список категорий для фильтрации данных
        subcategories (List[str]): Список подкатегорий для фильтрации данных

    Returns:
        Tuple: Датасеты для обучения, валидации (если есть) и тестирования
    """
    seed_everything(seed)

    assert 0 <= val_size < 1, "'val_size' should be in the range [0, 1)"
    assert 0 <= test_size < 1, "'test_size' should be in the range [0, 1)"

    if test_size == 0 and val_size > 0:
        test_size = val_size
        val_size = 0

    path = os.path.join(data_folder, 'metadata.csv')
    metadata = pd.read_csv(path)

    # Объединяем видео по категориям
    metadata = metadata.groupby(['video_id', 'title', 'description', 'tag'])[['category']].agg(
        separator.join).reset_index()

    # Фильтруем по выбранным категориям
    if categories is not None:
        metadata = metadata[metadata['category'].apply(lambda x: x not in categories)]

    if subcategories is not None:
        metadata = metadata[metadata['tag'].apply(lambda x: x not in subcategories)]

    # Разделение на тренировочный и тестовый датасет
    train_metadata, test_metadata = train_test_split(
        metadata,
        test_size=test_size,
        stratify=metadata['category'].values,
        random_state=seed
    )

    # Если валидационный датасет указан, делаем разделение
    if val_size > 0:
        train_metadata, val_metadata = train_test_split(
            train_metadata,
            test_size=val_size / (1 - test_size),
            stratify=train_metadata['category'].values,
            random_state=seed
        )
        val_dataset = VideoDataset(data_folder=data_folder,
                                   metadata=val_metadata,
                                   separator=separator,
                                   set_name="val")
    else:
        val_dataset = None

    # Создание датасетов
    train_dataset = VideoDataset(data_folder=data_folder, metadata=train_metadata, separator=separator)
    test_dataset = VideoDataset(data_folder=data_folder, metadata=test_metadata, separator=separator, set_name="test")

    if val_dataset is not None:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, test_dataset


class VideoDataset(Dataset):
    """
        Args:
            data_folder (str): Directory with all the video folders containing frames and audio.
            metadata (pd.DataFrame): DataFrame with columns ['video_id', 'description', 'tags'].
            separator (str): Separator used for splitting tags.
    """

    def __init__(
            self,
            data_folder: str,
            metadata: pd.DataFrame,
            set_name: str = 'train',
            separator: str = 'SEP',
    ):

        self.images_folder = os.path.join(path_to_project(), data_folder, "embeddings", "images")
        self.audios_folder = os.path.join(path_to_project(), data_folder, "embeddings", "audios")
        self.texts_folder = os.path.join(path_to_project(), data_folder, "embeddings", "texts")

        self.metadata = metadata
        self.separator = separator
        self.set_name = set_name

        self.seed_for_video = {k: v for k, v in zip(self.metadata.video_id.values,
                                                    torch.randint(low=0, high=100000, size=(len(self.metadata),))
                                                    )}
        self._build_target()

    def _build_target(self) -> None:
        """
            Builds label encodings for categories and tags
        """
        self.categories = self.metadata.category.values
        self.subcategories = self.metadata.tag.values

        # LabelEncoding for unique categories
        unique_categories = np.unique(self.categories)  # Уникальные категории
        self.cat2idx = {category: idx for idx, category in enumerate(unique_categories)}
        self.idx2cat = {idx: category for idx, category in enumerate(unique_categories)}
        self.num_categories = len(unique_categories)

        # LabelEncoding for unique categories
        unique_subcategories = np.unique(self.subcategories)  # Уникальные категории
        self.subcat2idx = {subcategory: idx for idx, subcategory in enumerate(unique_subcategories)}
        self.idx2subcat = {idx: subcategory for idx, subcategory in enumerate(unique_subcategories)}
        self.num_subcategories = len(unique_subcategories)

        log.info(f'''{self.set_name.upper()} INFO\nTotal: {self.num_categories} categories and {self.num_subcategories} subcategories''')

    def __len__(self) -> int:
        return len(self.metadata)

    def process_embeddings(self, video_id: str) -> torch.Tensor:
        """
            Processes embeddings for a given video ID
        """
        # fix seed on one video
        seed_everything(self.seed_for_video[video_id])

        images_embeddings_path = os.path.join(self.images_folder, f'{video_id}.pt')
        audios_embeddings_path = os.path.join(self.audios_folder, f'{video_id}.pt')
        texts_embeddings_path = os.path.join(self.texts_folder, f'{video_id}.pt')
        embeddings_path = [images_embeddings_path, audios_embeddings_path, texts_embeddings_path]

        emb_dir = {'images': None, 'audios': None, 'texts': None}

        for index, (path, (key, value)) in enumerate(zip(embeddings_path, emb_dir.items())):
            try:
                embedding = torch.load(path, weights_only=False)
                emb_dir[key] = embedding
            except Exception as e:
                log.error(f"Error processing embeddings {path}", exc_info=e)

        return emb_dir

    @staticmethod
    def truncate_string(text: str, max_length: int) -> str:
        try:
            if len(text) > max_length:
                return text[:max_length]
            return text
        except Exception as e:
            return ''

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        video_id = self.metadata['video_id'].values[idx]
        category = self.categories[idx]
        category_id = self.cat2idx[category]
        subcategory = self.subcategories[idx]
        subcategory_id = self.subcat2idx[subcategory]
        embeddings = self.process_embeddings(video_id)
        title = self.truncate_string(self.metadata['title'].values[idx], 10000)
        description = self.truncate_string(self.metadata['description'].values[idx],
                                           10000)

        return {
            "video_id": video_id,
            "image": embeddings['images'],
            "audio": embeddings['audios'],
            "text": embeddings['texts'],
            "title": title,
            "description": description,
            "category": category,
            "category_id": category_id,
            "subcategory": subcategory,
            "subcategory_id": subcategory_id
        }
