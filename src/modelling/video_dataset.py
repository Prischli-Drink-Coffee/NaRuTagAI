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
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


from src import path_to_project
from src.utils.custom_logging import setup_logging
from src.utils.seed import seed_everything
from src.utils.config_parser import ConfigParser
from src import path_to_config
from src.utils.string_filtration import process_single_text


log = setup_logging()
config = ConfigParser.parse(path_to_config())
train_config = config.get('Dataset', {})

text_tokenizer = AutoTokenizer.from_pretrained(train_config['pretrained_text_model'],
                                                legacy=True,
                                                use_fast=train_config['use_fast_tokenizer'],
                                                clean_up_tokenization_spaces=train_config['clean_up_tokenization_spaces'])

data_collator = DataCollatorForLanguageModeling(tokenizer=text_tokenizer, mlm=False, mlm_probability=0.2)

def select_frames(images: torch.Tensor) -> torch.Tensor:

    num_frames = images.size(0)  # Число кадров
    if num_frames <= train_config["sample_duration"]:
        return images

    # Оставляем каждый N-й кадр
    indices = torch.linspace(0, num_frames - 1, steps=train_config["sample_duration"]).long()
    return images[indices]


# Обновляем функцию collate_fn, добавляя возможность аугментации
def collate_fn(batch, use_lemmatization=False, use_augmentation=False):

    process_batch = {
        'video_id': [],
        'images': [],
        'audio': [],
        'audio_mask': [],
        'title': [],
        'title_attention_mask': [],
        'description': [],
        'description_attention_mask': [],
        'tags': [],
        'tags_attention_mask': [],
        'tags_ids': []
    }

    # Собираем все тексты из батча с применением аугментации, если нужно
    all_titles = [process_single_text(sample['title'], use_lemmatization, use_augmentation) for sample in batch]
    all_descriptions = [process_single_text(sample['description'], use_lemmatization, use_augmentation) for sample in batch]
    all_tags = [", ".join(sample['tags']) for sample in batch]

    # Токенизация всех текстов батча
    title_tokens = text_tokenizer(all_titles, padding=True, truncation=True, return_tensors="pt",
                                  max_length=train_config['max_title_length'])
    description_tokens = text_tokenizer(all_descriptions, padding=True, truncation=True, return_tensors="pt",
                                        max_length=train_config['max_description_length'])
    # tags_tokens = text_tokenizer(all_tags, padding=True, truncation=True, return_tensors="pt",
    #                              max_length=train_config['max_tag_length'])

    # Находим максимальную длину аудио данных в батче
    max_len = max([torch.tensor(sample['audio']).size(0) for sample in batch])

    for i, sample in enumerate(batch):
        video_ids = sample['video_id']
        tags = sample['tags']
        tags_ids = sample['tags_ids']
        audio = torch.tensor(sample['audio'], dtype=torch.float32)
        images = sample['images']

        # Отбираем только каждый 16 кадр
        images = select_frames(images)

        # Паддинг аудио данных до максимальной длины
        padded_audio = F.pad(audio, (0, max_len - audio.size(0)))

        # Создаем маски для аудио
        mask = torch.ones(audio.size(0), dtype=torch.float32)
        padded_mask = F.pad(mask, (0, max_len - mask.size(0)), value=0)

        # Добавляем данные в batch
        process_batch['video_id'].append(video_ids)
        process_batch['images'].append(images)
        process_batch['audio'].append(padded_audio)
        process_batch['audio_mask'].append(padded_mask)
        process_batch['title'].append(title_tokens['input_ids'][i])
        process_batch['title_attention_mask'].append(title_tokens['attention_mask'][i])
        process_batch['description'].append(description_tokens['input_ids'][i])
        process_batch['description_attention_mask'].append(description_tokens['attention_mask'][i])
        process_batch['tags'].append(tags)
        # process_batch['tags_attention_mask'].append(tags_tokens['attention_mask'][i])
        process_batch['tags_ids'].append(tags_ids)

    # Преобразуем остальные списки в тензоры с добавлением новой размерности для батча
    if train_config["use_random_token_mask"]:
        process_batch['title'] = data_collator(process_batch['title'])['input_ids']
        process_batch['description'] = data_collator(process_batch['description'])['input_ids']
        # process_batch['tags'] = data_collator(process_batch['tags'])['input_ids']
    else:
        process_batch['title'] = torch.stack(process_batch['title'])
        process_batch['description'] = torch.stack(process_batch['description'])
        # process_batch['tags'] = torch.stack(process_batch['tags'])
    
    process_batch['title_attention_mask'] = torch.stack(process_batch['title_attention_mask'])
    process_batch['description_attention_mask'] = torch.stack(process_batch['description_attention_mask'])
    # process_batch['tags_attention_mask'] = torch.stack(process_batch['tags_attention_mask'])
    # process_batch['tags_ids'] = torch.stack(process_batch['tags_ids'])
    process_batch['images'] = torch.stack(process_batch['images'])
    process_batch['audio'] = torch.stack(process_batch['audio'])
    process_batch['audio_mask'] = torch.stack(process_batch['audio_mask'])

    return process_batch


def get_datasets(data_folder: str,
                 val_size: float = 0,
                 test_size: float = 0,
                 separator: str = ', ',
                 seed: int = 17,
                 ):
    """
    Создает тренировочный, валидационный и тестовый датасеты на основе указанного CSV файла

    Args:
        data_folder (str): Путь к директории с данными
        val_size (float): Доля данных для валидации, в диапазоне [0, 1)
        test_size (float): Доля данных для тестирования, в диапазоне [0, 1)
        separator (str): Разделитель для тегов, если несколько тегов

    Returns:
        Tuple: Датасеты для обучения, валидации (если есть) и тестирования
    """
    seed_everything(seed)

    assert 0 <= val_size < 1, "'val_size' should be in the range [0, 1)"
    assert 0 <= test_size < 1, "'test_size' should be in the range [0, 1)"

    if test_size == 0 and val_size > 0:
        test_size = val_size
        val_size = 0

    # path = os.path.join(data_folder, 'metadata_stacked_filtered.csv')
    path = os.path.join(data_folder, 'metadata.csv')

    metadata = pd.read_csv(path)

    metadata.tag = metadata.tag.fillna('')

    # Объединяем теги по видео
    metadata = metadata.groupby(['video_id', 'title', 'description'])['tag'].agg(separator.join).reset_index()

    # Разделение на тренировочный и тестовый датасет
    train_metadata, test_metadata = train_test_split(
        metadata,
        test_size=test_size,
        random_state=seed
    )

    # Если валидационный датасет указан, делаем разделение
    if val_size > 0:
        train_metadata, val_metadata = train_test_split(
            train_metadata,
            test_size=val_size / (1 - test_size),
            random_state=seed
        )
        val_dataset = VideoDataset(data_folder=data_folder,
                                   metadata=val_metadata,
                                   separator=separator,
                                   set_name="val")
    else:
        val_dataset = None

    # Создание датасетов
    train_dataset = VideoDataset(data_folder=data_folder, metadata=train_metadata, separator=separator,
                                 set_name="train")
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

    def __init__(self,
                 data_folder: str,
                 metadata: pd.DataFrame,
                 set_name: str = 'train',
                 separator: str = ', ',
                 ):
        self.frames_folder = Path(path_to_project()) / Path(data_folder) / "frames"
        self.audio_folder = Path(path_to_project()) / Path(data_folder) / "audio"
        self.metadata = metadata
        self.separator = separator
        self.set_name = set_name

        self.seed_for_video = {k: v for k, v in zip(self.metadata.video_id.values,
                                                    torch.randint(low=0, high=100000, size=(len(self.metadata),))
                                                    )}

        if self.set_name == "train":
            self.transform = transforms.Compose([
                transforms.Resize(train_config["image_size"]),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if train_config["n_channels"] == 3 else transforms.Normalize(mean=[0.485], std=[0.229])
            ])

        elif self.set_name in ["val", "test"]:
            self.transform = transforms.Compose([
                transforms.Resize(train_config["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if train_config["n_channels"] == 3 else transforms.Normalize(mean=[0.485], std=[0.229])
            ])

        self._build_target()

    def _build_target(self) -> None:
        """
            Builds label encodings for tags
        """
        """
            Builds label encodings for tags
        """
        self.tags = self.metadata.tag.apply(lambda x: list(filter(bool, x.split(self.separator)))).values

        # LabelEncoding for tags
        self.all_tags = set()
        for tag in self.tags:
            self.all_tags |= set(tag)
        self.all_tags = list(self.all_tags)

        self.tag2idx = {tag: idx for idx, tag in enumerate(self.all_tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.num_tags = len(self.tag2idx)

        log.info(f'''{self.set_name.upper()} INFO\nTotal: {self.num_tags} TAGS''')

    def __len__(self) -> int:
        return len(self.metadata)

    def process_images(self, video_id: str) -> torch.Tensor:
        """
            Processes images for a given video ID
        """
        # fix seed on one video
        seed_everything(self.seed_for_video[video_id])

        if self.transform is None:
            return torch.empty(0)

        images = []
        images_path = self.frames_folder / video_id
        for image_file in images_path.iterdir():
            try:
                if train_config["n_channels"] == 1:
                    image = Image.open(image_file).convert('L')
                else:
                    image = Image.open(image_file).convert('RGB')
                image = self.transform(image)
                images.append(image)
            except Exception as e:
                log.error(f"Error processing image {image_file}: {e}")
        return torch.stack(images) if images else torch.empty(0)

    def process_audio(self, video_id: str, sampling_rate: int = 16_000):
        """
            Processes audio for a given video ID
        """
        audio_path = self.audio_folder / f'{video_id}.mp3'
        try:
            audio_array, _ = librosa.load(audio_path, sr=sampling_rate)
        except Exception as e:
            log.error(f"Error processing audio {audio_path}: {e}")
            audio_array = np.zeros(sampling_rate * 10)  # fallback to silence, 10 sec

        return audio_array

    @staticmethod
    def truncate_string(text: str, max_length: int) -> str:
        if len(text) > max_length:
            return text[:max_length]
        return text

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        video_id = self.metadata['video_id'].values[idx]

        tags = self.tags[idx]
        tags_id = [self.tag2idx[tag] for tag in tags]

        images = self.process_images(video_id)
        audio = self.process_audio(video_id)

        title = self.truncate_string(self.metadata['title'].values[idx], train_config['max_title_length'])
        description = self.truncate_string(self.metadata['description'].values[idx],
                                           train_config['max_description_length'])

        return {
            "video_id": video_id,
            "images": images,
            "audio": audio,
            "title": title,
            "description": description,
            "tags": tags,
            "tags_ids": tags_id,  # tags_ids[0]
        }
