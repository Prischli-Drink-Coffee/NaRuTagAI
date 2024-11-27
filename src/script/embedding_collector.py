import os
import re
import json

import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src import project_path
from src.utils.custom_logging import setup_logging
from FlagEmbedding import BGEM3FlagModel
from transformers import (AutoModel, CLIPImageProcessor,
                          AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline)
import torchaudio

from time import time

log = setup_logging()


@dataclass
class EmbeddingCollector:
    data_folder: str
    batch_size: int
    image_size: int
    image_model_path: str
    clip_image_processor: str
    text_model_path: str
    sample_rate: int
    audio_model_path: str

    def __post_init__(self):
        # Определяем путь к метадате
        self.metadata_path = os.path.join(project_path, self.data_folder, 'metadata.csv')
        self.metadata = pd.read_csv(self.metadata_path)
        # Очищаем метадату
        self.metadata = self._clean_metadata(self.metadata)
        # Определяем модели для обработки изображений, текста и аудио
        self.text_model: BGEM3FlagModel = None
        self.image_model: AutoModel = None
        self.image_processor: CLIPImageProcessor = None
        self.audio_model: AutoModelForSpeechSeq2Seq = None
        self.audio_processor: AutoProcessor = None
        # Создаем папку для хранения данных
        self.data_path = os.path.join(project_path, self.data_folder)
        self.create_dir_if_not_exists()
        # Определяем устройство для вычислений
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f'device: {self.device}')
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        log.info(f'torch_dtype: {self.torch_dtype}')
        # Получаем модели для обработки изображений, текста и аудио
        self._get_models_()

    @staticmethod
    def _clean_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Очищает метадату от пустых строк и дубликатов."""
        # Удаляем строки, где `tag`, `description` или `title` пустые или равны NaN/None
        cleaned_metadata = metadata.dropna(subset=['tag', 'description', 'title'])
        # Удаляем дубликаты
        cleaned_metadata = cleaned_metadata.drop_duplicates()
        log.info(f"Metadata cleaned: {len(metadata)} -> {len(cleaned_metadata)} rows")
        return cleaned_metadata

    def _get_models_(self):
        self.text_model = BGEM3FlagModel(self.text_model_path, use_fp16=True)
        self.image_model = AutoModel.from_pretrained(self.image_model_path,
                                                     torch_dtype=self.torch_dtype,
                                                     trust_remote_code=True).to(self.device).eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(self.clip_image_processor)
        self.audio_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.audio_model_path,
                                                                     low_cpu_mem_usage=True,
                                                                     use_safetensors=True).to(self.device)
        self.audio_processor = AutoProcessor.from_pretrained(self.audio_model_path)

    def text_pipeline(self, text: list[str]):
        return self.text_model.encode(text,
                                      max_length=8192,
                                      return_dense=True,
                                      return_colbert_vecs=True)

    def image_pipeline_batch(self, images: List[str]) -> torch.Tensor:
        """Обработка изображений батчами."""
        # Открываем и подготавливаем изображения
        batch_images = [Image.open(image).resize((self.image_size, self.image_size)) for image in images]
        input_pixels = self.image_processor(images=batch_images, return_tensors="pt").pixel_values.to(self.device)
        # Пропускаем через модель
        with torch.no_grad(), torch.amp.autocast(self.device, dtype=self.torch_dtype):
            embeddings = self.image_model.get_image_features(input_pixels)
        return embeddings

    def audio_pipeline(self, audio_path: str) -> Tuple[str, torch.Tensor]:
        """Обработка аудио и возврат текста и эмбеддинга."""
        # Загружаем аудио
        waveform, original_sample_rate = torchaudio.load(audio_path)
        # Приводим к ожидаемой частоте дискретизации
        if original_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        # Преобразуем в одномерный массив (Whisper ожидает вход [1, T])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Если стерео, конвертируем в моно
        waveform = waveform.squeeze(0).numpy()  # Преобразуем в массив numpy
        # Преобразуем аудио в input_features через процессор
        audio_features = self.audio_processor.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        input_features = audio_features["input_features"].to(self.device)
        # Генерация текста
        generate_kwargs = {
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "return_timestamps": True,
            "language": "russian"
        }
        output_tokens = self.audio_model.generate(input_features=input_features, **generate_kwargs)
        decoded_text = self.audio_processor.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        # Получение эмбеддинга аудио
        with torch.no_grad():
            encoder_outputs = self.audio_model.model.encoder(input_features=input_features)
            audio_embedding = encoder_outputs.last_hidden_state
        return decoded_text, audio_embedding

    def create_dir_if_not_exists(self):
        path = os.path.join(project_path, self.data_folder, 'embeddings')
        for p in [path,
                  os.path.join(path, 'images'),
                  os.path.join(path, 'texts'),
                  os.path.join(path, 'audios')]:
            if not os.path.exists(p):
                os.makedirs(p)

    def run(self):

        start = time()
        log.info(f'Запуск процесса эмбеддизации')

        # Получаем список обработанных видео
        processed_video_ids = {
            file.replace(".pt", "") for file in os.listdir(os.path.join(self.data_path, 'embeddings', 'texts'))
        }

        with tqdm(total=len(self.metadata)) as pbar_main:
            for index, row in self.metadata.iterrows():
                video_id = row['video_id']

                # Пропуск обработанных видео
                if video_id in processed_video_ids:
                    pbar_main.update(1)
                    continue

                title = row['title']
                description = row['description']
                path_audio = os.path.join(self.data_path, 'audio', f'{video_id}.mp3')
                path_frames = os.path.join(self.data_path, 'frames', f'{video_id}')

                # Эмбеддинг изображений
                path_images = [os.path.join(path_frames, p) for p in os.listdir(path_frames)]
                image_results = []
                for i in range(0, len(path_images), self.batch_size):
                    batch = path_images[i:i + self.batch_size]
                    image_results.append(self.image_pipeline_batch(batch))
                image_results = torch.cat(image_results, dim=0)

                # Эмбеддинг текста
                audio_text, audio_embedding = self.audio_pipeline(path_audio)
                text_results = self.text_pipeline([title, description, audio_text[0]])

                # Сохранение данных
                self.save_embeddings(video_id, image_results, audio_embedding, text_results)

                pbar_main.set_description(f'Collecting embeddings')
                pbar_main.update(1)

        log.info(f'Завершение процесса эмбеддизации')
        log.info(f'Время завершения: {time() - start} секунд')

    def save_embeddings(self, video_id, image_results, audio_embedding, text_results):
        path = os.path.join(project_path, self.data_folder, 'embeddings')
        path_images = os.path.join(path, 'images')
        path_texts = os.path.join(path, 'texts')
        path_audios = os.path.join(path, 'audios')
        image_path = os.path.join(path_images, f'{video_id}.pt')
        text_path = os.path.join(path_texts, f'{video_id}.pt')
        audio_path = os.path.join(path_audios, f'{video_id}.pt')
        torch.save(image_results, image_path)
        torch.save(text_results, text_path)
        torch.save(audio_embedding, audio_path)
