import os
import torch
from typing import List, Tuple
import numpy as np
import pandas as pd
# import faiss
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, CLIPImageProcessor,
                          AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModel, pipeline)
from FlagEmbedding import BGEM3FlagModel
from src import project_path
import torchaudio
from src.modelling.custom_model import CustomClassifier
# from src.script.embedding_generation import TagEmbeddingGeneration
# from src.utils.string_filtration import process_single_text
from src.utils.custom_logging import setup_logging
import torch.nn.functional as F
import json
import faiss
from pathlib import Path
from env import Env
from PIL import Image
from sentence_transformers import SentenceTransformer

env = Env()
log = setup_logging()


class VideoTagInference:
    def __init__(self,
                 path_model_t5='emelnov/keyT5_tags_custom',
                 name_model_t5='Embed2TagV1',
                 name_model_custom='Embed2CatSubcatv2',
                 image_model_path='microsoft/LLM2CLIP-Openai-B-16',
                 clip_image_processor='openai/clip-vit-base-patch16',
                 text_model_path='BAAI/bge-m3',
                 audio_model_path='openai/whisper-large-v3-turbo'):
        """
        Инициализация модели и токенизатора. По умолчанию используется модель 'emelnov/keyT5_tags_custom'.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path_model_t5 = path_model_t5
        self.name_model_t5 = name_model_t5
        self.name_model_custom = name_model_custom
        self.image_model_path = image_model_path
        self.clip_image_processor = clip_image_processor
        self.text_model_path = text_model_path
        self.audio_model_path = audio_model_path

        # Определяем модели для обработки изображений, текста и аудио
        self.text_model: BGEM3FlagModel = None
        self.image_model: AutoModel = None
        self.image_processor: CLIPImageProcessor = None
        self.audio_model: AutoModelForSpeechSeq2Seq = None
        self.audio_processor: AutoProcessor = None

        self.tokenizer_t5 = AutoTokenizer.from_pretrained(self.path_model_t5, legacy=True, use_fast=True,
                                                          clean_up_tokenization_spaces=True)
        self.model_t5 = AutoModelForSeq2SeqLM.from_pretrained(self.path_model_t5).to(self.device)

        self.model_custom = CustomClassifier()

        self.batch_size = 4
        self.image_size = 224
        self.sample_rate = 16000
        self.torch_dtype = torch.float16

        self.path_to_data = Path(os.path.join(project_path, env.__getattr__("DATA_PATH")))
        self.path_to_category_mapping = os.path.join(self.path_to_data, 'category_mapping.json')
        self.path_to_metadata = os.path.join(self.path_to_data, 'metadata.csv')
        self.metadata = pd.read_csv(self.path_to_metadata)

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

        # Инициализируем модели
        self._get_models_()

        # Загружаем кастомные модели
        self._load_custom_model_weights()

    def _load_custom_model_weights(self):
        """
        Загружает веса кастомной модели, если они есть в проекте.
        """
        try:
            if self.path_model_t5 == 'cointegrated/rut5-base-multitask':
                path_to_model = os.path.join(project_path, 'src/weights/embed2tag', f'{self.name_model_t5}.pt')
                log.info(f"Loading model from {path_to_model}")
            elif self.path_model_t5 == 'emelnov/keyT5_tags_custom':
                path_to_model = os.path.join(project_path, 'src/weights/embed2tag', f'{self.name_model_t5}.pt')
                log.info(f"Loading model from {path_to_model}")
            checkpoint = torch.load(path_to_model, map_location=self.device, weights_only=True)
            self.model_t5.load_state_dict(checkpoint['model_state_dict'])
            self.model_t5.to(self.device)
            path_to_model = os.path.join(project_path, 'src/weights/embed2catsubcat', f'{self.name_model_custom}.pt')
            checkpoint = torch.load(path_to_model, map_location=self.device, weights_only=True)
            self.model_custom.load_state_dict(checkpoint['model_state_dict'])
            self.model_custom.to(self.device)
            log.info(f"Loading model from {path_to_model}")
        except Exception as ex:
            log.warning(f"Error loading model: {ex}")

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

    @staticmethod
    # Функция для поиска ближайших тегов для каждой подкатегории
    def find_best_tags(subcategory_embeddings, topn=5):
        # Ищем топ-N тегов для подкатегории
        _, indices = faiss_index.search(subcategory_embeddings, topn)
        return indices

    @staticmethod
    def truncate_string(text: str, max_length: int) -> str:
        try:
            if len(text) > max_length:
                return text[:max_length]
            return text
        except Exception as e:
            return ''

    def predict(self, title: str = None, description: str = None, path_images: str = None, path_audio: str = None,
                use_lemmatization: bool = False, max_length_token: int = 512, max_length_generation: int = 15,
                num_beams: int = 3, num_return_sequences: int = 3, return_dict_in_generate: bool = True,
                output_scores: bool = False, early_stopping: bool = True, taxonomy_path='src/baseline/IAB_tags.csv',
                topn=3, embedding_dim=768, pretrained_model='DeepPavlov/rubert-base-cased-sentence'):
        """
        Основная функция предсказания, объединяющая текст, аудио и изображения для генерации тегов.
        """
        log.info('Preprocess title and description')
        if title is None and description is None and image is None and audio is None:
            raise ValueError("At least one of title, description, image, or audio must be provided")

        title = self.truncate_string(title, 512)
        description = self.truncate_string(description, 512)

        # Объединяем каждый элемент из titles с соответствующим элементом из descriptions
        combined_list = [t + " " + d for t, d in zip(title, description)]

        # Предобработка текстовых данных
        input_ids = self.tokenizer_t5(
            combined_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Эмбеддинг изображений
        path_images = [os.path.join(path_images, p) for p in os.listdir(path_images)]
        image_results = []
        for i in range(0, len(path_images), self.batch_size):
            batch = path_images[i:i + self.batch_size]
            image_results.append(self.image_pipeline_batch(batch))
        image_results = torch.cat(image_results, dim=0)

        # Эмбеддинг текста
        audio_text, audio_embedding = self.audio_pipeline(path_audio)
        text_results = self.text_pipeline([title, description, audio_text[0]])

        text = torch.tensor(text_results['dense_vecs'], dtype=torch.float32)
        audio = audio_embedding.to(dtype=torch.float32)
        image = image_results.to(dtype=torch.float32)

        # Обработка эмбеддингов
        audio = audio.mean(dim=0)
        audio = audio.mean(dim=0)
        image = image.mean(dim=0)
        text = text.mean(dim=0)

        audio = audio.to(self.device)
        image = image.to(self.device)
        text = text.to(self.device)

        # Добавляем батч размерность
        text = text.unsqueeze(0)
        image = image.unsqueeze(0)
        audio = audio.unsqueeze(0)

        log.info('End preprocessing')

        # --------------------------------------------------
        log.info("Run cat/sub prediction")

        # Инференс моделью
        category_logits, subcategory_logits = self.model_custom(img_emb=image,
                                                                audio_emb=audio,
                                                                text_emb=text)

        # Получаем вероятности для категорий и подкатегорий
        category_probs = F.softmax(category_logits, dim=-1)  # Применяем softmax для категорий
        subcategory_probs = F.softmax(subcategory_logits, dim=-1)  # Применяем softmax для подкатегорий

        # Преобразуем вероятности в одномерный тензор для упрощения
        category_probs = category_probs.squeeze()
        subcategory_probs = subcategory_probs.squeeze()

        # Загрузка словаря категорий и подкатегорий
        with open(self.path_to_category_mapping, 'r', encoding='utf-8') as f:
            category_mapping = json.load(f)

        # Шаг 1: Ищем три максимальных значения и их индексы вручную
        top3_category_indices = []
        top3_category_probs = []

        # Ищем 3 наибольшие вероятности
        for _ in range(2):
            max_prob, max_idx = category_probs.max(0)  # Ищем максимальное значение и его индекс
            top3_category_indices.append(max_idx.item())  # Добавляем индекс в список
            top3_category_probs.append(max_prob.item())  # Добавляем значение вероятности в список
            category_probs[max_idx] = -1  # Убираем максимальный элемент, чтобы найти следующий

        # Шаг 2: Извлекаем текстовые категории по индексам
        top3_categories = [self.idx2cat[idx] for idx in top3_category_indices]

        log.info(f"Top 3 categories: {top3_categories}")
        log.info(f"Top 3 probabilities: {top3_category_probs}")

        top_category_subcategories = {}

        # Для каждой категории выбираем топ-5 подкатегорий по вероятности
        for category in top3_categories:
            subcategories = category_mapping.get(category, [])  # Получаем список подкатегорий для этой категории
            if not subcategories:  # Если нет подкатегорий для этой категории, пропускаем её
                continue

            # Составляем список пар (подкатегория, вероятность)
            subcategory_probs_with_index = []

            # Для каждой подкатегории находим её индекс в общем списке подкатегорий через tag2idx
            for subcat in subcategories:
                # Получаем индекс подкатегории
                subcat_idx = self.subcat2idx.get(subcat)  # Получаем индекс подкатегории
                if subcat_idx is not None:  # Если подкатегория найдена в индексе
                    prob = subcategory_probs[subcat_idx].item()  # Извлекаем вероятность и преобразуем в число
                    subcategory_probs_with_index.append((subcat, prob, subcat_idx))
                else:
                    # Если подкатегория не найдена, можно присвоить 0.0 (или другие действия)
                    subcategory_probs_with_index.append((subcat, 0.0, None))

            # Сортируем подкатегории по вероятности и выбираем топ-5
            sorted_subcategory_probs = sorted(subcategory_probs_with_index, key=lambda x: x[1], reverse=True)[:2]

            # Записываем в итоговый результат: подкатегория и вероятность
            top_category_subcategories[category] = {
                self.idx2subcat.get(subcat_idx, subcat): prob  # Используем idx2tag для получения строки подкатегории
                for subcat, prob, subcat_idx in sorted_subcategory_probs
            }

        log.info(top_category_subcategories)

        # ---------------------------------------------------
        log.info("Run tag generation")
        # Генерация тегов
        outputs = self.model_t5.generate(
            input_ids=input_ids,
            max_length=max_length_generation,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=True,
            early_stopping=early_stopping
        )

        # Декодирование предсказанных тегов
        sequences = outputs.sequences

        log.info('End tag generation')

        # 1. Разделяем и убираем дубликаты предсказанных тегов
        predicted_tags = []
        for seq in sequences:
            decoded_sequence = self.tokenizer_t5.decode(seq, skip_special_tokens=True)
            tags = decoded_sequence.split(';')  # Разделяем по точке с запятой
            if seq == '':
                continue
            predicted_tags.extend(tags)  # Добавляем все теги в общий список

        # Убираем дубликаты
        predicted_tags = list(set(predicted_tags))
        log.info(f"Predicted tags: {predicted_tags}")

        # 2. Инициализация модели для эмбеддингов
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Используем модель для эмбеддингов

        # 3. Получаем эмбеддинги для предсказанных тегов
        predicted_tag_embeddings = model.encode(predicted_tags)  # Получаем эмбеддинги для всех предсказанных тегов
        predicted_tag_embeddings = np.array(predicted_tag_embeddings).astype('float32')  # Преобразуем в формат, понятный FAISS

        # 4. Настроим FAISS для поиска ближайших соседей
        embedding_dim = predicted_tag_embeddings.shape[1]  # Размерность эмбеддинга
        faiss_index = faiss.IndexFlatL2(embedding_dim)  # Инициализация FAISS индекса для L2 расстояния
        faiss_index.add(predicted_tag_embeddings)  # Добавляем эмбеддинги тегов в индекс FAISS

        # 5. Функция для получения эмбеддинга подкатегории
        def get_subcategory_embedding(subcategory):
            return model.encode([subcategory])[0]  # Это будет 1D numpy массив

        # 6. Для хранения финальных результатов
        final_results = {}

        # 7. Собираем все уникальные теги и распределяем их по категориям
        used_tags = set()  # Множество использованных тегов

        for category, subcategories in top_category_subcategories.items():
            final_results[category] = {}

            # Для каждой подкатегории распределяем теги
            for subcategory, subcategory_prob in subcategories.items():
                # Получаем эмбеддинг подкатегории
                subcategory_embedding = get_subcategory_embedding(subcategory)
                subcategory_embedding_np = np.array(subcategory_embedding).astype('float32')  # Преобразуем в numpy массив

                # Поиск ближайших тегов с использованием FAISS
                distances, indices = faiss_index.search(subcategory_embedding_np.reshape(1, -1), k=3)

                # Получаем теги по индексам, исключая уже использованные теги
                best_tags = []
                for idx in indices[0]:
                    tag = predicted_tags[idx]
                    if tag not in used_tags:  # Если тег еще не использован
                        best_tags.append(tag)
                        used_tags.add(tag)  # Отмечаем тег как использованный

                # Если тегов осталось больше, чем нужно, обрезаем
                final_results[category][subcategory] = best_tags[:5]  # Ограничиваем топ-5 тегами

        log.info(f"Final result: {final_results}")

        return final_results
