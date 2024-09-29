import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel, Wav2Vec2Processor, Wav2Vec2Model
import os
from src.utils.string_filtration import process_single_text
from src import project_path
from src.utils.custom_logging import setup_logging

log = setup_logging()


class VideoTagInference:
    def __init__(self, name_model='cointegrated/rut5-base-multitask', device=None):
        """
        Инициализация модели и токенизатора. По умолчанию используется модель 'cointegrated/rut5-base-multitask'.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name_model = name_model
        self.tokenizer = AutoTokenizer.from_pretrained(name_model, legacy=True, use_fast=False,
                                                       clean_up_tokenization_spaces=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name_model).to(self.device)

        # Загружаем кастомную модель
        self._load_custom_model_weights()

    def _load_custom_model_weights(self):
        """
        Загружает веса кастомной модели, если они есть в проекте.
        """
        try:
            if self.name_model == 'cointegrated/rut5-base-multitask':
                path_to_model = os.path.join(project_path, 'src/weights', 'txt2tag_t5_base_multitask.pt')
                log.info(f"Loading model from {path_to_model}")
            elif self.name_model == 'emelnov/keyT5_tags_custom':
                path_to_model = os.path.join(project_path, 'src/weights', 'txt2tag_keyT5_tags_custom.pt')
                log.info(f"Loading model from {path_to_model}")
            checkpoint = torch.load(path_to_model, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as ex:
            log.warning(f"Error loading model: {ex}")

    def _extract_image_features(self, image: torch.Tensor):
        """
        Извлекает признаки из изображения с помощью модели CLIP.
        """
        pass

    def _extract_audio_features(self, audio: torch.Tensor):
        """
        Извлекает признаки из аудио с помощью модели Wav2Vec2.
        """
        pass

    def preprocess_text(self, text: str, use_lemmatization: bool = False, max_length_token: int = 512):
        """
        Предобрабатывает текст (заголовок или описание) перед токенизацией, включая лемматизацию.
        """
        processed_text = process_single_text(text, use_lemmatization=use_lemmatization, use_augmentation=False)
        tokens = self.tokenizer(processed_text, padding=True, truncation=True, return_tensors="pt",
                                max_length=max_length_token).to(self.device)
        return tokens

    def inference(self, title: str = None, description: str = None, image: torch.Tensor = None, audio: torch.Tensor = None,
                  use_lemmatization: bool = False, max_length_token: int = 512, max_length_generation: int = 10,
                  num_beams: int = 5, num_return_sequences: int = 5, return_dict_in_generate: bool = True,
                  output_scores: bool = True, early_stopping: bool = True):
        """
        Основная функция предсказания, объединяющая текст, аудио и изображения для генерации тегов.
        """

        if title is None and description is None and image is None and audio is None:
            raise ValueError("At least one of title, description, image, or audio must be provided")

        # Предобработка текстовых данных
        title_tokens = self.preprocess_text(title, use_lemmatization=use_lemmatization,
                                            max_length_token=max_length_token) if title else None
        description_tokens = self.preprocess_text(description, use_lemmatization=use_lemmatization,
                                                  max_length_token=max_length_token) if description else None

        # Объединение токенов заголовка и описания
        if title_tokens is not None and description_tokens is not None:
            concatenated_input = torch.cat((title_tokens['input_ids'], description_tokens['input_ids']), dim=1)
            concatenated_mask = torch.cat((title_tokens['attention_mask'], description_tokens['attention_mask']), dim=1)
        elif title_tokens is not None:
            concatenated_input = title_tokens['input_ids']
            concatenated_mask = title_tokens['attention_mask']
        else:
            concatenated_input = description_tokens['input_ids']
            concatenated_mask = description_tokens['attention_mask']

        # Извлечение признаков из изображения и аудио, если они переданы
        # if image is not None:
        #     image_features = self._extract_image_features(image)
        #     concatenated_input = torch.cat((concatenated_input, image_features), dim=1)

        # if audio is not None:
        #     audio_features = self._extract_audio_features(audio)
        #     concatenated_input = torch.cat((concatenated_input, audio_features), dim=1)

        concatenated_input = concatenated_input.to(self.device)
        concatenated_mask = concatenated_mask.to(self.device)

        # Генерация тегов
        outputs = self.model.generate(
            input_ids=concatenated_input,
            attention_mask=concatenated_mask,
            max_length=max_length_generation,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            early_stopping=early_stopping
        )

        # Декодирование предсказанных тегов
        sequences = outputs.sequences
        scores = outputs.sequences_scores

        # Сортировка тегов по вероятности
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_sequences = sequences[sorted_indices]

        # Возвращаем список тегов
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in sorted_sequences]

