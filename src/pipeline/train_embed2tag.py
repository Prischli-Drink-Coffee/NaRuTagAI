# Импорт библиотек
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from functools import partial
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances as cosine
from src import project_path
from src.modelling.video_dataset import collate_fn, get_datasets
from src.utils.custom_logging import setup_logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.save_param import save_model, save_metrics_train, save_metrics_test
from transformers import (AutoConfig, AutoTokenizer, T5ForConditionalGeneration)
from dataclasses import dataclass\


# Логирование
log = setup_logging()


@dataclass
class GraduateEmbed2Tag:
    path_to_data: str = "./data"
    path_to_weights: str = "./weights"
    path_to_metrics: str = "./metrics"
    name_model: str = "Embed2TagV1",
    use_device: str = None,
    start_learning_rate: float = 0.0001,
    pretrained_weight_tag: str = "cointegrated/rut5-base-multitask",
    concat_embed_dim: int = 3840
    max_length_generation: int = 30,
    early_stopping: bool = False,
    num_beams: int = 3,
    batch_size: int = 2,
    num_workers: int = 4,
    pin_memory: bool = False,
    num_epochs: int = 30,
    tag_similarity: float = 0.8,
    name_optimizer: str = "Adam",
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 17

    def __post_init__(self):

        self.name_model = self.name_model if self.name_model else None

        self.path_to_data = Path(os.path.join(project_path, self.path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights), "embed2tag")
        self.path_to_matrics_train = Path(os.path.join(project_path, self.path_to_metrics), "embed2tag")
        self.path_to_matrics_test = Path(os.path.join(project_path, self.path_to_metrics), "embed2tag")

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_weight_tag,
                                                       legacy=True,
                                                       use_fast=True,
                                                       clean_up_tokenization_spaces=True)

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.total_samples = None
        self.targets = None
        self.criterion = None
        self.scheduler = None
        self.transform = None
        self.optimizer = None
        self.model = None
        self.checkpoint = None

        self.date = datetime.now()

        # Перемещение модели на GPU, если CUDA доступен
        if not self.use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.use_device == "cpu":
            self.device = torch.device("cpu")
        elif self.use_device == "cuda":
            self.device = torch.device("cuda")

        if self.device == "cpu":
            self.pin_memory = False

        # Создаем директории для сохранения весов и метрик
        self.create_directories_if_not_exist([self.path_to_weights,
                                              self.path_to_matrics_train,
                                              self.path_to_matrics_test])

    @staticmethod
    def create_directories_if_not_exist(directories: list):
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def graduate(self):
        # Получаем генераторы обучения, валидации и теста
        self.get_loaders()
        # Загружаем модель
        self.get_model()
        # Определяем оптимизатор, функцию потерь и планировщик
        self.get_opt_crit_sh()
        # Выводим информацию
        print(self.__str__())
        # Обучаем
        self.train_model()
        # Тестируем
        self.evaluate_model()

    def __str__(self):
        log.info(f"Определенное устройство: {self.device}")
        log.info(f"Количество эпох обучения {self.num_epochs}")
        log.info(f"Размер пакета: {self.batch_size}")
        log.info(f"Выбранная модель: {self.name_model}")
        log.info(f"Данные загружены из директории: {self.path_to_data}")
        log.info(f"Выбранный оптимизатор: {self.name_optimizer}")
        return """"""

    # Функция для загрузки данных
    def get_loaders(self):
        # Определяем класс video_dataset
        self.train_dataset, self.valid_dataset, self.test_dataset = get_datasets(self.path_to_data,
                                                                                 val_size=self.val_size,
                                                                                 test_size=self.test_size,
                                                                                 seed=self.seed)
        # Инициализируем DataLoader
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       collate_fn=partial(collate_fn),
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory)
        self.valid_loader = DataLoader(self.valid_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=False,
                                       collate_fn=collate_fn,
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      collate_fn=collate_fn,
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory)

    def get_model(self):
        path = os.path.join(self.path_to_weights, f"{self.name_model}.pt")
        if os.path.isfile(path):
            # Инициализируйте модель с конфигурацией
            self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_weight_tag).to(self.device)
            # Загрузите состояние модели и оптимизатора
            self.checkpoint = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                log.info("Веса успешно загружены")
            except Exception as ex:
                log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(self.pretrained_weight_tag).to(self.device)

    def get_opt_crit_sh(self):
        # Определение функции потерь с учетом весов классов
        self.optimizer = optim.__dict__[f"{self.name_optimizer}"](self.model.parameters(), lr=self.start_learning_rate)
        path = os.path.join(self.path_to_weights, f"{self.name_model}.pt")
        if os.path.isfile(path):
            # Если вы хотите также загрузить состояние оптимизатора, вы можете сделать это здесь:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        # Создание планировщика LR
        # ReduceLROnPlateau уменьшает скорость обучения, когда метрика перестает уменьшаться
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2)

    # Функция для обучения модели с валидацией
    def train_model(self):
        train_loss_values = []
        valid_loss_values = []

        for epoch in range(self.num_epochs):

            # Вычисление loss на тренировочном датасете
            self.model.train()
            train_loss = 0.0

            with tqdm(total=len(self.train_loader)) as pbar_train:
                for index, batch in enumerate(self.train_loader):

                    # Распаковка данных
                    video_ids = batch["video_ids"]
                    # images = batch["images"].to(self.device)
                    # audios = batch["audios"].to(self.device)
                    # texts = batch["texts"].to(self.device)
                    titles = batch["titles"]
                    descriptions = batch["descriptions"]
                    categories = batch["categories"]
                    category_ids = batch["category_ids"].to(self.device)
                    subcategories = batch["subcategories"]
                    subcategory_ids = batch["subcategory_ids"].to(self.device)

                    self.optimizer.zero_grad()

                    # Объединяем каждый элемент из titles с соответствующим элементом из descriptions
                    combined_list = [t + " " + d for t, d in zip(titles, descriptions)]

                    # Теперь передаем это в токенизатор
                    input_ids = self.tokenizer(
                        combined_list,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).input_ids.to(self.device)

                    labels = self.tokenizer(
                        subcategories,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length_generation,
                        return_tensors="pt"
                    ).input_ids.to(self.device)

                    # Генерация тегов
                    loss = self.model(input_ids=input_ids,
                                      labels=labels).loss

                    train_loss += loss.item() * self.batch_size
                    loss.backward()
                    self.optimizer.step()

                    # Обновляем бар
                    pbar_train.set_description(f"(Train)")
                    pbar_train.unit = " sample"
                    pbar_train.set_postfix(epoch=(epoch + 1), loss=train_loss / ((index + 1) * self.batch_size))
                    pbar_train.update(1)

            # Вычисление loss на валидационном датасете и метрик
            self.model.eval()
            valid_loss = 0.0
            cosine_distances = []

            with torch.no_grad():
                with tqdm(total=len(self.valid_loader)) as pbar_valid:
                    for index, batch in enumerate(self.valid_loader):

                        # Распаковка данных
                        video_ids = batch["video_ids"]
                        # images = batch["images"].to(self.device)
                        # audios = batch["audios"].to(self.device)
                        # texts = batch["texts"].to(self.device)
                        titles = batch["titles"]
                        descriptions = batch["descriptions"]
                        categories = batch["categories"]
                        category_ids = batch["category_ids"].to(self.device)
                        subcategories = batch["subcategories"]
                        subcategory_ids = batch["subcategory_ids"].to(self.device)

                        # Объединяем каждый элемент из titles с соответствующим элементом из descriptions
                        combined_list = [t + " " + d for t, d in zip(titles, descriptions)]

                        # Теперь передаем это в токенизатор
                        input_ids = self.tokenizer(
                            combined_list,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt"
                        ).input_ids.to(self.device)

                        labels = self.tokenizer(
                            subcategories,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length_generation,
                            return_tensors="pt"
                        ).input_ids.to(self.device)

                        # Генерация тегов
                        loss = self.model(input_ids=input_ids,
                                          labels=labels).loss

                        outputs = self.model.generate(
                            input_ids=input_ids,
                            max_length=self.max_length_generation,  # Максимальная длина сгенерированных тегов
                            num_beams=self.num_beams,  # Использование beam search
                            early_stopping=self.early_stopping
                        )
                        # Декодируем сгенерированную последовательность
                        decoded_predict_tags = [self.tokenizer.decode(output, skip_special_tokens=True) for output
                                                in outputs]
                        decoded_tags = [self.tokenizer.decode(output, skip_special_tokens=True) for output
                                        in labels]

                        valid_loss += loss.item() * self.batch_size

                        # Instead of encoding the entire list at once, loop through the list and encode each tag
                        output_embeddings = [self.tokenizer.encode(tag, return_tensors='pt').to(self.device) for tag in decoded_predict_tags]
                        target_embeddings = [self.tokenizer.encode(tag, return_tensors='pt').to(self.device) for tag in decoded_tags]

                        # Convert the tensors to float32 before calculating cosine similarity
                        output_embeddings = [embedding.float() for embedding in output_embeddings]
                        target_embeddings = [embedding.float() for embedding in target_embeddings]

                        # Now calculate cosine similarity for each pair of embeddings
                        cosine_distances = []
                        for output_embedding, target_embedding in zip(output_embeddings, target_embeddings):
                            # Pad the output and target embeddings to the same length
                            max_length = max(len(output_embedding[0]), len(target_embedding[0]))  # Find the max length

                            # Pad sequences with zeros to the max length
                            output_padded = F.pad(output_embedding[0], (0, max_length - len(output_embedding[0])), mode='constant', value=0)
                            target_padded = F.pad(target_embedding[0], (0, max_length - len(target_embedding[0])), mode='constant', value=0)

                            # Convert to float32 (required for cosine_similarity)
                            output_padded = output_padded.float()
                            target_padded = target_padded.float()

                            # Calculate cosine similarity
                            distance = F.cosine_similarity(output_padded.unsqueeze(0), target_padded.unsqueeze(0), dim=1).mean().item()
                            cosine_distances.append(distance)

                        # Обновляем бар
                        pbar_valid.set_description(f"(Valid)")
                        pbar_valid.unit = " sample"

                        pbar_valid.set_postfix(epoch=(epoch + 1), loss=valid_loss / ((index + 1) * self.batch_size))
                        log.info(f"label: {decoded_tags}\n"
                                 f"predicted: {decoded_predict_tags}\n")
                        pbar_valid.update(1)

            epoch_train_loss = train_loss / len(self.train_dataset.categories)
            epoch_valid_loss = valid_loss / len(self.valid_dataset.categories)

            save_model(self.path_to_weights,
                       self.name_model,
                       self.model.state_dict(),
                       self.optimizer.state_dict(),
                       self.num_epochs)

            # Сообщаем планировщику LR о текущей ошибке на валидационном наборе
            self.scheduler.step(epoch_valid_loss)

            # Добавление значений метрик в списки
            train_loss_values.append(epoch_train_loss)
            valid_loss_values.append(epoch_valid_loss)

            # Сохранение метрик
            save_metrics_train(
                self.path_to_matrics_train,
                train_loss_values,
                valid_loss_values,
                cosine_distances,
                "cosine_distance",
                self.date,
                self.name_model)

            log.info(
                f"\nEpoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_valid_loss}")

            count = sum(1 for distance in cosine_distances if distance > train_config['tag_similarity'])
            count_50 = sum(1 for distance in cosine_distances if distance > 0.7)
            count_30 = sum(1 for distance in cosine_distances if distance > 0.5)
            log.info(f"Count of cosine distances > {train_config['tag_similarity']}: {count}, Total: {len(cosine_distances)}\n"
                     f"Count of cosine distances > 50: {count_50}, Count of cosine distances > 30: {count_30}")

        log.info("Тренировка завершена!")

    # Функция для оценки модели на тестовом датасете
    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0
        cosine_distances = []

        with torch.no_grad():
            with tqdm(total=len(self.test_loader)) as pbar_test:
                for index, batch in enumerate(self.test_loader):

                    # Распаковка данных
                    video_ids = batch["video_ids"]
                    # images = batch["images"].to(self.device)
                    # audios = batch["audios"].to(self.device)
                    # texts = batch["texts"].to(self.device)
                    titles = batch["titles"]
                    descriptions = batch["descriptions"]
                    categories = batch["categories"]
                    category_ids = batch["category_ids"].to(self.device)
                    subcategories = batch["subcategories"]
                    subcategory_ids = batch["subcategory_ids"].to(self.device)

                    # Объединяем каждый элемент из titles с соответствующим элементом из descriptions
                    combined_list = [t + " " + d for t, d in zip(titles, descriptions)]

                    # Теперь передаем это в токенизатор
                    input_ids = self.tokenizer(
                        combined_list,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    ).input_ids.to(self.device)

                    labels = self.tokenizer(
                        subcategories,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length_generation,
                        return_tensors="pt"
                    ).input_ids.to(self.device)

                    # Генерация тегов
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        max_length=self.max_length_generation,  # Максимальная длина сгенерированных тегов
                        num_beams=self.num_beams,  # Использование beam search
                        early_stopping=self.early_stopping
                    )

                    # Декодируем сгенерированную последовательность
                    decoded_predict_tags = [self.tokenizer.decode(output, skip_special_tokens=True) for output
                                            in outputs]
                    decoded_tags = [self.tokenizer.decode(output, skip_special_tokens=True) for output
                                    in labels]

                    # Instead of encoding the entire list at once, loop through the list and encode each tag
                    output_embeddings = [self.tokenizer.encode(tag, return_tensors='pt').to(self.device) for tag in decoded_predict_tags]
                    target_embeddings = [self.tokenizer.encode(tag, return_tensors='pt').to(self.device) for tag in decoded_tags]

                    # Convert the tensors to float32 before calculating cosine similarity
                    output_embeddings = [embedding.float() for embedding in output_embeddings]
                    target_embeddings = [embedding.float() for embedding in target_embeddings]

                    # Now calculate cosine similarity for each pair of embeddings
                    cosine_distances = []
                    for output_embedding, target_embedding in zip(output_embeddings, target_embeddings):
                        # Pad the output and target embeddings to the same length
                        max_length = max(len(output_embedding[0]), len(target_embedding[0]))  # Find the max length

                        # Pad sequences with zeros to the max length
                        output_padded = F.pad(output_embedding[0], (0, max_length - len(output_embedding[0])), mode='constant', value=0)
                        target_padded = F.pad(target_embedding[0], (0, max_length - len(target_embedding[0])), mode='constant', value=0)

                        # Convert to float32 (required for cosine_similarity)
                        output_padded = output_padded.float()
                        target_padded = target_padded.float()

                        # Calculate cosine similarity
                        distance = F.cosine_similarity(output_padded.unsqueeze(0), target_padded.unsqueeze(0), dim=1).mean().item()
                        cosine_distances.append(distance)

                    count = sum(1 for distance in cosine_distances if distance > train_config['tag_similarity'])

                    # Обновляем бар
                    pbar_test.set_description(f"(Test)")
                    pbar_test.unit = " sample"
                    pbar_test.set_postfix(correct=count, total=len(cosine_distances))
                    pbar_test.update(1)

        save_metrics_test(
            self.path_to_matrics_test,
            self.name_model,
            count,
            'count_correct_distance',
            self.date)





if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())

    train_config = config.get('TrainParamEmbed2Tag', {})

    graduate = GraduateEmbed2Tag(path_to_data=env.__getattr__("DATA_PATH"),
                                 path_to_weights=env.__getattr__("WEIGHTS_PATH"),
                                 path_to_metrics=env.__getattr__("METRICS_PATH"),
                                 **train_config)
    graduate.graduate()
