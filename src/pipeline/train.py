import os
from pathlib import Path
from datetime import datetime
from functools import partial
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel
)
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText

from src import project_path
from src.utils.custom_logging import setup_logging
from src.utils.metrics import compute_metrics
from src.utils.audio_process import audio_processor, create_audio_chunks_and_masks
from src.modelling.video_dataset import get_datasets, collate_fn
from src.utils.attention_module import AttentionLayer

log = setup_logging()

@dataclass
class GraduateText2Cat:
    path_to_data: str = "./data"
    path_to_weights: str = "./weights"
    path_to_metrics: str = "./metrics"

    name_model: str = "ru_bert"
    name_audio_model: str = 't5'
    task: str = "txt2cat"
    use_device: Optional[str] = None
    start_learning_rate: float = 0.0001
    pretrained_text_model: str = "ai-forever/ruBert-base"
    pretrained_audio_model: str = "microsoft/speecht5_asr"

    use_text_lematization: bool = False
    use_text_augmentation: bool = False
    use_fast_tokenizer: bool = False
    clean_up_tokenization_spaces: bool = True
    list_no_include_cat: Optional[List[str]] = field(default_factory=list)

    image_size: int = 256
    n_channels: int = 1
    max_length_generation: int = 30
    num_beams: int = 3
    num_workers: int = 4
    pin_memory: bool = False
    sample_duration: int = 64
    embedding_dim: int = 1024
    tag_similarity: float = 0.8
    name_optimizer: str = "Adam"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 17

    batch_size: int = 2
    early_stopping: bool = False
    num_epochs: int = 30

    train_dataset: Optional[torch.utils.data.Dataset] = None
    valid_dataset: Optional[torch.utils.data.Dataset] = None
    test_dataset: Optional[torch.utils.data.Dataset] = None
    train_loader: Optional[torch.utils.data.DataLoader] = None
    valid_loader: Optional[torch.utils.data.DataLoader] = None
    test_loader: Optional[torch.utils.data.DataLoader] = None

    classes: Optional[List[str]] = None
    total_samples: Optional[int] = None
    targets: Optional[List[int]] = None
    class_counts: Optional[List[int]] = None
    class_weights: Optional[torch.Tensor] = None
    criterion: Optional[torch.nn.Module] = None
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    transform: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    model: Optional[torch.nn.Module] = None
    checkpoint: Optional[dict] = None
    attention_layer: Optional[torch.nn.Module] = None
    date: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Настройка путей
        self.path_to_data = Path(os.path.join(project_path, self.path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights))
        self.path_to_metrics_train = Path(os.path.join(project_path, self.path_to_metrics))
        self.path_to_metrics_test = Path(os.path.join(project_path, self.path_to_metrics))

        # Настройка устройства (GPU/CPU)
        if self.use_device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.use_device)

        if self.device == torch.device("cpu"):
            self.pin_memory = False

        # Инициализация токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_text_model,
            use_fast=self.use_fast_tokenizer,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )

        # Создание необходимых директорий
        self.create_directories_if_not_exist([
            self.path_to_weights,
            self.path_to_metrics_train,
            self.path_to_metrics_test
        ])

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

    def __str__(self):
        log.info(f"Определенное устройство: {self.device}")
        log.info(f"Количество эпох обучения {self.num_epochs}")
        log.info(f"Размер пакета: {self.batch_size}")
        log.info(f"Выбранная модель: {self.name_model}")
        log.info(f"Данные загружены из директории: {self.path_to_data}")
        log.info(f"Выбранный оптимизатор: {self.name_optimizer}")
        return """"""

    # Функция для сохранения модели
    def save_model(self):
        # Сохраняем модель
        path = os.path.join(self.path_to_weights, f"{self.task}_{self.name_model}.pt")
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            path)

    def save_metrics_train(self,
                           train_loss_values,
                           valid_loss_values,
                           metric_values,
                           name_metric
                           ):
        metrics = {
            'train_loss': train_loss_values,
            'valid_loss': valid_loss_values,
            f'valid_{name_metric}': metric_values
        }
        # Сохранение метрик
        path = os.path.join(self.path_to_metrics_train,
                            f"train_{self.task}_{self.name_model}_{self.date.strftime('%Y-%m-%d_%H-%M-%S')}.pt")
        torch.save(metrics, path)

    def save_metrics_test(self,
                          metrics: dict):
        # Сохранение метрик
        path = os.path.join(self.path_to_metrics_test,
                            f"test_{self.task}_{self.name_model}_{self.date.strftime('%Y-%m-%d_%H-%M-%S')}.pt")
        torch.save(metrics, path)

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
                                       collate_fn=partial(collate_fn,
                                                          use_augmentation=self.use_text_augmentation,
                                                          use_lemmatization=self.use_text_lematization),
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
            
        self.tags = self.train_dataset.all_tags
        self.tags_ids = [self.train_dataset.tag2idx[tag] for tag in self.train_dataset.all_tags]

    def get_model(self):
        path = os.path.join(self.path_to_weights, f"{self.task}_{self.name_model}.pt")
        audio_path = os.path.join(self.path_to_weights, f"{self.task}_{self.name_audio_model}.pt")

        self.audio_processor = SpeechT5Processor.from_pretrained(self.pretrained_audio_model)
        self.audio_model = SpeechT5ForSpeechToText.from_pretrained(self.pretrained_audio_model)

        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_text_model).to(self.device)
        
        if os.path.isfile(path):            
            self.checkpoint = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                log.info("Веса успешно загружены")
            except Exception as ex:
                log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)
        
        if os.path.isfile(audio_path):            
            # Загрузите состояние модели и оптимизатора
            self.audio_checkpoint = torch.load(audio_path, map_location=self.device)
            try:
                self.model.load_state_dict(self.audio_checkpoint['model_state_dict'])
                log.info("Веса успешно загружены")
            except Exception as ex:
                log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)
        
        # Определяем слой внимания
        self.attention_layer = AttentionLayer(hidden_size=self.train_dataset.num_tags).to(self.device)
        self.classifier = nn.Linear(self.model.config.n_embd, self.train_dataset.num_tags).to(self.device)
    
    def encode_labels(self, labels):
        # Преобразование меток в one-hot encoding
        one_hot_labels = torch.zeros(len(labels), self.train_dataset.num_tags).to(self.device)
        for idx, label in enumerate(labels):
            one_hot_labels[idx, label] = 1.0  # Установка 1 для класса метки
        return one_hot_labels
    
    def decode_labels(self, matrix):
        indices_per_row = torch.nonzero(matrix, as_tuple=True)[1]
        labels= [self.train_dataset.idx2tag[idx.item()] for idx in indices_per_row]
        return labels
    
    def get_opt_crit_sh(self):
        # Определение функции потерь с учетом весов классов
        # self.criterion = FocalLoss(alpha=self.class_weights, gamma=2)
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.optimizer = optim.__dict__[f"{self.name_optimizer}"](self.model.parameters(), lr=self.start_learning_rate)
        path = os.path.join(self.path_to_weights, f"{self.task}_{self.name_model}.pt")
        
        # Если вы хотите также загрузить состояние оптимизатора, вы можете сделать это здесь:
        if os.path.isfile(path):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)

    def train_model(self):
        train_loss_values = []
        valid_loss_values = []
        iou_values = []
        best_iou = 0.0

        for epoch in range(self.num_epochs):
            epoch_train_loss = self.train_one_epoch(epoch)
            epoch_valid_loss, metrics = self.evaluate(epoch, set_="(Val)", loader=self.valid_loader)

            iou = metrics['IoU']
            if iou > best_iou:
                self.save_model()

            train_loss_values.append(epoch_train_loss)
            valid_loss_values.append(epoch_valid_loss)
            iou_values.append(best_iou)

            # Обновляем планировщик LR
            self.scheduler.step(epoch_valid_loss)

            log.info(f"\nEpoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_valid_loss}")
            log.info(f"IoU-score: {best_iou}")

        log.info("Тренировка завершена!")

        epoch_test_loss, metrics = self.evaluate(epoch, set_="(Test)", loader=self.test_loader)
    
    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = 0.0

        with tqdm(total=len(self.train_loader)) as pbar_train:
            for index, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                if index == 10:
                    break
                
                ### 1. РАСПАКОВКА БАТЧА
                images = batch["images"].to(self.device)
                audios = batch["audio"].to(self.device)
                audios_mask = batch["audio_mask"].to(self.device)

                ### 2. ПРЕДОБРАБОТКА ТАРГЕТА
                tags = torch.tensor(batch["tags_ids"], dtype=torch.long).to(self.device)
                tags = self.encode_labels(tags)

                ### 3. ТЕКСТОВАЯ МОДАЛьНОСТЬ
                titles = batch["title"].to(self.device)
                titles_mask = batch["title_attention_mask"].to(self.device)
                descriptions = batch["description"].to(self.device)
                descriptions_mask = batch["description_attention_mask"].to(self.device)

                # Объединение заголовков и описаний
                concatenated_input = torch.cat((titles, descriptions), dim=1)
                concatenated_mask = torch.cat((titles_mask, descriptions_mask), dim=1)

                # Достаем эмбеддинги текста (скрытое представление - CLS)
                # [batch_size, seq_len, hidden_size]
                text_outputs = self.model(input_ids=concatenated_input.long(), attention_mask=concatenated_mask, output_hidden_states=True)
                text_last_hidden_state = text_outputs.hidden_states[-1]
                
                # [batch_size, seq_len, hidden_size]
                text_embedding = text_last_hidden_state[:, 0, :]

                ### 4. АУДИАЛЬНАЯ МОДАЛьНОСТЬ

                chunk_size = 128_000
                # [batch_size, n_chunks, chunk_size]
                chunks, chunks_mask = create_audio_chunks_and_masks(audios=audios, audios_mask=audios_mask, chunk_size=chunk_size)

                last_hiddens = []
                for i in tqdm(range(chunks.shape[1])):
                    # Пример использования
                    audio_inputs = audio_processor(chunks[:, 0], chunks_mask[:, 0])  # Убедитесь, что здесь передаются тензоры
                    l = self.audio_model(**audio_inputs, output_hidden_states=True).encoder_last_hidden_state
                    audio_embeddings.append(l)

                # [n_chunks, batch, seq_len, hid_size]
                last_hiddens = torch.stack(last_hiddens)
                # [num_chunks, batch_size, hidden_size]
                audio_embeddings = torch.mean(torch.stack(last_hiddens), dim=2)
                # [batch_size, hidden_size]
                audio_embeddings = torch.mean(audio_embeddings, dim=0)
                
                ### 5. СОВМЕЩЕНИЕ МОДАЛЬНОСТЕЙ
                video_embeddings = torch.cat()
                #TODO: concat / mean audio + text embeds
                print(audio_embeddings.shape, text_embedding.shape)
                logits = self.classifier(...)

                # return logits, tags

                loss = self.criterion(logits, tags)

                train_loss += loss.item() * self.batch_size
                loss.backward()
                self.optimizer.step()

                # Обновляем бар
                pbar_train.set_description(f"(Train)")
                pbar_train.unit = " sample"
                pbar_train.set_postfix(epoch=(epoch + 1), loss=train_loss / ((index + 1) * self.batch_size))
                pbar_train.update(1)

        epoch_train_loss = train_loss / len(self.train_loader.dataset)
        return epoch_train_loss
    
    def evaluate(self, epoch, loader, set_='(Test)'):
        self.model.eval()
        test_loss = 0.0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            with tqdm(total=len(loader)) as pbar:
                for index, batch in enumerate(loader):
                    images = batch["images"].to(self.device)
                    audios = batch["audio"].to(self.device)
                    audios_mask = batch["audio_mask"].to(self.device)

                    titles = batch["title"].to(self.device)
                    titles_mask = batch["title_attention_mask"].to(self.device)
                    descriptions = batch["description"].to(self.device)
                    descriptions_mask = batch["description_attention_mask"].to(self.device)
                    
                    tags = torch.tensor(batch["tags_ids"], dtype=torch.long).to(self.device)
                    tags = self.encode_labels(tags)

                    self.optimizer.zero_grad()

                    # Объединение заголовков и описаний
                    concatenated_input = torch.cat((titles, descriptions), dim=1)
                    concatenated_mask = torch.cat((titles_mask, descriptions_mask), dim=1)

                    # Генерация тегов
                    # [batch_size, seq_len, hidden_size]
                    outputs = self.model(input_ids=concatenated_input.long(), attention_mask=concatenated_mask, output_hidden_states=True)
                    last_hidden_state = outputs.hidden_states[-1]

                    cls_embedding = last_hidden_state[:, 0, :]
                    logits = self.classifier(cls_embedding)

                    loss = self.criterion(logits, tags)

                    test_loss += loss.item() * self.batch_size

                    probabilities = F.softmax(logits, dim=1)
                    # probabilities = torch.sigmoid(logits)
                    predicted_tags = (probabilities > 0.9).int()

                    # собираем предсказанные и истинные тэги
                    all_predictions.append(predicted_tags.cpu())
                    all_labels.append(tags.cpu())

                    # Обновляем бар
                    pbar.set_description(set_)
                    pbar.unit = " sample"
                    pbar.set_postfix(epoch=(epoch + 1), loss=test_loss / ((index + 1) * self.batch_size))
                    log.info(f"label: {self.decode_labels(tags.cpu())}\n"
                             f"predicted: {self.decode_labels(predicted_tags.cpu())}\n")
                    pbar.update(1)
        
        all_predictions = torch.stack(all_predictions).squeeze(1)
        all_labels = torch.stack(all_labels).squeeze(1)

        metrics, text_metrics = compute_metrics(y_pred=all_predictions, y_true=all_labels, num_classes=self.train_dataset.num_tags)

        log.info(text_metrics)

        epoch_test_loss = test_loss / len(loader.dataset)
        return epoch_test_loss, metrics


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())

    train_config = config.get('TrainParam', {})

    graduate = GraduateText2Cat(path_to_data=env.__getattr__("DATA_PATH"),
                                path_to_weights=env.__getattr__("WEIGHTS_PATH"),
                                path_to_metrics=env.__getattr__("METRICS_PATH"),
                                **train_config)
    graduate.graduate()
