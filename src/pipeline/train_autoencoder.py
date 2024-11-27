import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src import project_path
from src.utils.custom_logging import setup_logging
from pathlib import Path
from src.modelling.video_dataset import get_datasets, collate_fn
from datetime import datetime
from functools import partial
import torch.nn.functional as F
from src.utils.save_param import save_model, save_metrics_train, save_metrics_test
from src.utils.create_dir import create_directories_if_not_exist
from dataclasses import dataclass
from src.modelling.custom_model import CustomClassifierWithGAT, GraphReader

log = setup_logging()


@dataclass
class Graduate:
    path_to_data: str = "./data"
    path_to_weights: str = "./weights"
    path_to_metrics: str = "./metrics"
    name_model: str = "ru_bert"
    use_device: str = None
    start_learning_rate: float = 0.0001
    batch_size: int = 10
    num_workers: int = 4
    pin_memory: bool = False
    num_epochs: int = 10
    name_optimizer: str = "Adam"
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 17

    def __post_init__(self):
        self.date = datetime.now()
        self.name_model = self.name_model if self.name_model else None
        self.path_to_data = Path(os.path.join(project_path, self.path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights), 'autoencoder')
        self.path_to_metrics_train = Path(os.path.join(project_path, self.path_to_metrics), 'autoencoder')
        self.path_to_metrics_test = Path(os.path.join(project_path, self.path_to_metrics), 'autoencoder')

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.criterion = None
        self.scheduler = None
        self.transform = None
        self.optimizer = None
        self.model = None
        self.checkpoint = None

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
        create_directories_if_not_exist([self.path_to_weights,
                                         self.path_to_metrics_train,
                                         self.path_to_metrics_test])

    def graduate(self):
        # Получаем генераторы обучения, валидации и теста
        self.get_loaders()
        # Загружаем модель
        self.get_model()
        # Определяем оптимизатор, функцию потерь и планировщик
        self.get_opt_crit_sh()
        # Загружаем чекпоинт
        self.load_checkpoint()
        # Выводим информацию
        print(self.__str__())
        # Обучаем
        self.train_model()
        # Тестируем
        self.evaluate_model()

    def __str__(self):
        log.info(f"Определенное устройство: {self.use_device}")
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
                                       collate_fn=partial(collate_fn),
                                       num_workers=self.num_workers,
                                       pin_memory=self.pin_memory)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      collate_fn=partial(collate_fn),
                                      num_workers=self.num_workers,
                                      pin_memory=self.pin_memory)

    def get_model(self):
        pass
        # Инициализируем модель
        # self.model = CustomClassifierWithGAT(img_emb_shape=(1, 64, 1280),
        #                                      audio_emb_shape=(1, 1500, 1280),
        #                                      text_emb_shape=(1, 3, 1024),
        #                                      num_categories=self.num_classes,
        #                                      num_subcategories=self.num_subclasses).to(self.device)

    def get_opt_crit_sh(self):
        # Определение функции потерь с учетом весов классов
        self.criterion = None
        self.optimizer = optim.__dict__[f"{self.name_optimizer}"](self.model.parameters(), lr=self.start_learning_rate)
        # Создание планировщика LR
        # ReduceLROnPlateau уменьшает скорость обучения, когда метрика перестает уменьшаться
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)

    def load_checkpoint(self):
        path = os.path.join(self.path_to_weights, f"{self.name_model}.pt")
        try:
            if os.path.isfile(path):
                self.checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                try:
                    self.model.load_state_dict(self.checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
                    log.info("Веса успешно загружены")
                except Exception as ex:
                    log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)
            else:
                log.info("Не найден файл с моделью")
        except Exception as ex:
            log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)

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
                    images = batch["images"].to(self.device)
                    audios = batch["audios"].to(self.device)
                    texts = batch["texts"].to(self.device)

                    # log.info(f"text.shape: {texts.shape}")
                    # log.info(f"audio.shape: {audios.shape}")
                    # log.info(f"images.shape: {images.shape}")

                    # Обучаем модель
                    # category_logits, subcategory_logits = self.model(img_emb=images,
                    #                                                  audio_emb=audios,
                    #                                                  text_emb=texts,
                    #                                                  edge_index=self.edge_index,
                    #                                                  edge_attr=self.edge_attr)
                    # loss = (self.cat_criterion(category_logits, cat_labels_one_hot) +
                    #         self.sub_criterion(subcategory_logits, sub_labels_one_hot)) / 2

                    self.optimizer.zero_grad()

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
            best_mse = 0.0

            with torch.no_grad():
                with tqdm(total=len(self.valid_loader)) as pbar_valid:
                    for index, batch in enumerate(self.valid_loader):
                        # Распаковка данных
                        video_ids = batch["video_ids"]
                        images = batch["images"].to(self.device)
                        audios = batch["audios"].to(self.device)
                        texts = batch["texts"].to(self.device)

                        # Валидируем модель
                        # category_logits, subcategory_logits = self.model(img_emb=images,
                        #                                                  audio_emb=audios,
                        #                                                  text_emb=texts,
                        #                                                  edge_index=self.edge_index,
                        #                                                  edge_attr=self.edge_attr)
                        # loss = (self.cat_criterion(category_logits, cat_labels_one_hot) +
                        #         self.sub_criterion(subcategory_logits, sub_labels_one_hot)) / 2

                        valid_loss += loss.item() * self.batch_size

                        # _, cat_predicted = torch.max(category_logits, 1)
                        # _, sub_predicted = torch.max(subcategory_logits, 1)
                        # all_cat_predictions.extend(cat_predicted.cpu().numpy())
                        # all_sub_predictions.extend(sub_predicted.cpu().numpy())
                        # all_cat_labels.extend(cat_labels.cpu().numpy())
                        # all_sub_labels.extend(sub_labels.cpu().numpy())

                        # Обновляем бар
                        pbar_valid.set_description(f"(Valid)")
                        pbar_valid.unit = " sample"
                        pbar_valid.set_postfix(epoch=(epoch + 1), loss=valid_loss / ((index + 1) * self.batch_size))
                        pbar_valid.update(1)

            epoch_train_loss = train_loss / len(self.train_dataset.video_ids)
            epoch_valid_loss = valid_loss / len(self.valid_dataset.video_ids)


            log.info(f"Test MAE: {mae}, MSE: {mse}, R2: {r2}")

            # we want to save the model if the accuracy is the best
            if mse > best_mse:
                save_model(self.path_to_weights,
                           self.name_model,
                           self.model.state_dict(),
                           self.optimizer.state_dict(),
                           self.num_epochs)

            mse_values.append(mse)

            # Сообщаем планировщику LR о текущей ошибке на валидационном наборе
            self.scheduler.step(epoch_valid_loss)

            # Добавление значений метрик в списки
            train_loss_values.append(epoch_train_loss)
            valid_loss_values.append(epoch_valid_loss)

            # Сохранение метрик
            save_metrics_train(
                self.path_to_metrics_train,
                train_loss_values,
                valid_loss_values,
                mse_values,
                "mse",
                self.date,
                self.name_model
            )

            log.info(
                f"\nEpoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_valid_loss}")

        log.info("Тренировка завершена!")

    # Функция для оценки модели на тестовом датасете
    def evaluate_model(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            with tqdm(total=len(self.test_loader)) as pbar_test:
                for index, batch in enumerate(self.test_loader):

                    # Распаковка данных
                    video_ids = batch["video_ids"]
                    images = batch["images"].to(self.device)
                    audios = batch["audios"].to(self.device)
                    texts = batch["texts"].to(self.device)

                    # Тестируем модель
                    # category_logits, subcategory_logits = self.model(img_emb=images,
                    #                                                  audio_emb=audios,
                    #                                                  text_emb=texts,
                    #                                                  edge_index=self.edge_index,
                    #                                                  edge_attr=self.edge_attr)

                    # _, cat_predicted = torch.max(category_logits, 1)
                    # _, sub_predicted = torch.max(subcategory_logits, 1)
                    # all_cat_predictions.extend(cat_predicted.cpu().numpy())
                    # all_sub_predictions.extend(sub_predicted.cpu().numpy())
                    # all_cat_labels.extend(cat_labels.cpu().numpy())
                    # all_sub_labels.extend(sub_labels.cpu().numpy())

                    # Обновляем бар
                    pbar_test.set_description(f"(Test)")
                    pbar_test.unit = " sample"
                    pbar_test.set_postfix(correct=correct, total=total)
                    pbar_test.update(1)


        log.info(f"Test MAE: {mae}, MSE: {mse}, R2: {r2}")

        save_metrics_test(self.path_to_metrics_test,
                          self.name_model,
                          mse,
                          'mse',
                          None,
                          self.date)


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())

    train_config = config.get('TrainParamAutoencoder', {})

    graduate = Graduate(path_to_data=env.__getattr__("DATA_PATH"),
                        path_to_weights=env.__getattr__("WEIGHTS_PATH"),
                        path_to_metrics=env.__getattr__("METRICS_PATH"),
                        **train_config)
    graduate.graduate()
