import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
from src.modelling.data2vec import Data2VecMultimodal, MultimodalLoss

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
        self.path_to_weights = Path(os.path.join(project_path, self.path_to_weights), 'data2vec')
        self.path_to_metrics_train = Path(os.path.join(project_path, self.path_to_metrics), 'data2vec')
        self.path_to_metrics_test = Path(os.path.join(project_path, self.path_to_metrics), 'data2vec')

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
        # Инициализируем модель
        self.model = Data2VecMultimodal(['audio', 'text', 'vision'],
                                        {'audio': 1280, 'text': 1024, 'vision': 1280},
                                        1792,
                                        0.995).to(self.device)

    def get_opt_crit_sh(self):
        # Определение функции потерь с учетом весов классов
        self.criterion = MultimodalLoss(beta=1.0, alpha=0.1)
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

                    if index == 10:
                        break

                    # log.info(f"text.shape: {texts.shape}")
                    # log.info(f"audio.shape: {audios.shape}")
                    # log.info(f"images.shape: {images.shape}")

                    # Подаем весь батч в модель
                    reconstructed, z_mean, z_log_var = self.model({
                        'audio': audios.mean(dim=1).mean(dim=1),
                        'text': texts.mean(dim=1),
                        'vision': images.mean(dim=1)
                    })
                    self.model.ema_step()

                    # Вычисляем loss для всего батча
                    loss = self.criterion({
                        'audio': audios.mean(dim=1).mean(dim=1),
                        'text': texts.mean(dim=1),
                        'vision': images.mean(dim=1)
                    }, reconstructed, z_mean, z_log_var)

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
            best_mse = float('inf')

            mse_values = []

            # Инициализация словарей для хранения данных по модальностям
            all_reconstructed = {modality: [] for modality in ['audio', 'text', 'vision']}
            all_original = {modality: [] for modality in ['audio', 'text', 'vision']}

            with torch.no_grad():
                with tqdm(total=len(self.valid_loader)) as pbar_valid:
                    for index, batch in enumerate(self.valid_loader):
                        # Распаковка данных
                        video_ids = batch["video_ids"]
                        images = batch["images"].to(self.device)
                        audios = batch["audios"].to(self.device)
                        texts = batch["texts"].to(self.device)

                        if index == 10:
                            break

                        # Валидируем модель
                        reconstructed, z_mean, z_log_var = self.model({
                            'audio': audios.mean(dim=1).mean(dim=1),
                            'text': texts.mean(dim=1),
                            'vision': images.mean(dim=1)
                        })

                        # Вычисляем loss для всего батча
                        loss = self.criterion({
                            'audio': audios.mean(dim=1).mean(dim=1),
                            'text': texts.mean(dim=1),
                            'vision': images.mean(dim=1)
                        }, reconstructed, z_mean, z_log_var)

                        # Добавляем данные для расчета метрик
                        for modality in reconstructed:
                            if modality not in all_reconstructed:
                                all_reconstructed[modality] = []
                            all_reconstructed[modality].append(reconstructed[modality].cpu().numpy())

                        # Собираем оригинальные данные для каждой модальности
                        all_original['audio'].append(audios.mean(dim=1).mean(dim=1).cpu().numpy())
                        all_original['text'].append(texts.mean(dim=1).cpu().numpy())
                        all_original['vision'].append(images.mean(dim=1).cpu().numpy())

                        valid_loss += loss.item() * self.batch_size

                        # Обновляем бар
                        pbar_valid.set_description(f"(Valid)")
                        pbar_valid.unit = " sample"
                        pbar_valid.set_postfix(epoch=(epoch + 1), loss=valid_loss / ((index + 1) * self.batch_size))
                        pbar_valid.update(1)

            epoch_train_loss = train_loss / len(self.train_dataset)
            epoch_valid_loss = valid_loss / len(self.valid_dataset)

            # Вычисляем метрики для каждой модальности
            metrics = {}
            for modality in ['audio', 'text', 'vision']:
                total_mse = 0.0
                total_mae = 0.0
                total_r2 = 0.0
                for (y_true, y_pred) in zip(all_original[modality], all_reconstructed[modality]):
                    total_mse += mean_squared_error(y_true, y_pred)
                    total_mae += mean_absolute_error(y_true, y_pred)
                    total_r2 += r2_score(y_true, y_pred)
                metrics[modality] = {'MSE': total_mse / len(all_original[modality]),
                                     'MAE': total_mae / len(all_original[modality]),
                                     'R2': total_r2 / len(all_original[modality])}
                log.info(f"Validation {modality.capitalize()} MSE: {total_mse:.4f},"
                         f" MAE: {total_mae:.4f}, R2: {total_r2:.4f}")

            # Вычисляем среднее значение MSE по всем модальностям
            average_mse = np.mean([metrics[modality]['MSE'] for modality in metrics])

            # Проверка на лучшее значение среднего mse по всем модальностям
            if average_mse < best_mse:
                best_mse = average_mse
                # Сохранение модели, если среднее mse лучше
                save_model(self.path_to_weights,
                           self.name_model,
                           self.model.state_dict(),
                           self.optimizer.state_dict(),
                           self.num_epochs)

            mse_values.append(average_mse)

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
                "avg_mse",
                self.date,
                self.name_model
            )

            log.info(
                f"\nEpoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_valid_loss}")

        log.info("Тренировка завершена!")

    # Функция для оценки модели на тестовом датасете
    def evaluate_model(self):
        total_loss = 0.0
        all_reconstructed = []
        all_original = []

        # Инициализация словарей для хранения данных по модальностям
        all_reconstructed = {modality: [] for modality in ['audio', 'text', 'vision']}
        all_original = {modality: [] for modality in ['audio', 'text', 'vision']}

        with torch.no_grad():
            with tqdm(total=len(self.test_loader)) as pbar_test:
                for index, batch in enumerate(self.test_loader):
                    # Распаковка данных
                    video_ids = batch["video_ids"]
                    images = batch["images"].to(self.device)
                    audios = batch["audios"].to(self.device)
                    texts = batch["texts"].to(self.device)

                    if index == 10:
                        break

                    # Тестируем модель
                    reconstructed, z_mean, z_log_var = self.model({
                        'audio': audios.mean(dim=1).mean(dim=1),
                        'text': texts.mean(dim=1),
                        'vision': images.mean(dim=1)
                    })

                    # Вычисляем loss для текущего батча
                    loss = self.criterion({
                        'audio': audios.mean(dim=1).mean(dim=1),
                        'text': texts.mean(dim=1),
                        'vision': images.mean(dim=1)
                    }, reconstructed, z_mean, z_log_var)

                    total_loss += loss.item() * self.batch_size

                    # Добавляем данные для расчета метрик
                    for modality in reconstructed:
                        if modality not in all_reconstructed:
                            all_reconstructed[modality] = []
                        all_reconstructed[modality].append(reconstructed[modality].cpu().numpy())

                    # Собираем оригинальные данные для каждой модальности
                    all_original['audio'].append(audios.mean(dim=1).mean(dim=1).cpu().numpy())
                    all_original['text'].append(texts.mean(dim=1).cpu().numpy())
                    all_original['vision'].append(images.mean(dim=1).cpu().numpy())

                    # Обновляем бар
                    pbar_test.set_description(f"(Test)")
                    pbar_test.unit = " sample"
                    pbar_test.set_postfix(loss=total_loss / ((index + 1) * self.batch_size))
                    pbar_test.update(1)

        # Вычисляем метрики для каждой модальности
        metrics = {}
        for modality in ['audio', 'text', 'vision']:
            total_mse = 0.0
            total_mae = 0.0
            total_r2 = 0.0
            for (y_true, y_pred) in zip(all_original[modality], all_reconstructed[modality]):
                total_mse += mean_squared_error(y_true, y_pred)
                total_mae += mean_absolute_error(y_true, y_pred)
                total_r2 += r2_score(y_true, y_pred)
            metrics[modality] = {'MSE': total_mse / len(all_original[modality]),
                                 'MAE': total_mae / len(all_original[modality]),
                                 'R2': total_r2 / len(all_original[modality])}
            log.info(f"Text {modality.capitalize()} MSE: {total_mse:.4f},"
                     f" MAE: {total_mae:.4f}, R2: {total_r2:.4f}")

        # Вычисляем среднее значение MSE по всем модальностям
        average_mse = np.mean([metrics[modality]['MSE'] for modality in metrics])

        # Сохраняем метрики
        save_metrics_test(self.path_to_metrics_test,
                          self.name_model,
                          average_mse,
                          'avg_mse',
                          None,
                          self.date)


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())

    train_config = config.get('TrainParamData2Vec', {})

    graduate = Graduate(path_to_data=env.__getattr__("DATA_PATH"),
                        path_to_weights=env.__getattr__("WEIGHTS_PATH"),
                        path_to_metrics=env.__getattr__("METRICS_PATH"),
                        **train_config)
    graduate.graduate()
