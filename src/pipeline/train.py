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
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from functools import partial
from src.utils.attention_module import AttentionLayer
from sklearn.metrics.pairwise import cosine_distances as cosine
import torch.nn.functional as F

log = setup_logging()


class GraduateText2Cat:

    def __init__(self,
                 path_to_data: str = "./data",
                 path_to_weights: str = "./weights",
                 path_to_metrics: str = "./metrics",
                 name_model: str = "ru_bert",
                 task: str = "txt2cat",
                 use_device: str = None,
                 start_learning_rate: float = 0.0001,
                 pretrained_weight_bert_class: str = "ai-forever/ruBert-base",
                 pretrained_weight_tag: str = "cointegrated/rut5-base-multitask",
                 use_text_lematization: bool = False,
                 use_text_augmentation: bool = False,
                 use_fast_tokenizer: bool = False,
                 clean_up_tokenization_spaces: bool = True,
                 list_no_include_cat: list = None,
                 image_size: int = 256,
                 max_length_generation: int = 30,
                 early_stopping: bool = False,
                 num_beams: int = 3,
                 batch_size: int = 2,
                 num_workers: int = 4,
                 pin_memory: bool = False,
                 sample_duration: int = 64,
                 n_channels: int = 1,
                 num_classes: int = 43,
                 num_epochs: int = 30,
                 embedding_dim: int = 1024,
                 tag_similarity: float = 0.8,
                 name_optimizer: str = "Adam",
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 seed: int = 17):

        self.name_model = name_model if name_model else None

        self.path_to_data = Path(os.path.join(project_path, path_to_data))
        self.path_to_weights = Path(os.path.join(project_path, path_to_weights))
        self.path_to_matrics_train = Path(os.path.join(project_path, path_to_metrics))
        self.path_to_matrics_test = Path(os.path.join(project_path, path_to_metrics))
        self.num_epochs = num_epochs
        self.size_img = image_size
        self.name_optimizer = name_optimizer
        self.num_classes = num_classes
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.sample_duration = sample_duration
        self.n_channels = n_channels
        self.pretrained_weight_bert_class = pretrained_weight_bert_class
        self.pretrained_weight_tag = pretrained_weight_tag
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_text_lematization = use_text_lematization
        self.use_text_augmentation = use_text_augmentation
        self.list_no_include_cat = list_no_include_cat
        self.task = task
        self.tag_similarity = tag_similarity
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_weight_tag,
                                                       legacy=True,
                                                       use_fast=use_fast_tokenizer,
                                                       clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        self.max_length_generation = max_length_generation
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.start_learning_rate = start_learning_rate

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.classes = None
        self.total_samples = None
        self.targets = None
        self.class_counts = None
        self.class_weights = None
        self.criterion = None
        self.scheduler = None
        self.transform = None
        self.optimizer = None
        self.model = None
        self.checkpoint = None
        self.attention_layer = None

        self.date = datetime.now()

        # Перемещение модели на GPU, если CUDA доступен
        if not use_device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif use_device == "cpu":
            self.device = torch.device("cpu")
        elif use_device == "cuda":
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
        # Получаем веса классов
        self.get_classes_weights()
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
        log.info(f"Размер изображений для обучения: {self.size_img}")
        log.info(f"Выбранная модель: {self.name_model}")
        log.info(f"Данные загружены из директории: {self.path_to_data}")
        log.info(f"Веса каждого класса: {self.class_weights}")
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
                           name_metric):
        metrics = {
            'train_loss': train_loss_values,
            'valid_loss': valid_loss_values,
            f'valid_{name_metric}': metric_values
        }
        # Сохранение метрик
        path = os.path.join(self.path_to_matrics_train,
                            f"train_{self.task}_{self.name_model}_{self.date.strftime('%Y-%m-%d_%H-%M-%S')}.pt")
        torch.save(metrics, path)

    def save_metrics_test(self,
                          metric,
                          name_metric,
                          class_acc_dir=None):

        if class_acc_dir is not None:
            metric = {
                f'{name_metric}_value': metric,
                'Acc_dir': class_acc_dir
            }
        else:
            metric = {
                f'{metric}_value': metric
            }
        # Сохранение метрик
        path = os.path.join(self.path_to_matrics_test,
                            f"test_{self.task}_{self.name_model}_{self.date.strftime('%Y-%m-%d_%H-%M-%S')}.pt")
        torch.save(metric, path)

    # Функция для загрузки данных
    def get_loaders(self):
        # Определяем класс video_dataset
        self.train_dataset, self.valid_dataset, self.test_dataset = get_datasets(self.path_to_data,
                                                                                 val_size=self.val_size,
                                                                                 test_size=self.test_size,
                                                                                 seed=self.seed,
                                                                                 categories=self.list_no_include_cat)
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
        self.classes = np.unique(self.train_dataset.categories)

    def get_model(self):
        path = os.path.join(self.path_to_weights, f"{self.task}_{self.name_model}.pt")
        if os.path.isfile(path):
            if self.task == 'txt2cat':
                # Загрузите конфигурацию модели (если требуется)
                bert_class_config = AutoConfig.from_pretrained(self.pretrained_weight_bert_class,
                                                               num_labels=self.num_classes)
                # Инициализируйте модель с конфигурацией
                self.model = AutoModelForSequenceClassification.from_config(bert_class_config).to(self.device)
            elif self.task == 'txt2tag':
                # Инициализируйте модель с конфигурацией
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_weight_tag,
                                                                   num_labels=self.num_classes).to(self.device)
            # Загрузите состояние модели и оптимизатора
            self.checkpoint = torch.load(path, map_location=self.device)
            try:
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                log.info("Веса успешно загружены")
            except Exception as ex:
                log.info("Ошибка загрузки предварительно обученной модели", exc_info=ex)
        else:
            if self.task == 'txt2cat':
                # Определяем ключевую модель для классификации категорий
                self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_weight_bert_class,
                                                                                num_labels=self.num_classes).to(
                    self.device)
            elif self.task == 'txt2tag':
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_weight_tag,
                                                                   num_labels=self.num_classes).to(
                    self.device)
        # Определяем слой внимания
        if self.task == 'txt2cat':
            self.attention_layer = AttentionLayer(hidden_size=self.num_classes).to(self.device)

    def get_classes_weights(self):
        # Получение категорий из тренировочного датасета
        self.targets = self.train_dataset.categories
        # Преобразование категорий в числовые индексы (если они не числа)
        if isinstance(self.targets[0], str):
            self.targets = np.array([self.train_dataset.cat2idx[cat] for cat in self.targets])
        self.total_samples = len(self.targets)
        # Получение количества примеров для каждого класса
        self.class_counts = np.bincount(self.targets)
        # Вычисление весов классов
        self.class_weights = torch.tensor([self.total_samples / count for count in self.class_counts],
                                          dtype=torch.float)
        self.class_weights = self.class_weights.to(self.device)

    def get_opt_crit_sh(self):
        # Определение функции потерь с учетом весов классов
        self.class_weights = self.class_weights if self.class_weights is not None else None
        if self.task == 'txt2cat':
            # self.criterion = FocalLoss(alpha=self.class_weights, gamma=2)
            # self.criterion = nn.NLLLoss(weight=self.class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = optim.__dict__[f"{self.name_optimizer}"](self.model.parameters(), lr=self.start_learning_rate)
        path = os.path.join(self.path_to_weights, f"{self.task}_{self.name_model}.pt")
        if os.path.isfile(path):
            # Если вы хотите также загрузить состояние оптимизатора, вы можете сделать это здесь:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        # Создание планировщика LR
        # ReduceLROnPlateau уменьшает скорость обучения, когда метрика перестает уменьшаться
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=2, verbose=True)

    # Функция для обучения модели с валидацией
    def train_model(self):
        train_loss_values = []
        valid_loss_values = []
        f1_values = []

        for epoch in range(self.num_epochs):

            # Вычисление loss на тренировочном датасете
            self.model.train()
            train_loss = 0.0

            with tqdm(total=len(self.train_loader)) as pbar_train:
                for index, batch in enumerate(self.train_loader):

                    # Распаковка данных
                    # images = batch["images"].to(self.device)
                    # audios = batch["audio"].to(self.device)
                    # audios_mask = batch["audio_mask"].to(self.device)
                    titles = batch["title"].to(self.device)
                    titles_mask = batch["title_attention_mask"].to(self.device)
                    descriptions = batch["description"].to(self.device)
                    descriptions_mask = batch["description_attention_mask"].to(self.device)
                    tags = batch["tags"].to(self.device)
                    tags_mask = batch["tags_attention_mask"].to(self.device)
                    # tags_ids = batch["tags_ids"]
                    labels = batch["category_id"].to(self.device)

                    self.optimizer.zero_grad()

                    if self.task == 'txt2cat':
                        outputs_titles = self.model(input_ids=titles,
                                                    attention_mask=titles_mask).logits
                        outputs_descriptions = self.model(input_ids=descriptions,
                                                          attention_mask=descriptions_mask).logits
                        outputs = self.attention_layer(outputs_titles, outputs_descriptions)
                        loss = self.criterion(outputs, labels)
                    elif self.task == 'txt2tag':
                        # Объединение заголовков и описаний
                        concatenated_input = torch.cat((titles, descriptions), dim=1)
                        concatenated_mask = torch.cat((titles_mask, descriptions_mask), dim=1)
                        # Генерация тегов
                        loss = self.model(input_ids=concatenated_input,
                                          attention_mask=concatenated_mask,
                                          labels=tags,
                                          decoder_attention_mask=tags_mask).loss

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
            best_f1 = 0.0
            all_predictions = []
            all_labels = []
            cosine_distances = []

            with torch.no_grad():
                with tqdm(total=len(self.valid_loader)) as pbar_valid:
                    for index, batch in enumerate(self.valid_loader):

                        # Распаковка данных
                        # images = batch["images"].to(self.device)
                        # audios = batch["audio"].to(self.device)
                        # audios_mask = batch["audio_mask"].to(self.device)
                        titles = batch["title"].to(self.device)
                        titles_mask = batch["title_attention_mask"].to(self.device)
                        descriptions = batch["description"].to(self.device)
                        descriptions_mask = batch["description_attention_mask"].to(self.device)
                        tags = batch["tags"].to(self.device)
                        tags_mask = batch["tags_attention_mask"].to(self.device)
                        # tags_ids = batch["tags_ids"]
                        labels = batch["category_id"].to(self.device)

                        if self.task == 'txt2cat':
                            outputs_titles = self.model(input_ids=titles,
                                                        attention_mask=titles_mask).logits
                            outputs_descriptions = self.model(input_ids=descriptions,
                                                              attention_mask=descriptions_mask).logits
                            outputs = self.attention_layer(outputs_titles, outputs_descriptions)
                            loss = self.criterion(outputs, labels)
                        elif self.task == 'txt2tag':
                            # Объединение заголовков и описаний
                            concatenated_input = torch.cat((titles, descriptions), dim=1)
                            concatenated_mask = torch.cat((titles_mask, descriptions_mask), dim=1)
                            # Генерация тегов
                            loss = self.model(input_ids=concatenated_input,
                                              attention_mask=concatenated_mask,
                                              labels=tags,
                                              decoder_attention_mask=tags_mask).loss
                            outputs = self.model.generate(
                                input_ids=concatenated_input,
                                attention_mask=concatenated_mask,
                                max_length=self.max_length_generation,  # Максимальная длина сгенерированных тегов
                                num_beams=self.num_beams,  # Использование beam search
                                early_stopping=self.early_stopping
                            )
                            # Декодируем сгенерированную последовательность
                            decoded_predict_tags = [self.tokenizer.decode(output, skip_special_tokens=True) for output
                                                    in outputs]
                            decoded_tags = [self.tokenizer.decode(output, skip_special_tokens=True) for output
                                            in tags]

                        valid_loss += loss.item() * self.batch_size

                        if self.task == 'txt2cat':
                            _, predicted = torch.max(outputs, 1)
                            all_predictions.extend(predicted.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                        elif self.task == 'txt2tag':
                            # Расчет косинусного расстояния
                            for output, target in zip(outputs, tags):
                                # Приводим к numpy
                                output_np = output.cpu().numpy()
                                target_np = target.cpu().numpy()
                                # Находим максимальную длину
                                max_length = max(len(output_np), len(target_np))
                                # Дополняем векторы до максимальной длины
                                output_padded = F.pad(torch.tensor(output_np), (0, max_length - len(output_np)),
                                                      mode='constant', value=0)
                                target_padded = F.pad(torch.tensor(target_np), (0, max_length - len(target_np)),
                                                      mode='constant', value=0)
                                # Теперь можно вычислять косинусное расстояние
                                distance = cosine(output_padded.numpy().reshape(1, -1),
                                                  target_padded.numpy().reshape(1, -1))[0][0]
                                cosine_distances.append(distance)

                        # Обновляем бар
                        pbar_valid.set_description(f"(Valid)")
                        pbar_valid.unit = " sample"
                        if self.task == 'txt2cat':
                            pbar_valid.set_postfix(epoch=(epoch + 1), loss=valid_loss / ((index + 1) * self.batch_size),
                                                   label=labels.cpu().numpy(), predicted=predicted.cpu().numpy())
                        elif self.task == 'txt2tag':
                            pbar_valid.set_postfix(epoch=(epoch + 1), loss=valid_loss / ((index + 1) * self.batch_size))
                            log.info(f"label: {decoded_tags}\n"
                                     f"predicted: {decoded_predict_tags}\n")
                        pbar_valid.update(1)

            epoch_train_loss = train_loss / len(self.train_dataset.categories)
            epoch_valid_loss = valid_loss / len(self.valid_dataset.categories)

            if self.task == 'txt2cat':
                accuracy = accuracy_score(all_labels, all_predictions)
                precision = precision_score(all_labels, all_predictions, average='weighted')
                recall = recall_score(all_labels, all_predictions, average='weighted')
                f1 = f1_score(all_labels, all_predictions, average='weighted')

                # we want to save the model if the accuracy is the best
                if f1 > best_f1:
                    self.save_model()

                f1_values.append(f1)

            if self.task == 'txt2tag':
                self.save_model()

            # Сообщаем планировщику LR о текущей ошибке на валидационном наборе
            self.scheduler.step(epoch_valid_loss)

            # Добавление значений метрик в списки
            train_loss_values.append(epoch_train_loss)
            valid_loss_values.append(epoch_valid_loss)

            # Сохранение метрик
            if self.task == 'txt2cat':
                self.save_metrics_train(train_loss_values,
                                        valid_loss_values,
                                        f1_values,
                                        "f1")
            elif self.task == 'txt2tag':
                self.save_metrics_train(train_loss_values,
                                        valid_loss_values,
                                        cosine_distances,
                                        "cosine_distance")

            log.info(
                f"\nEpoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_valid_loss}")

            if self.task == 'txt2cat':
                log.info(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")
            elif self.task == 'txt2tag':
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
        all_predictions = []
        all_labels = []
        cosine_distances = []

        # Initialize variables to track correct predictions for each class
        class_correct = [0] * self.class_counts
        class_total = [0] * self.class_counts

        with torch.no_grad():
            with tqdm(total=len(self.test_loader)) as pbar_test:
                for index, batch in enumerate(self.test_loader):

                    # Распаковка данных
                    # images = batch["images"].to(self.device)
                    # audios = batch["audio"].to(self.device)
                    # audios_mask = batch["audio_mask"].to(self.device)
                    titles = batch["title"].to(self.device)
                    titles_mask = batch["title_attention_mask"].to(self.device)
                    descriptions = batch["description"].to(self.device)
                    descriptions_mask = batch["description_attention_mask"].to(self.device)
                    tags = batch["tags"]
                    tags_mask = batch["tags_attention_mask"].to(self.device)
                    # tags_ids = batch["tags_ids"]
                    labels = batch["category_id"].to(self.device)

                    if self.task == 'txt2cat':
                        outputs_titles = self.model(input_ids=titles,
                                                    attention_mask=titles_mask).logits
                        outputs_descriptions = self.model(input_ids=descriptions,
                                                          attention_mask=descriptions_mask).logits
                        outputs = self.attention_layer(outputs_titles, outputs_descriptions)
                    elif self.task == 'txt2tag':
                        # Объединение заголовков и описаний
                        concatenated_input = torch.cat((titles, descriptions), dim=1)
                        concatenated_mask = torch.cat((titles_mask, descriptions_mask), dim=1)
                        # Генерация тегов
                        outputs = self.model.generate(
                            input_ids=concatenated_input,
                            attention_mask=concatenated_mask,
                            max_length=self.max_length_generation,  # Максимальная длина сгенерированных тегов
                            num_beams=self.num_beams,  # Использование beam search
                            early_stopping=self.early_stopping
                        )

                    if self.task == 'txt2cat':
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                        # Calculate class-wise correct predictions
                        for i in range(len(labels)):
                            label = labels[i].item()
                            class_correct[label] += (predicted[i] == labels[i]).item()
                            class_total[label] += 1

                    elif self.task == 'txt2tag':
                        # Расчет косинусного расстояния
                        for output, target in zip(outputs, tags):
                            # Приводим к numpy
                            output_np = output.cpu().numpy()
                            target_np = target.cpu().numpy()
                            # Находим максимальную длину
                            max_length = max(len(output_np), len(target_np))
                            # Дополняем векторы до максимальной длины
                            output_padded = F.pad(torch.tensor(output_np), (0, max_length - len(output_np)),
                                                  mode='constant', value=0)
                            target_padded = F.pad(torch.tensor(target_np), (0, max_length - len(target_np)),
                                                  mode='constant', value=0)
                            # Теперь можно вычислять косинусное расстояние
                            distance = cosine(output_padded.numpy().reshape(1, -1),
                                              target_padded.numpy().reshape(1, -1))[0][0]
                            cosine_distances.append(distance)
                        count = sum(1 for distance in cosine_distances if distance > train_config['tag_similarity'])

                    # Обновляем бар
                    pbar_test.set_description(f"(Test)")
                    pbar_test.unit = " sample"
                    if self.task == 'txt2cat':
                        pbar_test.set_postfix(correct=correct, total=total)
                    elif self.task == 'txt2tag':
                        pbar_test.set_postfix(correct=count, total=len(cosine_distances))
                    pbar_test.update(1)

        if self.task == 'txt2cat':
            accuracy = correct / len(self.test_dataset.categories)
            log.info(f"Test Accuracy: {accuracy}")

            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            log.info(f"Precision: {precision}, Recall: {recall}, F1-score: {f1}")

            class_acc_dir = {}
            # Print accuracy for each class
            for i in range(len(self.classes)):
                class_acc = 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0
                class_acc_dir[self.classes[i]] = class_acc
                log.info('Accuracy of %5s : %2d %%' % (self.classes[i], class_acc))

            self.save_metrics_test(f1,
                                   'f1',
                                   class_acc_dir=class_acc_dir)
        elif self.task == 'txt2tag':
            self.save_metrics_test(count,
                                   'count_correct_distance')


if __name__ == "__main__":
    from src import path_to_config
    from src.utils.config_parser import ConfigParser
    from env import Env

    env = Env()
    config = ConfigParser.parse(path_to_config())

    train_config = config.get('TrainParam', {})

    for name in train_config["name_models"]:
        graduate = GraduateText2Cat(path_to_data=env.__getattr__("DATA_PATH"),
                                    path_to_weights=env.__getattr__("WEIGHTS_PATH"),
                                    path_to_metrics=env.__getattr__("METRICS_PATH"),
                                    name_model=name,
                                    task=train_config['task'],
                                    use_device=train_config['use_device'],
                                    start_learning_rate=train_config['start_learning_rate'],
                                    pretrained_weight_bert_class=train_config['pretrained_weight_bert_class'],
                                    pretrained_weight_tag=train_config['pretrained_weight_tag'],
                                    use_text_lematization=train_config['use_text_lematization'],
                                    use_text_augmentation=train_config['use_text_augmentation'],
                                    use_fast_tokenizer=train_config['use_fast_tokenizer'],
                                    clean_up_tokenization_spaces=train_config['clean_up_tokenization_spaces'],
                                    list_no_include_cat=train_config['list_no_include_cat'],
                                    image_size=train_config["image_size"],
                                    max_length_generation=train_config['max_length_generation'],
                                    early_stopping=train_config['early_stopping'],
                                    num_beams=train_config['num_beams'],
                                    batch_size=train_config["batch_size"],
                                    num_workers=train_config["num_workers"],
                                    pin_memory=train_config["pin_memory"],
                                    sample_duration=train_config["sample_duration"],
                                    n_channels=train_config["n_channels"],
                                    num_classes=train_config["num_classes"],
                                    num_epochs=train_config["num_epochs"],
                                    embedding_dim=train_config["embedding_dim"],
                                    tag_similarity=train_config['tag_similarity'],
                                    name_optimizer=train_config["name_optimizer"],
                                    val_size=train_config["val_size"],
                                    test_size=train_config["test_size"],
                                    seed=train_config["seed"])
        graduate.graduate()