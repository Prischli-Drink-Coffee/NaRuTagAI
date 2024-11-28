import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from env import Env
from __init__ import project_path
from src.utils.custom_logging import setup_logging


env = Env()
log = setup_logging()


class MetricsVisualizer:
    def __init__(self,
                 path_to_metrics: str = None,
                 path_to_save_plots: str = None,
                 part_sub: int = 13,
                 task: str = None):

        if path_to_metrics is not None:
            self.path_to_metrics = path_to_metrics
        else:
            self.path_to_metrics = os.path.join(project_path, env.__getattr__("METRICS_PATH"))

        if path_to_save_plots is not None:
            self.path_to_save_plots = path_to_save_plots
        else:
            self.path_to_save_plots = os.path.join(project_path, env.__getattr__("PLOTS_PATH"))

        if task == "main":
            self.path_to_metrics = os.path.join(self.path_to_metrics, "main")
            self.path_to_save_plots = os.path.join(self.path_to_save_plots, "train_main")
        elif task == "data2vec":
            self.path_to_metrics = os.path.join(self.path_to_metrics, "data2vec")
            self.path_to_save_plots = os.path.join(self.path_to_save_plots, "train_data2vec")
        elif task is None:
            self.path_to_metrics = self.path_to_metrics
            self.path_to_save_plots = self.path_to_metrics
        else:
            raise NotImplementedError("main or autoencoder or None")

        os.makedirs(self.path_to_metrics, exist_ok=True)
        os.makedirs(self.path_to_save_plots, exist_ok=True)

        self.part_sub = part_sub

        self.train_loss_values = {}
        self.valid_loss_values = {}
        self.metric_values_valid = {}
        self.metric_values_test = {}
        self.class_acc_dir_values = {}

        self.name_metric = None

    def run(self):
        # Инициализируем сохранение графиков
        log.info("Старт процесса построения графиков")
        self.load_train_metrics()
        self.load_test_metrics()
        self.plot_metrics()
        log.info("Графики построены и сохранены")

    def _load_metrics(self, directory, files_dict, key_name, test=False):
        if not os.listdir(directory):
            raise FileNotFoundError("Не найдены файлы с метриками, нечего строить")
        for file in os.listdir(directory):
            if test:
                if file.startswith("test") and file.endswith(".pt"):
                    log.info(f"Loading {file}")
                    metrics = torch.load(os.path.join(directory, file), weights_only=False)
                    model_name = file.replace('.pt', '')
                    if key_name != "value":
                        try:
                            files_dict[model_name] = metrics[key_name]
                        except KeyError:
                            self.class_acc_dir_values = None
                    else:
                        for key in metrics.keys():
                            if key.endswith('value'):
                                files_dict[model_name] = metrics[key]
                                self.name_metric = key
            else:
                if file.startswith("train") and file.endswith(".pt"):
                    log.info(f"Loading {file}")
                    metrics = torch.load(os.path.join(directory, file), weights_only=False)
                    model_name = file.replace('.pt', '')
                    if key_name != "value":
                        files_dict[model_name] = metrics[key_name]
                    else:
                        for key in metrics.keys():
                            if key.endswith('loss'):
                                files_dict[model_name] = metrics[key]
                                self.name_metric = key

    def load_train_metrics(self):
        self._load_metrics(self.path_to_metrics, self.train_loss_values, 'train_loss')
        self._load_metrics(self.path_to_metrics, self.valid_loss_values, 'valid_loss')
        self._load_metrics(self.path_to_metrics, self.metric_values_valid, 'value')

    def load_test_metrics(self):
        self._load_metrics(self.path_to_metrics, self.metric_values_test, 'value', True)
        self._load_metrics(self.path_to_metrics, self.class_acc_dir_values, 'Acc_dir', True)

    def plot_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # --- Потери на тренировке ---
        axs[0, 0].set_title('Функция потерь на тренировке', fontsize=12)
        for model, train_loss in self.train_loss_values.items():
            cleaned_model_name = model.replace("train_", "").replace("_test", "")
            axs[0, 0].plot(train_loss, label=cleaned_model_name, linewidth=2)

        axs[0, 0].set_xlabel('Эпоха')
        axs[0, 0].set_ylabel('Значение функции потерь', fontsize=12)
        axs[0, 0].legend(fontsize=12)

        # --- Потери на валидации ---
        axs[0, 1].set_title('Функция потерь на валидации', fontsize=12)
        for model, valid_loss in self.valid_loss_values.items():
            cleaned_model_name = model.replace("train_", "").replace("_test", "")
            axs[0, 1].plot(valid_loss, label=cleaned_model_name, linewidth=2)

        axs[0, 1].set_xlabel('Эпоха')
        axs[0, 1].set_ylabel('Значение функции потерь', fontsize=12)
        axs[0, 1].legend(fontsize=12)

        # --- F1-мера на валидации ---
        axs[1, 0].set_title(f'{self.name_metric} на валидации', fontsize=12)
        for model, f1_valid in self.metric_values_valid.items():
            cleaned_model_name = model.replace("train_", "").replace("_test", "")
            axs[1, 0].plot(f1_valid, label=cleaned_model_name, linewidth=2)

        axs[1, 0].set_xlabel('Эпоха')
        axs[1, 0].set_ylabel(f'Значение метрики {self.name_metric}', fontsize=12)
        axs[1, 0].legend(fontsize=12)

        # --- F1-мера на тесте ---
        axs[1, 1].set_title(f'{self.name_metric} на тесте', fontsize=12)
        for model, f1_score in self.metric_values_test.items():
            cleaned_model_name = model.replace("train_", "").replace("test_", "")
            bar = axs[1, 1].bar(cleaned_model_name, f1_score, label=cleaned_model_name)
            axs[1, 1].text(bar[0].get_x() + bar[0].get_width() / 2., bar[0].get_y() + bar[0].get_height() / 2.,
                           f'{f1_score:.3f}', ha='center', va='bottom', fontsize=18, color='white')

        axs[1, 1].set_xlabel('Название модели')
        axs[1, 1].set_ylabel(f'Значение метрики {self.name_metric}', fontsize=12)
        plt.setp(axs[1, 1].get_xticklabels(), fontsize=12)

        # Общая настройка
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        path = os.path.join(self.path_to_save_plots, "PlotsMetrics.png")
        plt.savefig(path, dpi=300)
        plt.close()

        part_sub = self.part_sub

        if self.class_acc_dir_values is not None:
            # --- Итерация по моделям для категорий и подкатегорий ---
            for model, class_acc_dir in self.class_acc_dir_values.items():
                cleaned_model_name = model.replace("train_", "").replace("test_", "")

                # Создаём новую фигуру для каждой модели
                # Используем более сбалансированное соотношение ширины и высоты
                fig, axes = plt.subplots(part_sub + 1, 1, figsize=(10, 5 * (part_sub + 1)))  # Высота пропорционально количеству сабплотов

                # --- График для категорий ---
                category_names = list(class_acc_dir['category'].keys())
                category_values = list(class_acc_dir['category'].values())
                ax = axes[0]

                # Ограничиваем длину текста категорий
                trimmed_category_names = [name if len(name) <= 10 else name[:10] + "..." for name in category_names]
                bars = ax.bar(trimmed_category_names, category_values, edgecolor='black', color='skyblue')
                ax.set_xlabel('Категории', fontsize=10)
                ax.set_ylabel('Accuracy', fontsize=10)
                ax.set_title('Accuracy по категориям', fontsize=12)

                for bar, acc in zip(bars, category_values):
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_y() + bar.get_height(),
                            f'{acc:.2f}', ha='center',
                            va='bottom', fontsize=4, color='black')
                ax.tick_params(axis='x', labelrotation=45, labelsize=4)

                # --- Графики для подкатегорий ---
                subclass_names = list(class_acc_dir['subcategory'].keys())
                subclass_values = list(class_acc_dir['subcategory'].values())

                # Разбиваем подкатегории на part_sub частей
                chunk_size = len(subclass_names) // part_sub
                for i in range(part_sub):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i != (part_sub - 1) else len(subclass_names)  # для последнего индекса захватываем все оставшиеся подкатегории
                    subcategory_chunk_names = subclass_names[start_idx:end_idx]
                    subcategory_chunk_names = [name if len(name) <= 20 else name[:20] + "..." for name in subcategory_chunk_names]
                    subcategory_chunk_values = subclass_values[start_idx:end_idx]

                    ax = axes[i + 1]  # Делаем subplots начиная с 1, так как 0 уже занят категориями

                    bars = ax.bar(subcategory_chunk_names, subcategory_chunk_values, color='salmon', edgecolor='black')
                    ax.set_xlabel('Подкатегории', fontsize=10)
                    ax.set_ylabel('Accuracy', fontsize=10)
                    ax.set_title(f'Accuracy по подкатегориям ({i + 1})', fontsize=12)

                    for bar, acc in zip(bars, subcategory_chunk_values):
                        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() - 0.02, f'{acc:.2f}', ha='center',
                                va='bottom', fontsize=4, color='black')
                    ax.set_xticks(range(len(subcategory_chunk_names)))
                    ax.tick_params(axis='x', labelrotation=90, labelsize=4)

                plt.tight_layout()
                path = os.path.join(self.path_to_save_plots, f"AccuracyForSubClass_{cleaned_model_name}.png")
                plt.savefig(path, dpi=300)
                plt.close()

            log.info("Сохранение графиков завершено")
