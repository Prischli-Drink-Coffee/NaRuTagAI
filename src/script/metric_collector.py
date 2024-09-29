import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from config_file import load_config


class MetricsVisualizer:
    def __init__(self,
                 path_to_metrics_train: str = "./metrics_train",
                 path_to_metrics_test: str = "./metrics_test",
                 path_to_save_plots: str = None):
        self.script_path = load_config()["script_path"]
        if path_to_save_plots:
            os.makedirs(path_to_save_plots, exist_ok=True)
        self.path_to_save_plots = path_to_save_plots if path_to_save_plots is not None else self.script_path
        self.metrics_train_dir = os.path.join(self.script_path, path_to_metrics_train)
        self.metrics_test_dir = os.path.join(self.script_path, path_to_metrics_test)
        self.train_loss_values = {}
        self.valid_loss_values = {}
        self.f1_values_valid = {}
        self.f1_values_test = {}
        self.class_acc_dir_values = {}

        # Инициализируем сохранение графиков
        self.load_train_metrics()
        self.load_test_metrics()
        self.plot_metrics()

    def _load_metrics(self, directory, files_dict, key_name):
        for file in os.listdir(directory):
            if file.endswith(".pt"):
                metrics = torch.load(os.path.join(directory, file))
                model_name = file.replace('.pt', '')
                files_dict[model_name] = metrics[key_name]

    def load_train_metrics(self):
        self._load_metrics(self.metrics_train_dir, self.train_loss_values, 'train_loss')
        self._load_metrics(self.metrics_train_dir, self.valid_loss_values, 'valid_loss')
        self._load_metrics(self.metrics_train_dir, self.f1_values_valid, 'valid_f1')

    def load_test_metrics(self):
        self._load_metrics(self.metrics_test_dir, self.f1_values_test, 'f1_value')
        self._load_metrics(self.metrics_test_dir, self.class_acc_dir_values, 'Acc_dir')

    def plot_metrics(self):
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Сравнение категориальной кроссэнтропии для разных моделей на тренировке
        axs[0, 0].set_title('Функция потерь focal loss на тренировке', fontsize=18)
        for model, train_loss in self.train_loss_values.items():
            line, = axs[0, 0].plot(train_loss, label=model, linewidth=2)
            last_value = train_loss[-1]

        axs[0, 0].set_xlabel('Эпоха')
        axs[0, 0].set_ylabel('Значение функции потерь', fontsize=18)
        axs[0, 0].legend(fontsize=16)

        # Ошибка в коде, этого коэффициента не должно быть
        k = 1

        axs[0, 1].set_title('Функция потерь focal loss на валидации', fontsize=18)
        for model, valid_loss in self.valid_loss_values.items():
            # Умножаем все значения на оси y на константу k
            valid_loss_modified = [value * k for value in valid_loss]
            line, = axs[0, 1].plot(valid_loss_modified, label=model, linewidth=2)
            last_value = valid_loss_modified[-1]

        axs[0, 1].set_xlabel('Эпоха')
        axs[0, 1].set_ylabel('Значение функции потерь', fontsize=18)
        axs[0, 1].legend(fontsize=16)

        axs[1, 0].set_title('F1-мера на валидации', fontsize=18)
        for model, f1_valid in self.f1_values_valid.items():
            line, = axs[1, 0].plot(f1_valid, label=model, linewidth=2)
            last_value = f1_valid[-1]

        axs[1, 0].set_xlabel('Эпоха')
        axs[1, 0].set_ylabel('Значение метрики F1-мера', fontsize=18)
        axs[1, 0].legend(fontsize=16)
        # axs[1, 0].set_ylim([0.4, 0.90])

        # F1-мера для разных моделей на тесте
        axs[1, 1].set_title('F1-мера на тесте', fontsize=18)
        for model, f1_score in zip(self.f1_values_test.keys(), self.f1_values_test.values()):
            bar = axs[1, 1].bar(model, f1_score, label=model)

            axs[1, 1].text(bar.patches[0].get_x() + bar.patches[0].get_width() / 2., bar.patches[0].get_height()-0.1,
                           f'{f1_score: .3f}',
                           ha='center', va='bottom', fontsize=18, color='white')

        axs[1, 1].set_xlabel('Архитектура модели')
        axs[1, 1].set_ylabel('Значение метрики F1-мера', fontsize=18)

        # Повернуть названия моделей на оси x
        plt.setp(axs[1, 1].get_xticklabels(), fontsize=18)

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        path = os.path.join(self.path_to_save_plots, "PlotsMetrics.png")
        plt.savefig(path, dpi=500)
        plt.show()

        plt.figure(figsize=(8, 8))

        class_names = list(self.class_acc_dir_values[list(self.class_acc_dir_values.keys())[0]].keys())
        num_classes = len(class_names)

        for i, class_name in enumerate(class_names):
            plt.subplot(num_classes, 1, i + 1)

            for model, class_acc_dir in self.class_acc_dir_values.items():
                acc_values = [list(class_acc_dir.values())[i] for model in self.class_acc_dir_values.keys()]
                bars = plt.bar(model, acc_values[i], label=model)

                for bar, acc in zip(bars, acc_values):
                    plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height()-20, f'{acc:.2f}', ha='center',
                             va='bottom', rotation=90, fontsize=6, color='white')

            plt.xlabel('Модель')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy для класса: {class_name}')

            plt.xticks(rotation=90, fontsize=6)
            plt.subplots_adjust(hspace=0.5)

        plt.tight_layout()
        path = os.path.join(self.path_to_save_plots, "AccuracyForClass.png")
        plt.savefig(path, dpi=500)
        plt.show()
