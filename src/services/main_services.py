import os
import cv2
import shutil
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
# from src.database.models import Predict, Video, VideoInference, Inference
from src.modelling.video_processing import VideoProcessor
from src.utils.custom_logging import setup_logging
from src.script.predict import VideoTagInference

log = setup_logging()


# class VideoAudioProcessor:
#     """
#     Класс для обработки видео и аудио.
#     Включает функции для обработки кадров из видео и аудио-дорожек.
#     """
#
#     def __init__(self, frame_size: Tuple[int, int] = (224, 224), num_frames: int = 64, audio_sample_rate: int = 16000):
#         """
#         Инициализирует параметры обработки.
#
#         :param frame_size: Размер каждого кадра (по умолчанию 224x224).
#         :param num_frames: Количество кадров для извлечения.
#         :param audio_sample_rate: Частота дискретизации аудио.
#         """
#         self.frame_size = frame_size
#         self.num_frames = num_frames
#         self.audio_sample_rate = audio_sample_rate
#
#     def process_frames_from_folder(self, frame_path: str) -> torch.Tensor:
#         """
#         Читает кадры из папки, масштабирует до нужного размера и возвращает их как тензор.
#
#         :param frame_path: Путь к папке с кадрами.
#         :return: Тензор кадров с размером (num_frames, 3, 224, 224).
#         """
#         frames = []
#         for i in range(self.num_frames):
#             frame_file = os.path.join(frame_path, f"frame_{i}.png")
#             if os.path.exists(frame_file):
#                 frame = cv2.imread(frame_file)
#                 frame = cv2.resize(frame, self.frame_size)
#                 frames.append(frame)
#             else:
#                 log.warning(f"Frame {i} not found at {frame_file}")
#                 break
#
#         # Если кадров меньше, чем необходимо, дублируем последние
#         while len(frames) < self.num_frames:
#             frames.append(frames[-1])
#
#         # Преобразуем в тензор
#         frames_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
#         return frames_tensor
#
#     def process_audio(self, audio_path: str) -> torch.Tensor:
#         """
#         Извлекает аудио из файла и приводит его к нужной частоте дискретизации и длине.
#
#         :param audio_path: Путь к аудиофайлу.
#         :return: Тензор аудио.
#         """
#         audio, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
#
#         # Если длина аудио меньше, дополняем нулями, иначе обрезаем
#         if len(audio) < self.audio_sample_rate:
#             audio = np.pad(audio, (0, max(0, self.audio_sample_rate - len(audio))), 'constant')
#         else:
#             audio = audio[:self.audio_sample_rate]
#
#         # Преобразуем аудио в тензор
#         audio_tensor = torch.tensor(audio, dtype=torch.float32)
#         return audio_tensor
#
#     def process_video_and_audio(self, frame_path: str, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Обрабатывает кадры и аудио и возвращает их в виде тензоров.
#
#         :param frame_path: Путь к кадрам видео.
#         :param audio_path: Путь к аудио файлу.
#         :return: Тензоры кадров и аудио.
#         """
#         frames_tensor = self.process_frames_from_folder(frame_path)
#         audio_tensor = self.process_audio(audio_path)
#         return frames_tensor, audio_tensor


class VideoInferencePipeline:
    """
    Класс для управления процессом обработки видео, инференса и удаления временных файлов.
    """

    def __init__(self, url: str, temp_path: str):
        """
        Инициализирует параметры инференса.

        :param predict: Объект модели Predict с данными видео.
        :param temp_path: Временная папка для хранения видео и аудио файлов.
        """
        self.predict = url
        self.temp_path = temp_path
        self.audio_folder = Path(self.temp_path) / "audio"
        self.frame_folder = Path(self.temp_path) / "frames"
        self.processor = VideoProcessor(
            url=self.predict,
            data_folder=self.temp_path,
            min_duration=0,
            max_duration=60,
            quality='worst',
            n_frames=64,
            frame_dimensions=f"{(224, 224)}",
        )

    def run(self):
        """
        Выполняет полный процесс обработки видео, включая инференс и очистку временной папки.

        :return: Список предсказанных тегов и категорий.
        """
        try:
            # Обрабатываем видео
            title, description, video_id = self.processor.process_video()
            if not video_id:
                raise Exception("Video not found")

            frame_path = self.frame_folder / f"{video_id}"
            audio_path = self.audio_folder / f"{video_id}.mp3"

            # # Создаем запись видео в базе данных
            # video_services.create_video(Video(url=self.predict.Url,
            #                                   name=video_id,
            #                                   title=title,
            #                                   dscription=description,
            #                                   duration=0))

            video_tag_inference = VideoTagInference()

            # Выполняем инференс
            predict = video_tag_inference.predict(title=title,
                                                  description=description,
                                                  path_images=str(frame_path),
                                                  path_audio=str(audio_path))

            # inference = inference_services.create_inference(Inference(CategoryIDS=encode_list_to_string(predict_list),
                                                                         # TagIDS=None))
            # video_inference_services.create_video_inference(VideoInference(VideoID=video_id, InferenceID=inference.ID))

            return predict

        except Exception as ex:
            log.exception(f"Error during prediction process: {ex}")
        finally:
            # Удаляем временную папку после завершения работы
            self.cleanup_temp()

    def cleanup_temp(self):
        """
        Удаляет временные файлы и папки, созданные в процессе обработки видео.
        """
        if os.path.exists(self.temp_path):
            try:
                shutil.rmtree(self.temp_path)
                log.info(f"Temporary folder {self.temp_path} successfully deleted.")
            except Exception as e:
                log.error(f"Failed to delete temporary folder {self.temp_path}: {e}")


def predict(url: str):
    """
    Функция предсказания тегов и категорий для видео.

    :param url: Ссылка url на видео.
    :return: Список предсказанных тегов и категорий.
    """
    from __init__ import path_to_project
    path_to_temp = os.path.join(path_to_project(), "src/script/temp")

    # Инициализируем и запускаем инференс видео
    video_inference = VideoInferencePipeline(url, temp_path=path_to_temp)
    return video_inference.run()
