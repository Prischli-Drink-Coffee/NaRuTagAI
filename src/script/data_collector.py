import os
from re import split

import requests
import pandas as pd
from typing import Tuple, Optional, List, Dict
from bs4 import BeautifulSoup
from tqdm import tqdm
import json

from src.utils.count_category import count_category
from src import project_path
from src.modelling.video_processing import VideoProcessor
from src.utils.custom_logging import setup_logging
import re

log = setup_logging()


class RutubeVideoCollector:
    def __init__(self,
                 data_folder: str,
                 max_videos_per_category: int = 30,
                 max_videos_per_tag: int = 10,
                 min_duration: int = 0,
                 max_duration: int = 10,
                 quality: str = 'best',
                 audio_fps: int = 16000,
                 n_frames: int = 100,
                 frame_dimensions: Tuple[int, int] = (224, 224),
                 ):
        self.base_url = "https://rutube.ru"
        self.max_videos_per_category = max_videos_per_category 
        self.max_videos_per_tag = max_videos_per_tag
        self.data_folder = data_folder

        if not os.path.isdir(self.data_folder):
            os.makedirs(self.data_folder)

        # video stuff
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.quality = quality

        # audio stuff
        self.audio_fps = audio_fps

        # frame stuff
        self.n_frames = n_frames
        self.frame_dimensions = frame_dimensions

        self.tag_url = f"{self.base_url}/tags/video/"
        self.category_url = f"{self.base_url}/video/category/"
        self.file_path = os.path.join(self.data_folder, 'metadata.csv')
        self.categories_path = os.path.join(self.data_folder, 'categories_dict.json')
        self.path_to_temp = os.path.join(project_path, 'temp_collector.txt')
        
        # Check if the file exists and is not empty
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            self.DF = pd.read_csv(self.file_path)
            with open(self.categories_path, "r") as json_file:
                self.count_category = json.load(json_file)
        else:
            self.DF = pd.DataFrame(None, columns=['video_id', 'category', 'tag', 'title', 'description'])
            self.count_category = count_category()

        if os.path.exists(self.path_to_temp):
            with open(self.path_to_temp, 'r') as temp_file:
                try:
                    self.start = int(temp_file.read())
                except IOError:
                    self.start = 576
        else:
            self.start = 576
    
    def _save_categories(self):
        with open(self.categories_path, "w") as json_file:
            json.dump(self.count_category, json_file)

    @staticmethod
    def get_video_category(url: str) -> Tuple[Optional[str], Optional[str]]:

        # Парсим информацию со ссылки
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Response code {response.status_code}")

        # Извлекаем информацию из response
        soup = BeautifulSoup(response.text, 'html.parser')
        script_tag = soup.find('script', type='application/ld+json')

        # Получаем категорию видео
        pattern = r'"genre"\s*:\s*"([^"]*)"'
        match = re.search(pattern, script_tag.string or '')
        genre = match.group(1) if match else None

        return genre

    def fetch_tags(self, tag_url) -> Dict[str, Dict[str, str]]:

        # Парсим страницу по ссылке тега
        response = requests.get(tag_url)
        if response.status_code != 200:
            raise Exception(f"Response code {response.status_code}")

        # Извлекаем информацию из response
        soup = BeautifulSoup(response.text, 'html.parser')

        # Вытягиваем заголовок тега
        tag = soup.find('h1', class_='freyja_pen-page-header__pen-page-header_main-header_ft7Yj'
                                     ' freyja_pen-page-header__pen-page-header_color-default_z49mH'
                                     ' freyja_pen-page-header__pen-page-header_size-big_NgShu')

        # Вытягиваем из эелемента заголовка название тега
        if tag is not None:
            tag = tag.get('title')
        else:
            raise Exception(f"Tag {tag_url} not found")

        #Получаем сетку с видео на странице тега
        grids = soup.find_all('a', class_='wdp-link-module__link wdp-card-description-module__title'
                                          ' wdp-card-description-module__url wdp-card-description-module__videoTitle')

        # Извлекаем ссылки на видео из сетки
        videos = []
        for index, grid in enumerate(grids):
            # Получаем ссылку на тег из атрибута href
            video_url = f"{self.base_url}{grid.get('href')}"
            # Добавляем данные в список
            videos.append(video_url)
            # Если видео больше чем нужно, то прерываем сбор ссылок
            if index + 1 == self.max_videos_per_tag:
                break

        return videos, tag

    def download_video_metadata(self, video_url: str, video_idx: int, num_video:int) -> Tuple[List[str], List[str], List[str]]:

        log.info(f'Processing {video_url} ({video_idx+1} / {num_video})')

        # Инициализируем видео процессор
        processor = VideoProcessor(
            url=video_url,
            data_folder=self.data_folder,
            min_duration=self.min_duration,
            max_duration=self.max_duration,
            quality=self.quality,
            n_frames=self.n_frames,
            frame_dimensions=self.frame_dimensions
        )

        # Получаем название и описание видео
        title, description, video_id = processor.process_video()
        if video_id is None:
            raise Exception(f"Video ID not found: {video_url}")
        if title is None:
            raise Exception(f"Title not found: {video_url}")
        if description is None:
            raise Exception(f"Description not found: {video_url}")

        return title, description, video_id

    def run(self) -> pd.DataFrame:

        for index in range(self.start, 10000):

            with open(os.path.join(project_path, 'temp_collector.txt'), 'w') as temp_file:
                temp_file.write(f'{index}')

            log.info(f"STARTING NEW TAG #{index} {'=' * 40}")

            try:
                # Получаем список ссылок на видео для тега
                videos, tag = self.fetch_tags(f"{self.tag_url}{index}")
                log.info(f"Tag: '{tag}'")
                log.info(f"num_video: {len(videos)}")
            except Exception as ex:
                log.warning(f"{ex}")
                continue

            video_data = []

            for video_idx, url in enumerate(videos):
                log.info('*'*60)

                try:
                    # Получаем название категории для видео
                    category = self.get_video_category(url)
                except Exception as ex:
                    log.warning(f"{ex}")
                    continue

                if self.count_category.get(category, 30) == self.max_videos_per_category:
                    log.warning(f"Category '{category}' was finish")
                    continue

                try:
                    # Получаем название, описание и id видео
                    title, description, video_id = self.download_video_metadata(url, video_idx, num_video=len(videos))
                    self.count_category[category] += 1
                    log.info(f"Category '{category}', {self.count_category[category]} video(-s) have already been received")

                except Exception as ex:
                    log.warning(f"{ex}")
                    continue

                # Дополняем список с данными о видео полученными аргументами
                video_data.append({
                    'video_id': video_id,
                    'category': category,
                    'tag': tag,
                    'title': title,
                    'description': description,
                })

            df = pd.DataFrame(video_data)

            self.DF = pd.concat([self.DF, df])

            # update index
            with open(self.path_to_temp, 'w') as temp_file:
                print(self.start + 1)
            
            # save current categories dict
            self._save_categories()

            if video_data is not []:
                try:
                    # Записываем данные в csv файл метадаты
                    self.DF.to_csv(self.file_path, index=False)
                except IOError as e:
                    log.exception(f"Error writing to file {self.file_path}")
            else:
                continue
