import os
import re
import json

import torch
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm
from PIL import Image

from src import project_path
from src.utils.custom_logging import setup_logging

from time import time

log = setup_logging()


@dataclass
class ClusterCollector:
    data_folder: str


    def __post_init__(self):
        # Определяем путь к метадате
        self.metadata_path = os.path.join(project_path, self.data_folder, 'metadata.csv')
        self.metadata = pd.read_csv(self.metadata_path)
        self.data_path = os.path.join(project_path, self.data_folder)

    def run(self):

        start = time()
        log.info(f'Запуск процесса построения кластеров')


        log.info(f'Завершение процесса кластеризации')
        log.info(f'Время завершения: {time() - start} секунд')
