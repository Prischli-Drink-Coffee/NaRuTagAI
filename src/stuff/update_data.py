import os
import pandas as pd
from tqdm import tqdm
import time
import threading
import numpy as np
import subprocess

BASE = '/mnt/nfs'
metadata_r = pd.read_csv(f'{BASE}/data/metadata_old.csv')
metadata_o = pd.read_csv(f'{BASE}/data/metadata_renamed.csv')

ids_to_delete = set()

# объединяем датасеты
for video_id in tqdm(metadata_o.video_id):
    audio_path = f'{BASE}/parsed_data/audio/{video_id}.mp3'
    frames_path = f'{BASE}/parsed_data/frames/{video_id}/'

    if not os.path.exists(frames_path) or not os.path.isdir(frames_path) or len(os.listdir(frames_path)) < 64:
        ids_to_delete.add(video_id)
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        ids_to_delete.add(video_id)

metadata_o = metadata_o[~metadata_o.video_id.isin(ids_to_delete)]

# перемещаем видео и аудио
for video_id in tqdm(metadata_o.video_id):
    audio_path = f'{BASE}/parsed_data/audio/{video_id}.mp3'
    new_audio_path = f'{BASE}/data/audio/{video_id}.mp3'

    if not os.path.exists(new_audio_path) or os.path.getsize(new_audio_path) == 0:
        audio_cp = f'cp {audio_path} {new_audio_path}'
        subprocess.run(audio_cp, shell=True)
    
    frames_path = f'{BASE}/parsed_data/frames/{video_id}/'
    new_frames_path = f'{BASE}/data/frames/{video_id}/'

    if not os.path.exists(new_frames_path) or not os.path.isdir(new_frames_path) or len(os.listdir(new_frames_path)) < 64:
        frames_cp = f'cp -r {frames_path} {new_frames_path}'
        subprocess.run(frames_cp, shell=True)

metadata = pd.concat((metadata_o, metadata_r))
metadata.to_csv(f'{BASE}/data/metadata.csv', index=False)