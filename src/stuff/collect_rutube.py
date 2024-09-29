import pandas as pd
from tqdm import tqdm
import concurrent.futures
import os

from src.modelling.video_processing import VideoProcessor

SAVE_FOLDER = '/mnt/nfs/data/'
data = pd.read_csv(os.path.join(SAVE_FOLDER, 'metadata_rutube.csv'))
failed_video = os.path.join(SAVE_FOLDER, 'failed.txt')

def process_video(row):
    video_id = row['video_id']
    url = 'https://rutube.ru/video/' + video_id

    processor = VideoProcessor(
        url=url,
        data_folder=SAVE_FOLDER,
        min_duration=0, 
        max_duration=500,
        quality='worst',
        audio_fps=16000,
        n_frames=64,
        frame_dimensions='(256, 256)'
    )
    
    video_folder = os.path.join(SAVE_FOLDER, 'videos')
    audio_folder = os.path.join(SAVE_FOLDER, 'audio')
    frames_folder = os.path.join(SAVE_FOLDER, 'frames')

    # Скачиваем видео, если его нет
    if f'{video_id}.mp4' not in os.listdir(video_folder):
        with open(failed_video, 'a') as f:
            print(video_id, file=f, end = '\n')

    else:
        # Параллельно извлекаем аудио и кадры
        if f'{video_id}.mp3' not in os.listdir(audio_folder) and f'{video_id}' not in os.listdir(frames_folder):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_audio = executor.submit(processor._extract_audio, video_id)
                future_frames = executor.submit(processor._extract_frames, video_id)

                # Ожидаем завершения обеих задач
                future_audio.result()
                future_frames.result()

# Запускаем обработку в цикле
for i in tqdm(range(len(data))):
    row = data.iloc[i, :]
    process_video(row)