import subprocess
import re
import cv2
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import requests
import logging
from bs4 import BeautifulSoup
from moviepy.editor import VideoFileClip
from src.utils.custom_logging import setup_logging

log = setup_logging()


class VideoProcessor:
    def __init__(self, 
                 url: str, 
                 data_folder: str, 
                 min_duration: int = 0, 
                 max_duration: int = 10,
                 quality: str = 'best',
                 audio_fps: int = 16000,
                 n_frames: int = 100,
                 frame_dimensions: Tuple[int, int] = (224, 224)
                 ):
        self.url = url
        self.data_folder = Path(data_folder)
        self.video_folder = self.data_folder / "videos"
        self.audio_folder = self.data_folder / "audio"
        self.frames_folder = self.data_folder / "frames"

        self.min_duration = min_duration
        self.max_duration = max_duration
        self.quality = quality
        self.audio_fps = audio_fps
        self.frame_dimensions = tuple(map(int, frame_dimensions.strip('()').split(', ')))
        self.n_frames = n_frames

        self._ensure_directory(self.data_folder)
        self._ensure_directory(self.video_folder)
        self._ensure_directory(self.audio_folder)
        self._ensure_directory(self.frames_folder)

    def process_video(self) -> Optional[Tuple[str, Optional[str]]]:
        video_id = self._extract_video_id(self.url)
        if video_id is None:
            log.warning("Invalid URL, unable to extract video ID.")
            return None, None, None

        if not self._is_video_valid(self.url):
            log.warning(f"Video does not meet the criteria. Skipping.")
            return None, None, None

        if not self._download_video():
            log.warning(f"Failed to download video.")
            return None, None, None

        title, description = self._fetch_title_description()
        if title is None or description is None:
            log.warning(f"Video does not have a title or description")
            return None, None, None

        log.info(f"Title saved: {title}")
        log.info(f"Description saved")

        self._extract_audio(video_id)
        log.info(f"Audio saved")

        self._extract_frames(video_id)
        log.info(f"Frames saved")

        video_path = self.video_folder / f"{video_id}.mp4"
        video_path.unlink()
        log.info(f"Video was deleted")

        return title, description, video_id

    def _fetch_title_description(self) -> Optional[str]:
        response = requests.get(self.url)
        if response.status_code != 200:
            log.warning(f"Failed to load page, status code: {response.status_code}")
            return None, None

        soup = BeautifulSoup(response.text, 'html.parser')

        title_tag = soup.find('meta', property='og:title')
        script_tag = soup.find('script', type='application/ld+json')

        pattern = r'"description"\s*:\s*"([^"]*)"'
        match = re.search(pattern, script_tag.string or '')

        if title_tag and match:
            title = title_tag.get('content')
            description = match.group(1)
            return title, description
        return None, None

    def _extract_audio(self, video_id: str) -> None:
        audio_path = self.audio_folder / f"{video_id}.mp3"
        video_path = self.video_folder / f"{video_id}.mp4"

        with VideoFileClip(str(video_path)) as video:
            audio = video.audio
            audio.write_audiofile(str(audio_path), fps=self.audio_fps, logger=None)

    def _extract_frames(self, video_id: str) -> int:
        video_path = self.video_folder / f"{video_id}.mp4"
        frame_folder = self.frames_folder / video_id
        self._ensure_directory(frame_folder)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log.error(f"Failed to open video file: {video_path}")
            return 0

        # Получаем свойства видео
        video_fps = cap.get(cv2.CAP_PROP_FPS)  # кадры в секунду
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # общее количество кадров в видео
        duration = total_frames / video_fps  # общая продолжительность в секундах

        # Рассчитываем шаг между кадрами, основываясь на фактическом числе кадров и желаемом числе n_frames
        frame_interval = max(total_frames // self.n_frames, 1)

        frame_saved = 0
        for i in range(self.n_frames):
            frame_number = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # Устанавливаем конкретный кадр

            success, frame = cap.read()
            if not success:
                break

            # Изменяем размер кадра
            resized_frame = cv2.resize(src=frame, dsize=self.frame_dimensions)
            frame_filepath = frame_folder / f'frame_{i}.png'

            try:
                cv2.imwrite(str(frame_filepath), resized_frame)
                frame_saved += 1
            except Exception as e:
                log.error(f"Error saving frame {i}: {e}")

        cap.release()
        return frame_saved

    @staticmethod
    def _extract_video_id(url: str) -> Optional[str]:
        return url.split("/")[-2] if url else None

    def _is_video_valid(self, url: str) -> bool:
        try:
            result = subprocess.run(
                ['yt-dlp', '--get-duration', url],
                capture_output=True,
                text=True,
                check=True
            )
            duration_str = result.stdout.strip()
            log.info(f"Video duration string: {duration_str}")

            if not re.match(r'\d+:\d+(:\d+)?', duration_str):
                return False

            duration = self._convert_duration_to_minutes(duration_str)
            if not (self.min_duration <= duration <= self.max_duration):
                log.warning(f"Video duration {duration} is not valid.")
                return False

            resolution = self._fetch_video_resolution(url)
            if resolution:
                width, height = resolution
                if width < self.frame_dimensions[0] or height < self.frame_dimensions[1]:
                    log.warning(f"Video resolution {resolution} is below the minimum required.")
                    return False

            return True

        except subprocess.CalledProcessError as e:
            log.error(f"Failed to get video details from {url}: ", exc_info=e)
            return False

    def _fetch_video_resolution(self, url: str) -> Optional[Tuple[int, int]]:
        try:
            result = subprocess.run(
                ['yt-dlp', '--get-filename', '-f', 'best', url],
                capture_output=True,
                text=True,
                check=True
            )
            filename = result.stdout.strip()
            match = re.search(r'(\d{3,4}p)', filename)
            if match:
                resolution_str = match.group(1)
                return self._parse_resolution(resolution_str)

        except subprocess.CalledProcessError as e:
            log.error(f"Failed to get video resolution from {url}: ", exc_info=e)
            return None

    @staticmethod
    def _parse_resolution(resolution_str: str) -> Tuple[int, int]:
        if resolution_str.endswith('p'):
            height = int(resolution_str[:-1])
            width = int(height * 16 / 9)
            return width, height
        return 0, 0

    @staticmethod
    def _convert_duration_to_minutes(duration_str: str) -> float:
        parts = duration_str.split(':')
        hours = int(parts[0]) if len(parts) > 2 else 0
        minutes = int(parts[-2]) if len(parts) > 1 else 0
        return hours * 60 + minutes

    @staticmethod
    def _ensure_directory(directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

    def _download_video(self) -> bool:
        command = f'yt-dlp -P {self.video_folder} -f {self.quality} -o "%(id)s.%(ext)s" {self.url}'
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            result.check_returncode()  # Убедиться, что процесс завершился успешно
            log.info(f"Successfully downloaded video from {self.url}")
            return True
        except subprocess.CalledProcessError as e:
            log.error(f"Failed to download video from {self.url}: ", exc_info=e)
            return False
