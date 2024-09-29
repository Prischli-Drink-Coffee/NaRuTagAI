from src.database.my_connector import db
from src.database.models import Video


def get_all_videos():
    query = "SELECT * FROM video"
    return db.fetch_all(query)


def get_video_by_id(video_id: int):
    query = "SELECT * FROM video WHERE id=%s"
    return db.fetch_one(query, (video_id,))


def create_video(video: Video):
    query = ("INSERT INTO video (url, name, title, description,"
             " duration, date_upload) VALUES (%s, %s, %s, %s, %s, %s)")
    params = (video.Url, video.Name, video.Title, video.Description,
              video.Duration, video.DateUpload)
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_video(video_id: int, video: Video):
    query = ("UPDATE video SET url=%s, name=%s, title=%s, description=%s,"
             " duration=%s, date_upload=%s WHERE id=%s")
    params = (video.Url, video.Name, video.Title, video.Description,
              video.Duration, video.DateUpload, video_id)
    db.execute_query(query, params)


def delete_video(video_id: int):
    query = "DELETE FROM video WHERE id=%s"
    db.execute_query(query, (video_id,))
