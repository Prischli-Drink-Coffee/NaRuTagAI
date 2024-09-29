from src.database.my_connector import db
from src.database.models import VideoInference


def get_all_video_inferences():
    query = "SELECT * FROM video_inference"
    return db.fetch_all(query)


def get_video_inference_by_id(video_inference_id: int):
    query = "SELECT * FROM video_inference WHERE id=%s"
    return db.fetch_one(query, (video_inference_id,))


def create_video_inference(video_inference: VideoInference):
    query = "INSERT INTO video_inference (video_id, inference_id) VALUES (%s, %s)"
    params = video_inference.VideoID, video_inference.InferenceID
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_video_inference(video_inference_id: int, video_inference: VideoInference):
    query = "UPDATE video_inference SET video_id=%s, inference_id=%s WHERE id=%s"
    params = video_inference.VideoID, video_inference.InferenceID, video_inference_id
    db.execute_query(query, params)


def delete_video_inference(video_inference_id: int):
    query = "DELETE FROM video_inference WHERE id=%s"
    db.execute_query(query, (video_inference_id,))
