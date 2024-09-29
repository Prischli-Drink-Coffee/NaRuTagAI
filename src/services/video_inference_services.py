from src.repository import video_inference_repository
from src.database.models import VideoInference
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists
from src.services.video_services import get_video_by_id
from src.services.inference_services import get_inference_by_id


def get_all_video_inferences():
    video_inferences = video_inference_repository.get_all_video_inferences()
    return [VideoInference(**video_inference) for video_inference in video_inferences]


def get_video_inference_by_id(video_inference_id: int):
    video_inference = video_inference_repository.get_video_inference_by_id(video_inference_id)
    if not video_inference:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Video inference not found')
    return VideoInference(**video_inference) if video_inference else None


def create_video_inference(video_inference: VideoInference):
    get_video_by_id(video_inference.VideoID)
    get_inference_by_id(video_inference.InferenceID)
    video_inference_id = video_inference_repository.create_video_inference(video_inference)
    return get_video_inference_by_id(video_inference_id)


def update_video_inference(video_inference_id: int, video_inference: VideoInference):
    get_video_inference_by_id(video_inference_id)
    get_video_by_id(video_inference.VideoID)
    get_inference_by_id(video_inference.InferenceID)
    video_inference_repository.update_video_inference(video_inference_id, video_inference)
    return {"message": "Video inference updated successfully"}


def delete_video_inference(video_inference_id: int):
    get_video_inference_by_id(video_inference_id)
    video_inference_repository.delete_video_inference(video_inference_id)
    return {"message": "Video inference deleted successfully"}
