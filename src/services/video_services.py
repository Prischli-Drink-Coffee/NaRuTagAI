from src.repository import video_repository
from src.database.models import Video
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, return_id_if_exists


def get_all_videos():
    videos = video_repository.get_all_videos()
    return [Video(**video) for video in videos]


def get_video_by_id(video_id: int):
    video = video_repository.get_video_by_id(video_id)
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Video not found')
    return Video(**video) if video else None


def create_video(video: Video):
    video_id = None
    if video.Name is None:
        video_id = return_id_if_exists(
            get_all=get_all_videos,
            attr_name="Name",
            attr_value=video.Name
        )
    elif video.Url is None:
        video_id = return_id_if_exists(
            get_all=get_all_videos,
            attr_name="Url",
            attr_value=video.Url
        )
    if video_id:
        return get_video_by_id(video_id)
    else:
        video_id = video_repository.create_video(video)
        return get_video_by_id(video_id)


def update_video(video_id: int, video: Video):
    check_for_duplicates(
        get_all=get_all_videos,
        check_id=video_id,
        attr_name="Name",
        attr_value=video.Name,
        exception_detail='Video already exist'
    )
    check_for_duplicates(
        get_all=get_all_videos,
        check_id=video_id,
        attr_name="Url",
        attr_value=video.Name,
        exception_detail='Video already exist'
    )
    video_repository.update_video(video_id, video)
    return {"message": "Video updated successfully"}


def delete_video(video_id: int):
    get_video_by_id(video_id)
    video_repository.delete_video(video_id)
    return {"message": "Video deleted successfully"}

