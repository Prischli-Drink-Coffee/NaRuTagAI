from src.repository import tag_repository
from src.database.models import Tag
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists


def get_all_tags():
    tags = tag_repository.get_all_tags()
    return [Tag(**tag) for tag in tags]


def get_tag_by_id(tag_id: int):
    tag = tag_repository.get_tag_by_id(tag_id)
    if not tag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Tag not found')
    return Tag(**tag) if tag else None


def get_tag_by_name(tag_name: str):
    tag = tag_repository.get_tag_by_name(tag_name)
    if not tag:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Tag not found')
    return Tag(**tag) if tag else None


def create_tag(tag: Tag):
    check_if_exists(
        get_all=get_all_tags,
        attr_name="Name",
        attr_value=tag.Name,
        exception_detail='Tag already exist'
    )
    tag_id = tag_repository.create_tag(tag)
    return get_tag_by_id(tag_id)


def update_tag(tag_id: int, tag: Tag):
    check_for_duplicates(
        get_all=get_all_tags,
        check_id=tag_id,
        attr_name="Name",
        attr_value=tag.Name,
        exception_detail='Tag already exist'
    )
    tag_repository.update_tag(tag_id, tag)
    return {"message": "Tag updated successfully"}


def delete_tag(tag_id: int):
    get_tag_by_id(tag_id)
    tag_repository.delete_tag(tag_id)
    return {"message": "Tag deleted successfully"}
