from unicodedata import category

from src.repository import inference_repository
from src.database.models import Inference
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists
from src.services.category_services import get_category_by_id
from src.services.tag_services import get_tag_by_id
from src.utils.list_to_str import decode_string_to_list


def get_all_inferences():
    inferences = inference_repository.get_all_inferences()
    return [Inference(**inference) for inference in inferences]


def get_inference_by_id(inference_id: int):
    inference = inference_repository.get_inference_by_id(inference_id)
    if not inference:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Inference not found')
    return Inference(**inference) if inference else None


def create_inference(inference: Inference):
    category_ids = decode_string_to_list(inference.CategoryIDS)
    for category_id in category_ids:
        get_category_by_id(category_id)
    if inference.TagIDS:
        tag_ids = decode_string_to_list(inference.TagIDS)
        for tag_id in tag_ids:
            get_tag_by_id(tag_id)
    inference_id = inference_repository.create_inference(inference)
    return get_inference_by_id(inference_id)


def update_inference(inference_id: int, inference: Inference):
    get_inference_by_id(inference_id)
    category_ids = decode_string_to_list(inference.CategoryIDS)
    for category_id in category_ids:
        get_category_by_id(category_id)
    if inference.TagIDS:
        tag_ids = decode_string_to_list(inference.TagIDS)
        for tag_id in tag_ids:
            get_tag_by_id(tag_id)
    inference_repository.update_inference(inference_id, inference)
    return {"message": "Inference updated successfully"}


def delete_inference(inference_id: int):
    get_inference_by_id(inference_id)
    inference_repository.delete_inference(inference_id)
    return {"message": "Inference deleted successfully"}
