from src.repository import category_repository
from src.database.models import Category
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists


def get_all_categories():
    categories = category_repository.get_all_categories()
    return [Category(**category) for category in categories]


def get_category_by_id(category_id: int):
    category = category_repository.get_category_by_id(category_id)
    if not category:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Category not found')
    return Category(**category) if category else None


def get_category_by_name(category_name: str):
    category = category_repository.get_category_by_name(category_name)
    if not category:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Category not found')
    return Category(**category) if category else None


def create_category(category: Category):
    check_if_exists(
        get_all=get_all_categories,
        attr_name="Name",
        attr_value=category.Name,
        exception_detail='Category already exist'
    )
    category_id = category_repository.create_category(category)
    return get_category_by_id(category_id)


def update_category(category_id: int, category: Category):
    check_for_duplicates(
        get_all=get_all_categories,
        check_id=category_id,
        attr_name="Name",
        attr_value=category.Name,
        exception_detail='Category already exist'
    )
    category_repository.update_category(category_id, category)
    return {"message": "Category updated successfully"}


def delete_category(category_id: int):
    get_category_by_id(category_id)
    category_repository.delete_category(category_id)
    return {"message": "Category deleted successfully"}
