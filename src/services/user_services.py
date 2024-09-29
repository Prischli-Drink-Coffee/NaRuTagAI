from src.repository import user_repository
from src.database.models import Users
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists
from src.utils.hashing import hash_password, validate_password


def get_all_users():
    users = user_repository.get_all_users()
    return [Users(**user) for user in users]


def get_user_by_id(user_id: int):
    user = user_repository.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
    return Users(**user) if user else None


def get_user_by_email(email: str):
    user = user_repository.get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
    return Users(**user) if user else None


def create_user(user: Users):
    check_if_exists(
        get_all=get_all_users,
        attr_name="Email",
        attr_value=user.Email,
        exception_detail='Email already exist'
    )
    user.Password = hash_password(user.Password)
    user_id = user_repository.create_user(user)
    return get_user_by_id(user_id)


def update_user(user_id: int, user: Users):
    check_for_duplicates(
        get_all=get_all_users,
        check_id=user_id,
        attr_name="Email",
        attr_value=user.Email,
        exception_detail='Email already exist'
    )
    user.Password = hash_password(user.Password)
    user_repository.update_user(user_id, user)
    return {"message": "User updated successfully"}


def delete_user(user_id: int):
    get_user_by_id(user_id)
    user_repository.delete_user(user_id)
    return {"message": "User deleted successfully"}
