from typing import Annotated
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from src.utils.hashing import hash_password, validate_password
from src.database.models import Users, APIKey
from src.services.user_services import get_user_by_email, get_user_by_id
from src.script.apikey import generate_encrypted_api_key, decrypt_api_key
from src.services.api_key_services import get_api_key_by_user_id, update_api_key
import bcrypt
import re

security = HTTPBasic()


# Валидация пароля: минимум 8 символов, хотя бы 1 заглавная буква и хотя бы 1 цифра
def validate_password_strength(password: str) -> bool:
    if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'[0-9]', password):
        return False
    return True


# Регистрация нового пользователя
def register_user(email: str, password: str) -> Users:
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is required",
        )
    if not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password is required",
        )
    if not validate_password_strength(password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long and contain at least one uppercase letter and one digit",
        )

    # Проверка, существует ли пользователь с таким email
    try:
        current_user = get_user_by_email(email)
    except HTTPException:
        current_user = None

    if current_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists",
        )

    return Users(email=email, password=password)


def get_current_user(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
):

    email = credentials.username
    password = credentials.password

    # Проверяем пользователя с таким email
    try:
        current_user = get_user_by_email(email)
    except HTTPException:
        current_user = None

    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    # Проверяем пароль пользователя
    if not validate_password(password, current_user.Password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    return current_user


def get_current_api_key(
        user_id: int,
        usage_limit: int,
        key_name: str
) -> APIKey:

    if not user_id or not usage_limit or not key_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required parameters",
        )

    # Генерация зашифрованного API ключа
    api_key = generate_encrypted_api_key(user_id, usage_limit, key_name)
    return APIKey(key=api_key, user_id=user_id)


def validate_api_key(api_key: str = None):
    # Проверка, был ли предоставлен API ключ
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key is required",
            headers={"WWW-Authenticate": "Basic"},
        )

    try:
        # Дешифруем API ключ
        decrypted_api_key = decrypt_api_key(api_key)
        user_id = decrypted_api_key.user_id
        key_name = decrypted_api_key.key_name
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Проверка существования пользователя
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Проверяем ключ в базе данных
    stored_api_key = get_api_key_by_user_id(user_id)
    if not stored_api_key or stored_api_key.usage_limit <= 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key usage limit exceeded or key not found",
            headers={"WWW-Authenticate": "Basic"},
        )

    if api_key != stored_api_key.key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Basic"},
        )

    # Уменьшаем лимит использования
    stored_api_key.usage_limit -= 1

    # Обновляем ключ в базе данных с новым лимитом
    update_api_key(stored_api_key.ID, {"usage_limit": stored_api_key.usage_limit})

    # Возвращаем ключ как подтверждение успешного обновления
    return {"message": "API key is valid", "remaining_usage": stored_api_key.usage_limit}
