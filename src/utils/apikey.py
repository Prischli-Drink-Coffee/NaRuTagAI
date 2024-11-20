from cryptography.fernet import Fernet
from datetime import datetime
from uuid import uuid4
from src.database.models import APIKeyData
from env import Env

env = Env()
secret_key = env.__getattr__("SECRET_KEY")
cipher = Fernet(bytes(secret_key, "utf-8"))


def generate_encrypted_api_key(user_id: int, usage_limit: int, key_name: str) -> str:
    """
    Генерирует зашифрованный API ключ, включающий данные о пользователе, лимите использования и названии ключа.

    :param user_id: ID пользователя
    :param usage_limit: Лимит использования ключа
    :param key_name: Название ключа, указанное пользователем
    :return: Зашифрованный API ключ в виде строки
    """
    # Данные для шифрования
    api_key_data = APIKeyData(
        user_id=user_id,
        usage_limit=usage_limit,
        key_name=key_name
    )

    # Преобразуем данные в строку для шифрования
    data_str = f"{api_key_data.user_id}:{api_key_data.usage_limit}:{api_key_data.key_name}:{api_key_data.created_at}"

    # Зашифровываем данные
    encrypted_data = cipher.encrypt(data_str.encode())

    # Генерация уникального ключа (UUID4) для хранения
    unique_key_id = str(uuid4())

    # Возвращаем ключ в формате: [UUID].[Зашифрованные данные]
    return f"{unique_key_id}.{encrypted_data.decode()}"


def decrypt_api_key(api_key: str) -> APIKeyData:
    """
    Расшифровывает данные API ключа.

    :param api_key: Строка API ключа в формате [UUID].[Зашифрованные данные]
    :return: Расшифрованные данные в виде APIKeyData
    """
    try:
        # Извлекаем зашифрованные данные из ключа
        _, encrypted_data = api_key.split('.')

        # Расшифровываем данные
        decrypted_data = cipher.decrypt(encrypted_data.encode()).decode()

        # Преобразуем строку обратно в данные
        user_id, usage_limit, key_name, created_at = decrypted_data.split(':')

        return APIKeyData(
            user_id=int(user_id),
            usage_limit=int(usage_limit),
            key_name=key_name,
            created_at=datetime.fromisoformat(created_at)
        )

    except Exception as e:
        raise ValueError("Invalid API key") from e
