
from src.repository import api_key_repository
from src.database.models import APIKey
from fastapi import HTTPException, status
from src.utils.exam_services import check_for_duplicates, check_if_exists


def get_all_api_keys():
    apikeys = api_key_repository.get_all_api_keys()
    return [APIKey(**apikey) for apikey in apikeys]


def get_api_key_by_id(api_key_id: int):
    api_key = api_key_repository.get_api_key_by_id(api_key_id)
    if not api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='API key not found')
    return APIKey(**api_key) if api_key else None


def get_api_key_by_user_id(user_id: int):
    api_key = api_key_repository.get_api_key_by_user_id(user_id)
    if not api_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='API key not found')
    return APIKey(**api_key) if api_key else None


def create_api_key(api_key: APIKey):
    check_if_exists(
        get_all=get_all_api_keys,
        attr_name="Key",
        attr_value=api_key.Key,
        exception_detail='Key already exist'
    )
    api_key_id = api_key_repository.create_api_key(api_key)
    return get_api_key_by_id(api_key_id)


def update_api_key(api_key_id: int, api_key: APIKey):
    get_api_key_by_id(api_key_id)
    check_for_duplicates(
        get_all=get_all_api_keys,
        check_id=api_key_id,
        attr_name="Key",
        attr_value=api_key.Key,
        exception_detail='Key already exist'
    )
    api_key_repository.update_api_key(api_key_id, api_key)
    return {"message": "API key updated successfully"}


def delete_api_key(api_key_id: int):
    get_api_key_by_id(api_key_id)
    api_key_repository.delete_api_key(api_key_id)
    return {"message": "API key deleted successfully"}
