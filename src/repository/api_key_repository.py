from src.database.my_connector import db
from src.database.models import APIKey


def get_all_api_keys():
    query = "SELECT * FROM api_keys"
    return db.fetch_all(query)


def get_api_key_by_id(api_key_id: int):
    query = "SELECT * FROM api_keys WHERE id=%s"
    return db.fetch_one(query, (api_key_id,))


def get_api_key_by_user_id(user_id: int):
    query = "SELECT * FROM api_keys WHERE id=%s"
    return db.fetch_one(query, (user_id,))


def create_api_key(api_key: APIKey):
    query = "INSERT INTO api_keys (`key`, user_id) VALUES (%s, %s)"
    params = api_key.Key, api_key.UserID
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_api_key(api_key_id: int, api_key: APIKey):
    query = "UPDATE api_keys SET `key`=%s, user_id=%s WHERE id=%s"
    params = api_key.Key, api_key.UserID, api_key_id
    db.execute_query(query, params)


def delete_api_key(api_key_id: int):
    query = "DELETE FROM api_keys WHERE id=%s"
    db.execute_query(query, (api_key_id,))
