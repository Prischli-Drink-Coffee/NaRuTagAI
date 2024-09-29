from http.client import responses
from datetime import datetime
import pytest
import random
import string
from copy import deepcopy
from fastapi.testclient import TestClient
from src.pipeline.server import app
from setup.debug_info import machine
from src.utils.custom_logging import setup_logging
from src.utils.list_to_str import encode_list_to_string
import os

log = setup_logging()
client = TestClient(app)

"""

Ошибка Not Found вероятно говорит о неправильно созданном роуте, или не правильно переданным параметрам в тесты

"""


# Вспомогательная функция для генерации случайных данных
def generate_random_data(data_type, length=8):
    if data_type == "string":
        return ''.join(random.choices(string.ascii_letters, k=length))
    elif data_type == "number":
        return random.randint(1, 1000000)
    elif data_type == "datetime":
        return datetime.now()
    return None


# Вспомогательная функция для выполнения запросов
def api_request(method, url, json_data=None):
    response = client.request(method, url, json=json_data)
    return response


# Вспомогательная функция для проверки статуса и получения данных
def assert_response(response, expected_status, keys=None):
    log.info("-------------------------------------")
    assert response.status_code == expected_status, \
        f"Unexpected status code: {response.status_code}, Response: {response.text}"
    if keys:
        response_data = response.json()
        if isinstance(response_data, list):
            for item in response_data:
                for key in keys:
                    assert key in item
        else:
            for key in keys:
                assert key in response_data
        return response_data
    return None


# Генерация тестовых данных для различных сущностей
def generate_test_data(entity_type):
    data_map = {
        "category": {
            "name": generate_random_data("string"),
        },
        "tag": {
            "name": generate_random_data("string")
        },
        "video": {
            "url": generate_random_data("string"),
            "name": generate_random_data("string"),
            "title": generate_random_data("string"),
            "description": generate_random_data("string"),
            "duration": generate_random_data("number"),
            "date_upload": f"{generate_random_data('datetime')}"
        },
        "inference": {
            "category_ids": None,
            "tag_ids": None
        },
        "video_inference": {
            "video_id": None,
            "inference_id": None
        },
        "api_key": {
            "key": generate_random_data("string"),
            "user_id": None
        },
        "user": {
            "email": generate_random_data("string"),
            "password": generate_random_data("string"),
            "created_at": f"{generate_random_data('datetime')}"
        }
    }
    return data_map.get(entity_type)


tag_ids = []
category_ids = []


def setup_entity(entity_type, endpoint):
    if entity_type == "inference":
        inference_data = generate_test_data("inference")
        for index in range(1, 4):
            response = api_request("POST", f"server/categories/", json_data={"name": f"category{generate_random_data('number')}"})
            category_id = response.json()["id"]
            category_ids.append(category_id)
        category_ids_str = encode_list_to_string(category_ids)
        for index in range(1, 4):
            response = api_request("POST", f"server/tags/", json_data={"name": f"tag{generate_random_data('number')}"})
            tag_id = response.json()["id"]
            tag_ids.append(tag_id)
        tag_ids_str = encode_list_to_string(tag_ids)
        entity_data = {**inference_data,
                       "category_ids": category_ids_str,
                       "tag_ids": tag_ids_str}
    elif entity_type == "video_inference":
        video_id = setup_entity("video", "server/videos")
        inference_id = setup_entity("inference", "server/inferences")
        video_inference_data = generate_test_data("video_inference")
        entity_data = {**video_inference_data,
                       "video_id": video_id,
                       "inference_id": inference_id}
    elif entity_type == "api_key":
        user_id = setup_entity("user", "server/users")
        entity_data = {**generate_test_data("api_key"),
                       "user_id": user_id}
    else:
        entity_data = generate_test_data(entity_type)
    log.info(f"Creating {entity_type} with data: {entity_data}")
    response = api_request("POST", f"/{endpoint}/", json_data=entity_data)
    log.info(f"POST {endpoint}/ response: {response.json()}")
    response_data = assert_response(response, 200, keys=["id"])
    if entity_type == "inference":
        for tag_id in tag_ids:
            teardown_entity("server/tags", int(tag_id))
        tag_ids.clear()
    return response_data["id"]


# Функция для удаления сущности
def teardown_entity(endpoint, entity_id):
    response = api_request("DELETE", f"/{endpoint}/{entity_id}")
    assert_response(response, 200)


@pytest.mark.parametrize("entity_type, endpoint, expected_keys", [
    ("category", "server/categories", ["name"]),
    ("tag", "server/tags", ["name"]),
    ("video", "server/videos", ["url"]),
    ("inference", "server/inferences", ["category_ids", "tag_ids"]),
    ("video_inference", "server/video_inferences", ["id"]),
    ("api_key", "server/api_keys", ["key"]),
    ("user", "server/users", ["email"]),
])
def test_create_and_get_entity(entity_type, endpoint, expected_keys):
    log.info("-------------------------------------")
    log.info(f"entity_type: {entity_type}, endpoint: {endpoint}, expected_keys: {expected_keys}")
    entity_id = setup_entity(entity_type, endpoint)
    response = api_request("GET", f"/{endpoint}/")
    assert_response(response, 200, keys=["id"] + expected_keys)
    response = api_request("GET", f"/{endpoint}/{entity_type}_id/{entity_id}")
    assert_response(response, 200, keys=["id"] + expected_keys)
    teardown_entity(endpoint, entity_id)


@pytest.mark.parametrize("entity_type, endpoint, update_data", [
    ("category", "server/categories", {"name": generate_random_data("string")}),
    ("tag", "server/tags", {"name": generate_random_data("string")}),
    ("video", "server/videos", {"url": generate_random_data("string")}),
    ("inference", "server/inferences", {"tag_ids": None}),
    ("video_inference", "server/video_inferences", {"id": generate_random_data("number")}),
    ("api_key", "server/api_keys", {"key": generate_random_data("string")}),
    ("user", "server/users", {"email": generate_random_data("string")}),
])
def test_update_entity(entity_type, endpoint, update_data):
    log.info("-------------------------------------")
    log.info(f"entity_type: {entity_type}, endpoint: {endpoint}, update_data: {update_data}")
    entity_id = setup_entity(entity_type, endpoint)
    response = api_request("GET", f"/{endpoint}/{entity_type}_id/{entity_id}")
    test_data = response.json()
    updated_data = deepcopy(test_data)
    updated_data.update(update_data)
    response = api_request("PUT", f"/{endpoint}/{entity_id}", json_data=updated_data)
    assert_response(response, 200)
    teardown_entity(endpoint, entity_id)


if __name__ == "__main__":
    pytest.main([__file__])
