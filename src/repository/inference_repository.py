from src.database.my_connector import db
from src.database.models import Inference


def get_all_inferences():
    query = "SELECT * FROM inference"
    return db.fetch_all(query)


def get_inference_by_id(inference_id: int):
    query = "SELECT * FROM inference WHERE id=%s"
    return db.fetch_one(query, (inference_id,))


def create_inference(inference: Inference):
    query = "INSERT INTO inference (category_ids, tag_ids) VALUES (%s, %s)"
    params = inference.CategoryIDS, inference.TagIDS
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_inference(inference_id: int, inference: Inference):
    query = "UPDATE inference SET category_ids=%s, tag_ids=%s WHERE id=%s"
    params = inference.CategoryIDS, inference.TagIDS, inference_id
    db.execute_query(query, params)


def delete_inference(inference_id: int):
    query = "DELETE FROM inference WHERE id=%s"
    db.execute_query(query, (inference_id,))
