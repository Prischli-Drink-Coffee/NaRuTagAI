from src.database.my_connector import db
from src.database.models import Tag


def get_all_tags():
    query = "SELECT * FROM tags"
    return db.fetch_all(query)


def get_tag_by_id(tag_id: int):
    query = "SELECT * FROM tags WHERE id=%s"
    return db.fetch_one(query, (tag_id,))


def get_tag_by_name(tag_name: str):
    query = "SELECT * FROM tags WHERE name=%s"
    return db.fetch_one(query, (tag_name,))


def create_tag(tag: Tag):
    query = "INSERT INTO tags (name) VALUES (%s)"
    params = tag.Name
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_tag(tag_id: int, tag: Tag):
    query = "UPDATE tags SET name=%s WHERE id=%s"
    params = tag.Name, tag_id
    db.execute_query(query, params)


def delete_tag(tag_id: int):
    query = "DELETE FROM tags WHERE id=%s"
    db.execute_query(query, (tag_id,))
