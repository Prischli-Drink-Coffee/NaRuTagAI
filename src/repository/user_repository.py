from src.database.my_connector import db
from src.database.models import Users


def get_all_users():
    query = "SELECT * FROM users"
    return db.fetch_all(query)


def get_user_by_id(user_id: int):
    query = "SELECT * FROM users WHERE id=%s"
    return db.fetch_one(query, (user_id,))


def get_user_by_email(email: str):
    query = "SELECT * FROM users WHERE email=%s"
    return db.fetch_one(query, (email,))


def create_user(user: Users):
    query = ("INSERT INTO users (email, password)"
             " VALUES (%s, %s)")
    params = (user.Email, user.Password)
    cursor = db.execute_query(query, params)
    return cursor.lastrowid


def update_user(user_id: int, user: Users):
    query = "UPDATE users SET email=%s, password=%s WHERE id=%s"
    params = (user.Email, user.Password, user_id)
    db.execute_query(query, params)


def delete_user(user_id: int):
    query = "DELETE FROM users WHERE id=%s"
    db.execute_query(query, (user_id,))
