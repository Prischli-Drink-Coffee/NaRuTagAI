import os
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile, status, Form
from typing import Dict
from fastapi.openapi.models import Tag as OpenApiTag
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from src.utils.custom_logging import setup_logging
from env import Env
from src import path_to_project
from src.database.models import Category, Tag, Video, VideoInference, Inference, Users, APIKey, Predict
from src.services import (category_services, tag_services, video_services,
                          video_inference_services, inference_services, main_services,
                          user_services, api_key_services, authenticate_services)

env = Env()
log = setup_logging()


app_server = FastAPI(title="API - server")
app_public = FastAPI(title="API - public")

app = FastAPI()

app.mount("/server", app_server)
app.mount("/public", app_public)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Определяем теги
PublicMainTag = OpenApiTag(name="Main", description="CRUD operations main")
ServerMainTag = OpenApiTag(name="Main", description="CRUD operations main")
ServerAPIKeyTag = OpenApiTag(name="APIKey", description="CRUD operations APIKey")
ServerUserTag = OpenApiTag(name="User", description="CRUD operations user")
ServerCategoryTag = OpenApiTag(name="Category", description="CRUD operations category")
ServerTagTag = OpenApiTag(name="Tag", description="CRUD operations tag")
ServerVideoTag = OpenApiTag(name="Video", description="CRUD operations video")
ServerVideoInferenceTag = OpenApiTag(name="VideoInference", description="CRUD operations video inference")
ServerInferenceTag = OpenApiTag(name="Inference", description="CRUD operations inference")

# Настройка документации с тегами
app_server.openapi_tags = [
    ServerMainTag.model_dump(),
    ServerAPIKeyTag.model_dump(),
    ServerUserTag.model_dump(),
    ServerCategoryTag.model_dump(),
    ServerTagTag.model_dump(),
    ServerVideoTag.model_dump(),
    ServerVideoInferenceTag.model_dump(),
    ServerInferenceTag.model_dump()
]

app_public.openapi_tags = [
    PublicMainTag.model_dump(),
]



@app_public.post("/signup/", response_model=Users, tags=["Main"])
async def signup(email: str = Form(...), password: str = Form(...)):
    """
    Регистрация нового пользователя.
    """
    try:
        user = authenticate_services.register_user(email, password)
        return user_services.create_user(user)
    except HTTPException as ex:
        log.exception("Error during registration", exc_info=ex)
        raise ex


@app_public.post("/get_api_key/", response_model=APIKey, tags=["Main"])
async def get_api_key(user: Users = Depends(authenticate_services.get_current_user),
                      key_name: str = Form(...)):
    """
    Генерация нового API ключа.
    """
    try:
        key = authenticate_services.get_current_api_key(user.ID, 10, key_name)
        return api_key_services.create_api_key(key)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_public.post("/predict/", response_model=list, tags=["Main"])
async def predict(predict: Predict,
                  api_key: str = authenticate_services.validate_api_key):
    """
    Route for predict tags and category.

    :param predict: Model predict tags and category. [Predict]

    :return: response model dict.
    """
    try:
        return main_services.predict(predict)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/predict/", response_model=list, tags=["Main"])
async def predict(predict: Predict):
    """
    Route for predict tags and category.

    :param predict: Model predict tags and category. [Predict]

    :return: response model dict.
    """
    try:
        return main_services.predict(predict)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/api_keys/", response_model=list[APIKey], tags=["APIKey"])
async def get_all_api_keys():
    """
    Route for get all api keys from basedata.

    :return: response model List[APIKey].
    """
    try:
        return api_key_services.get_all_api_keys()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/api_keys/api_key_id/{api_key_id}", response_model=APIKey, tags=["APIKey"])
async def get_api_key_by_id(api_key_id: int):
    """
    Route for get api_key by APIKeyID.

    :param api_key_id: ID by APIKey. [int]

    :return: response model APIkey.
    """
    try:
        return api_key_services.get_api_key_by_id(api_key_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/api_keys/user_id/{user_id}", response_model=APIKey, tags=["APIKey"])
async def get_api_key_by_user_id(user_id: int):
    """
    Route for get api_key by APIKeyID.

    :param user_id: ID by APIKey. [int]

    :return: response model APIkey.
    """
    try:
        return api_key_services.get_api_key_by_user_id(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/api_keys/", response_model=APIKey, tags=["APIKey"])
async def create_api_key(api_key: APIKey = Depends(authenticate_services.get_current_api_key)):
    """
    Route for create api_key in basedata.

    :param api_key: Model APIKey. [APIKey]

    :return: response model APIKey.
    """
    try:
        return api_key_services.create_api_key(api_key)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.put("/api_keys/{api_key_id}", response_model=Dict, tags=["APIKey"])
async def update_api_key(api_key_id: int, api_key: APIKey = Depends(authenticate_services.get_current_api_key)):
    """
    Route for update api_key in basedata.

    :param api_key_id: ID by APIKey. [int]

    :param api_key: Model APIKey. [APIKey]

    :return: response model dict.
    """
    try:
        return api_key_services.update_api_key(api_key_id, api_key)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/api_keys/{api_key_id}", response_model=Dict, tags=["APIKey"])
async def delete_api_key(api_key_id: int):
    """
    Route for delete api_key from basedata.

    :param api_key_id: ID by APIKey. [int]

    :return: response model dict.
    """
    try:
        return api_key_services.delete_api_key(api_key_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/", response_model=list[Users], tags=["User"])
async def get_all_users():
    """
    Route for get all users from basedata.

    :return: response model List[Users].
    """
    try:
        return user_services.get_all_users()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/user_id/{user_id}", response_model=Users, tags=["User"])
async def get_user_by_id(user_id: int):
    """
    Route for get user by UserID.

    :param user_id: ID by user. [int]

    :return: response model Users.
    """
    try:
        return user_services.get_user_by_id(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/users/email/{email}", response_model=Users, tags=["User"])
async def get_user_by_email(email: str):
    """
    Route for get user by user email.

    :param email: Email by user. [int]

    :return: response model Users.
    """
    try:
        return user_services.get_user_by_email(email)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/users/", response_model=Users, tags=["User"])
async def create_user(user: Users):
    """
    Route for create user in basedata.

    :param user: Model user. [Users]

    :return: response model Users.
    """
    try:
        return user_services.create_user(user)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.put("/users/{user_id}", response_model=Dict, tags=["User"])
async def update_user(user_id, user: Users):
    """
    Route for update user in basedata.

    :param user_id: ID by user. [int]

    :param user: Model user. [Users]

    :return: response model dict.
    """
    try:
        return user_services.update_user(user_id, user)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/users/{user_id}", response_model=Dict, tags=["User"])
async def delete_user(user_id):
    """
    Route for delete user from basedata.

    :param user_id: ID by user. [int]

    :return: response model dict.
    """
    try:
        return user_services.delete_user(user_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/categories/", response_model=list[Category], tags=["Category"])
async def get_all_categories():
    """
    Route for get all categories from basedata.

    :return: response model List[Categories].
    """
    try:
        return category_services.get_all_categories()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/categories/category_id/{category_id}", response_model=Category, tags=["Category"])
async def get_category_by_id(category_id: int):
    """
    Route for get category by CategoryID.

    :param category_id: ID by category. [int]

    :return: response model Categories.
    """
    try:
        return category_services.get_category_by_id(category_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/categories/category_name/{category_name}", response_model=Category, tags=["Category"])
async def get_category_by_name(category_name: str):
    """
    Route for get category by CategoryName.

    :param category_name: Name by category. [int]

    :return: response model Categories.
    """
    try:
        return category_services.get_category_by_name(category_name)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/categories/", response_model=Category, tags=["Category"])
async def create_category(category: Category):
    """
    Route for create category in basedata.

    :param category: Model category. [Category]

    :return: response model Categories.
    """
    try:
        return category_services.create_category(category)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.put("/categories/{category_id}", response_model=Dict, tags=["Category"])
async def update_category(category_id, category: Category):
    """
    Route for update category in basedata.

    :param category_id: ID by category. [int]

    :param category: Model category. [Categories]

    :return: response model dict.
    """
    try:
        return category_services.update_category(category_id, category)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/categories/{category_id}", response_model=Dict, tags=["Category"])
async def delete_category(category_id):
    """
    Route for delete user from basedata.

    :param category_id: ID by Category. [int]

    :return: response model dict.
    """
    try:
        return category_services.delete_category(category_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/tags/", response_model=list[Tag], tags=["Tag"])
async def get_all_tags():
    """
    Route for get all tags from basedata.

    :return: response model List[Tag].
    """
    try:
        return tag_services.get_all_tags()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/tags/tag_id/{tag_id}", response_model=Tag, tags=["Tag"])
async def get_tag_by_id(tag_id: int):
    """
    Route for get tag by TagID.

    :param tag_id: ID by tag. [int]

    :return: response model Tag.
    """
    try:
        return tag_services.get_tag_by_id(tag_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/tags/tag_name/{tag_name}", response_model=Tag, tags=["Tag"])
async def get_tag_by_name(tag_name: str):
    """
    Route for get tag by TagName.

    :param tag_name: Name by tag. [int]

    :return: response model Tag.
    """
    try:
        return tag_services.get_tag_by_name(tag_name)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/tags/", response_model=Tag, tags=["Tag"])
async def create_tag(tag: Tag):
    """
    Route for create tag in basedata.

    :param tag: Model tag. [Tag]

    :return: response model Tag.
    """
    try:
        return tag_services.create_tag(tag)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.put("/tags/{tag_id}", response_model=Dict, tags=["Tag"])
async def update_tag(tag_id, tag: Tag):
    """
    Route for update tag in basedata.

    :param tag_id: ID by tag. [int]

    :param tag: Model tag. [Tag]

    :return: response model dict.
    """
    try:
        return tag_services.update_tag(tag_id, tag)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/tags/{tag_id}", response_model=Dict, tags=["Tag"])
async def delete_tag(tag_id):
    """
    Route for delete tag from basedata.

    :param tag_id: ID by Tag. [int]

    :return: response model dict.
    """
    try:
        return tag_services.delete_tag(tag_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/inferences/", response_model=list[Inference], tags=["Inference"])
async def get_all_inferences():
    """
    Route for get all inferences from basedata.

    :return: response model List[Inference].
    """
    try:
        return inference_services.get_all_inferences()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/inferences/inference_id/{inference_id}", response_model=Inference, tags=["Inference"])
async def get_inference_by_id(inference_id: int):
    """
    Route for get inference by InferenceID.

    :param inference_id: ID by inference. [int]

    :return: response model Inference.
    """
    try:
        return inference_services.get_inference_by_id(inference_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/inferences/", response_model=Inference, tags=["Inference"])
async def create_inference(inference: Inference):
    """
    Route for create inference in basedata.

    :param inference: Model inference. [Inference]

    :return: response model Inference.
    """
    try:
        return inference_services.create_inference(inference)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.put("/inferences/{inference_id}", response_model=Dict, tags=["Inference"])
async def update_inference(inference_id, inference: Inference):
    """
    Route for update inference in basedata.

    :param inference_id: ID by inference. [int]

    :param inference: Model inference. [Inference]

    :return: response model dict.
    """
    try:
        return inference_services.update_inference(inference_id, inference)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/inferences/{inference_id}", response_model=Dict, tags=["Inference"])
async def delete_inference(inference_id):
    """
    Route for delete inference from basedata.

    :param inference_id: ID by inference. [int]

    :return: response model dict.
    """
    try:
        return inference_services.delete_inference(inference_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video_inferences/", response_model=list[VideoInference], tags=["VideoInference"])
async def get_all_video_inferences():
    """
    Route for get all video inferences from basedata.

    :return: response model List[VideoInference].
    """
    try:
        return video_inference_services.get_all_video_inferences()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/video_inferences/video_inference_id/{video_inference_id}", response_model=VideoInference, tags=["VideoInference"])
async def get_video_inference_by_id(video_inference_id: int):
    """
    Route for get video inference by VideoInferenceID.

    :param video_inference_id: ID by video inference. [int]

    :return: response model VideoInference.
    """
    try:
        return video_inference_services.get_video_inference_by_id(video_inference_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/video_inferences/", response_model=VideoInference, tags=["VideoInference"])
async def create_video_inference(video_inference: VideoInference):
    """
    Route for create video inference in basedata.

    :param video_inference: Model video inference. [VideoInference]

    :return: response model VideoInference.
    """
    try:
        return video_inference_services.create_video_inference(video_inference)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.put("/video_inferences/{video_inference_id}", response_model=Dict, tags=["VideoInference"])
async def update_video_inference(video_inference_id, video_inference: VideoInference):
    """
    Route for update video inference in basedata.

    :param video_inference_id: ID by video inference. [int]

    :param video_inference: Model video inference. [VideoInference]

    :return: response model dict.
    """
    try:
        return video_inference_services.update_video_inference(video_inference_id, video_inference)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/video_inferences/{video_inference_id}", response_model=Dict, tags=["VideoInference"])
async def delete_video_inference(video_inference_id):
    """
    Route for delete video inference from basedata.

    :param video_inference_id: ID by video inference. [int]

    :return: response model dict.
    """
    try:
        return video_inference_services.delete_video_inference(video_inference_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/videos/", response_model=list[Video], tags=["Video"])
async def get_all_videos():
    """
    Route for get all videos from basedata.

    :return: response model List[Video].
    """
    try:
        return video_services.get_all_videos()
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.get("/videos/video_id/{video_id}", response_model=Video, tags=["Video"])
async def get_video_by_id(video_id: int):
    """
    Route for get video by VideoID.

    :param video_id: ID by video. [int]

    :return: response model Video.
    """
    try:
        return video_services.get_video_by_id(video_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.post("/videos/", response_model=Video, tags=["Video"])
async def create_video(video: Video):
    """
    Route for create video in basedata.

    :param video: Model video. [Video]

    :return: response model Video.
    """
    try:
        return video_services.create_video(video)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.put("/videos/{video_id}", response_model=Dict, tags=["Video"])
async def update_video(video_id, video: Video):
    """
    Route for update video in basedata.

    :param video_id: ID by video. [int]

    :param video: Model video. [Video]

    :return: response model dict.
    """
    try:
        return video_services.update_video(video_id, video)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


@app_server.delete("/videos/{video_id}", response_model=Dict, tags=["Video"])
async def delete_video(video_id):
    """
    Route for delete video from basedata.

    :param video_id: ID by video. [int]

    :return: response model dict.
    """
    try:
        return video_services.delete_video(video_id)
    except HTTPException as ex:
        log.exception(f"Error", exc_info=ex)
        raise ex


def run_server():
    import logging.config as logging
    import uvicorn
    import yaml
    uvicorn_log_config = os.path.join(path_to_project(), 'logging.yaml')
    with open(uvicorn_log_config, 'r') as f:
        uvicorn_config = yaml.safe_load(f.read())
        logging.dictConfig(uvicorn_config)
    uvicorn.run("server:app", host=env.__getattr__("HOST"), port=int(env.__getattr__("SERVER_PORT")),
                reload=True, log_config=uvicorn_log_config)


if __name__ == "__main__":
    # Создание датабазы и таблиц, если они не существуют
    log.info("Start create/update database")
    from create_sql import CreateSQL

    create_sql = CreateSQL()
    create_sql.read_sql()

    # Запуск сервера и бота
    log.info("Start run server")
    run_server()