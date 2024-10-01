from pydantic import (BaseModel, Field, StrictStr, Json, condecimal,
                      StrictInt, PrivateAttr, SecretBytes, StrictBytes, StrictBool, root_validator,
                      SecretStr)
from enum import Enum
from typing import Optional, List, ClassVar
from datetime import datetime
import os
from pathlib import Path
from env import Env
from cryptography.fernet import Fernet
from datetime import datetime
from uuid import uuid4

env = Env()


class Video(BaseModel):
    """
    Model of video
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    Url: Optional[StrictStr] = Field(None,
                                     alias="url",
                                     examples=["https://rutube.ru/video/10001/"])
    Name: Optional[StrictStr] = Field(None,
                                      alias="name",
                                      examples=["video.mp4"])
    Title: Optional[StrictStr] = Field(None,
                                       alias="title",
                                       examples=["Обзор телефона"])
    Description: Optional[StrictStr] = Field(None,
                                             alias="description",
                                             examples=["В этом видео мы поговорим о телефоне Apple iPhone 12 Pro"])
    Duration: StrictInt = Field(...,
                                alias="duration",
                                examples=[10])
    DateUpload: Optional[datetime] = Field(datetime.now(),
                                           alias="date_upload",
                                           examples=[f"{datetime.now()}"])


class Category(BaseModel):
    """
    Model of category
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    Name: StrictStr = Field(...,
                            alias="name",
                            examples=["Обзоры и распаковки товаров"])


class Tag(BaseModel):
    """
    Model of tag
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    Name: StrictStr = Field(...,
                            alias="name",
                            examples=["Гарри Поттер"])


class Inference(BaseModel):
    """
    Model of inference
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    CategoryIDS: StrictStr = Field(...,
                                   alias="category_ids",
                                   examples=["1,2,4"])
    TagIDS: Optional[StrictStr] = Field(None,
                                        alias="tag_ids",
                                        examples=['1,2,3'])


class VideoInference(BaseModel):
    """
    Model of video inference
    """
    ID: Optional[int] = Field(None,
                              alias="id")
    VideoID: StrictInt = Field(...,
                               alias="video_id",
                               examples=[1])
    InferenceID: StrictInt = Field(...,
                                   alias="inference_id",
                                   examples=[1])


class Predict(BaseModel):
    """
    Model of predict
    """
    Url: StrictStr = Field(...,
                           alias="url",
                           examples=["https://rutube.ru/video/98a85192e297ff4db1860f43ff7a2738/"])


class Users(BaseModel):
    """
    Модель пользователя
    """
    ID: Optional[int] = Field(None,
                              alias="id",
                              examples=[1])
    Email: StrictStr = Field(...,
                             alias="email",
                             examples=["john.doe@example.com"])
    Password: StrictStr = Field(...,
                                alias="password",
                                examples=["password"])
    CreatedAt: Optional[datetime] = Field(datetime.now(),
                                          alias="created_at",
                                          examples=[f"{datetime.now()}"])


class APIKeyData(BaseModel):
    """
    Модель данных, которые нужно зашифровать
    """
    user_id: StrictInt
    usage_limit: StrictInt
    key_name: StrictStr
    created_at: Optional[datetime] = Field(datetime.now(), alias="created_at")


class APIKey(BaseModel):
    """
    Модель API ключа
    """
    ID: Optional[int] = Field(None,
                              alias="id",
                              examples=[1])
    Key: StrictStr = Field(...,
                           alias="key",
                           examples=[str(uuid4())])
    UserID: StrictInt = Field(...,
                              alias="user_id",
                              examples=[1])
    CreatedAt: Optional[datetime] = Field(datetime.now(),
                                          alias="created_at",
                                          examples=[f"{datetime.now()}"])
