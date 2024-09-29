from pydantic import BaseModel, ValidationError
from typing import Type, Any


def check_valid(model: Type[BaseModel], data: Any) -> bool:
    try:
        model(**data)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False

