# src/schema/sign_schema.py
from pydantic import BaseModel

class ErrorResponse(BaseModel):
    detail: str