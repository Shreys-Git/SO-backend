from fastapi import APIRouter

from config import BaseConfig

documents_router = APIRouter()
settings = BaseConfig()