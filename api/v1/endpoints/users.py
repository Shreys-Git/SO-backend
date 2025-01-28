import uuid

from fastapi import APIRouter
from pymongo import MongoClient

from config import BaseConfig
from schemas.users import User

router = APIRouter()
settings = BaseConfig()

client = MongoClient(settings.DB_URL)
db_name = settings.DB_NAME
collection_name = settings.USERS_COLLECTION
collection = client[db_name][collection_name]

@router.get("/")
def get_users():
    return list(collection.find({}, { "_id": 0 }))

@router.get("/{user_id}")
def get_users(user_id: str):
    return collection.find_one({"user_id": user_id}, { "_id": 0 })

@router.post("/")
def create_user(user: User):
    user_id = str(uuid.uuid4())
    user = { "user_id": user_id,
             "first_name": user.first_name,
             "last_name": user.last_name,
             "email": user.email }

    collection.insert_one(user)

@router.delete("/{user_id}")
def delete_user(user_id: int):
    collection.delete_one({"user_id": user_id})


