import uuid

from fastapi import APIRouter

from config import BaseConfig
from db.database import tasks
from schemas.tasks import Tasks

router = APIRouter()
settings = BaseConfig()

@router.get("/")
async def get_tasks():
    all_tasks = list(tasks.find({},{"_id": 0}))
    return all_tasks

@router.post("/")
async def add_task(task: Tasks):
    tasks.insert_one({"id": task.id, "title" :task.title, "description" : task.description, "priority": task.priority, "tags": task.tags, "status": task.status })
    return "success"

@router.put("/{task_id}/{newStatus}")
async def add_task(task_id: str, new_status: str):
    tasks.update_one({"id": task_id}, {"$set":{"status": new_status}})
    return "success"

@router.delete("/{task_id}")
async def add_task(task_id: str):
    tasks.delete_one({"id": task_id})
    return "success"
