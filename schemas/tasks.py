from typing import List
from pydantic import BaseModel

class Tasks(BaseModel):
    id: str
    title: str
    description: str
    priority: str
    tags: List[str]
    status: str

