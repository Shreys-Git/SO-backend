from typing import List

from fastapi import UploadFile
from pydantic import BaseModel


class UserChatMessage(BaseModel):
    session_id: str
    message: str
    document_ids : List[str]

class AdditionalDocs(BaseModel):
    additional_docs: List[UploadFile]
