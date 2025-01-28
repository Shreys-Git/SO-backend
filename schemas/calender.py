from typing import TypedDict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Extraction(BaseModel):
    obligation_type: str = Field(
        description="Name of the obligation being extracted"
    )
    extraction: str = Field(
        description="Exact words extracted from the given document",
    )
    due_date_applicable: bool = Field(
        description="Boolean entry for whether an obligation has a due date",
    )
    due_date: str = Field(
        descrption="If an obligation has an applicable due date, add here in DD-MM-YYYY format"
    )
    actions_needed: bool = Field(
        descrption ="Whether an obligation needs an active action. Examples are paying dues, generating reports, etc"
    )

class Obligations(BaseModel):
  obligations: List[Extraction]

class ExtractionState(TypedDict):
  extractions: List[Extraction]
  vector_search: List[str]

class Event(BaseModel):
    event_id: Optional[str] = Field(None)
    document_id: str
    title: str
    description: str
    start: datetime
    end: Optional[datetime] = None
    allDay: Optional[bool] = False

