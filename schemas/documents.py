from operator import add
from typing import TypedDict, Literal, Optional, Annotated, List

from pydantic import BaseModel, Field

from schemas.users import User


class SignEmail(BaseModel):
    document_id: str
    subject: str
    file_name: str
    file_content: str
    primary_users: List[User]
    cc_users: List[User]


class UserPrompt(BaseModel):
    prompt: str

class Document(BaseModel):
    document_text: str

class EditInput(BaseModel):
    prompt: str
    agreement: str
    document_id: str

class AIEdit(BaseModel):
  original_agreement_text: str = Field(
        description="Contains the exact original legal agreement provided by the user",
    )
  updated_agreement_text: str = Field(
        description="Contains the updated response",
    )
  update_summary: str = Field(
        description="Summary of the changes made described using bullet points",
    )

class SectionState(TypedDict):
    section: AIEdit
    prompt: str
    agreement_text: str
    provisions: str
    updated_agreement_text: str

class Insight(BaseModel):
    insight_type: Literal["clause", "obligation"] = Field(
        description="Describes whether a clause or an obligation is being extracted",
    )
    explanation: str = Field(
        description="Brief explanation of the field being extracted",
    )
    extraction: str = Field(
        description="Exact words extracted from the given document",
    )
    document_lookup: bool = Field(
        description="Whether more information is needed from the document to help research this section",
    )
    deviation: bool = Field(
        description="Whether this kind of text is expected in the document",
    )
    insight_generated: str = Field(
        description="Insight generated using a combination of web search result and optionally, more information from the document"
    )

class Insights(BaseModel):
    insights: List[Insight] = Field(
        description="Insights generated from the document",
    )

class Extraction(BaseModel):
    extraction: str = Field(description="Obligation or Clause wording extracted directly from the document")

class Extractions(BaseModel):
    extractions: List[Extraction] = Field(description="List of all the extractions from the document")

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class InsightState(TypedDict):
    tavily_topic: Literal["general", "news"]
    tavily_days: Optional[int]
    number_of_queries: int
    insight: Insight
    extractions: Extractions
    agreement: str
    search_queries: list[SearchQuery] # List of search queries
    source_str: str # String of formatted source content from web search
    report_insights_from_research: str # String of any completed sections from research to write final sections
    completed_insights: Annotated[list[Insight], add]

class InsightAgreement(BaseModel):
    agreement: str
    insight_type: str

class ReportState(TypedDict):
    topic: str  # Report topic
    insights: List[Insight]  # List of report insights
    completed_insights: List[Insight]  # Completed insights for the report
    report_insights_from_research: str  # String of any completed insights from research to write final insights
    final_report: str  # Final report
    number_of_queries: int
    tavily_topic: str
    tavily_days: str