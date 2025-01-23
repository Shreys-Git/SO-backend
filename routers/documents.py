import io
import uuid
from typing import List

from PyPDF2 import PdfReader
from fastapi import APIRouter, UploadFile, File
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from pydantic import BaseModel
from pymongo import MongoClient

from config import BaseConfig

import asyncio
import operator
from typing_extensions import TypedDict
from typing import  Annotated, List, Optional, Literal
from pydantic import BaseModel, Field

from tavily import TavilyClient, AsyncTavilyClient

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langsmith import traceable

from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

from routers.mongo import create_chunks_with_recursive_split

documents_router = APIRouter()
settings = BaseConfig()

class UserChatMessage(BaseModel):
    docs: List[UploadFile]

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

# Prompt to extract the clauses/ obligation from the report
obligation_extraction_instructions = """You are an expert legal researcher. You goal is, given a legal document,
extract all the Obligations from it.

While extracting the legal obligations make sure:
1. Extract the exact words from the legal document, without changing them in any way.
2. Along with the words of the obligation, also, if applicable, extract the obligation's due date.
3. Only present the extracted obligation without any additional information or conversation.

The context to extact obligation from is:

{vector_search}

"""
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    max_retries=1,
    api_key=settings.OPENAI_SECRET_KEY
)


async def extract_obligations(state: ExtractionState):
    # Get the vector search results
    vector_search = state["vector_search"]

    extraction_instruction_query = obligation_extraction_instructions.format(vector_search=vector_search)
    structured_extraction_llm = llm.with_structured_output(Obligations)
    extractions = structured_extraction_llm.invoke(extraction_instruction_query)

    return {"extractions": extractions}

@documents_router.post("/upload")
async def documents_process(docs: List[UploadFile] = File([])):
    agreement_ids = []
    processed_docs = []

    # Access the collection
    client = MongoClient(settings.DB_URL, uuidRepresentation="standard")
    db_name = settings.DB_NAME
    collection_name = "ShrsingCollection20012025"
    collection = client[db_name][collection_name]

    # Search Index for the chunked agreement data
    vector_search_index = "vector_index_20012025"

    # Initialise the vector store
    vector_store = MongoDBAtlasVectorSearch(
        embedding=OpenAIEmbeddings(api_key = settings.OPENAI_SECRET_KEY, disallowed_special=()),
        collection=collection,
        index_name=vector_search_index,
    )

    for doc in docs:
        file_content = await doc.read()
        file_bytes = io.BytesIO(file_content)
        pdf_reader = PdfReader(file_bytes)

        document_id = uuid.uuid4()
        print("Generated doc id: " + str(document_id))
        agreement_ids.append(str(document_id))
        for page in pdf_reader.pages:
            text = page.extract_text()
            page_doc = Document(page_content=text, metadata={"document_id": document_id})
            processed_docs.append(page_doc)

    # Chunk (if applicable)
    chunks = create_chunks_with_recursive_split(processed_docs, 1000, 200)
    # Add chunks to the Vector Store
    vector_store.add_documents(documents=chunks)

    # Instantiate Atlas Vector Search as a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.75,
            "pre_filter": {"document_id": {"$in": [uuid.UUID(id) for id in agreement_ids]}}
        }
    )

    vector_search_results = retriever.invoke("obligations")

    return vector_search_results

    # extraction_builder = StateGraph(ExtractionState)
    # extraction_builder.add_node("extract_obligations", extract_obligations)
    #
    # extraction_builder.add_edge(START, "extract_obligations")
    # extraction_builder.add_edge("extract_obligations", END)
    #
    # extraction_graph = extraction_builder.compile()
    # extractions = await extraction_graph.ainvoke({"vector_search": vector_search_results})
    # # Insert documents into the database
    # # Documents should be Document text + Relevant Context + Extractions
    #
    # result = collection.insert_many(extractions)
    #
    # # Return the inserted IDs
    # return {"inserted_ids": [str(id) for id in result.inserted_ids]}

