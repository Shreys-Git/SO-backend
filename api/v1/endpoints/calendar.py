import uuid
from typing import List

from bson import ObjectId
from fastapi import APIRouter, HTTPException
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pymongo import MongoClient

from config import BaseConfig
from core.utility.helpers.calender import extract_obligations
from core.utility.helpers.common import process_all_nav_documents
from db.database import calendar, documents
from schemas.calender import Event, ExtractionState

router = APIRouter()
settings = BaseConfig()

@router.post("/", response_model=Event)
async def create_event(event: Event):
    event_data = event.model_dump(by_alias=True)  # Convert to MongoDB-compatible format
    print(event_data)
    result = calendar.insert_one(event_data)
    event_data["_id"] = str(result.inserted_id)
    return event_data

@router.get("/", response_model=List[Event])
async def get_events():
    events = list(calendar.find())
    for event in events:
        event["_id"] = str(event["_id"])
    return events

@router.put("/{event_id}", response_model=Event)
async def update_event(event_id: str, event: Event):
    update_data = event.dict(by_alias=True, exclude_unset=True)
    result = calendar.find_one_and_update(
        {"_id": ObjectId(event_id)},
        {"$set": update_data},
        return_document=True
    )
    if not result:
        raise HTTPException(status_code=404, detail="Event not found")
    result["_id"] = str(result["_id"])
    return result

@router.delete("/{event_id}", response_model=dict)
async def delete_event(event_id: str):
    result = calendar.delete_one({"_id": ObjectId(event_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted successfully"}

@router.get("/setup")
async def get_events():
    try:
        docs = list(documents.find())
        # Extract events from Docs
        events = []
        for doc in docs:
            document_id = doc.get("document_id")

            provisions = doc.get("navigator_extractions", {}).get("provisions", {})
            effective_date = provisions.get("effective_date", "")
            execution_date = provisions.get("execution_date", "")

            file_name = doc.get("navigator_extractions", {}).get("file_name", "")

            if effective_date and file_name:
                event = {
                    "event_id": str(uuid.uuid4()),  # Each event gets a unique UUID
                    "document_id": document_id,
                    "title": file_name,
                    "description": "Effective date due",
                    "start": effective_date,
                    "end": effective_date,
                    "allDay": True
                }
                events.append(event)

            if execution_date and file_name:
                event = {
                    "event_id": str(uuid.uuid4()),  # Each event gets a unique UUID
                    "document_id": document_id,
                    "title": file_name,
                    "description": "Execution date due",
                    "start": execution_date,
                    "end": execution_date,
                    "allDay": True
                }
                events.append(event)

        ids = []
        if events:
            # Insert events into the calendar collection
            insert_result = calendar.insert_many(events)
            # Convert ObjectIds to strings
            ids = [str(inserted_id) for inserted_id in insert_result.inserted_ids]

        return ids

    except Exception as e:
        return {"error": str(e)}


@router.post("/obligation/setup")
async def documents_process():
    # Fetch all the document text from the Documents Collection
    # Chunk it up, vector search and then get all the Obligations

    # Access the collection
    client = MongoClient(settings.DB_URL)
    db_name = settings.DB_NAME
    collection_name = settings.VECTOR_SEARCH_COLLECTION
    collection = client[db_name][collection_name]

    # Search Index for the chunked agreement data
    vector_search_index = "converse_search_index"

    # Initialise the vector store
    vector_store = MongoDBAtlasVectorSearch(
        embedding=OpenAIEmbeddings(api_key = settings.OPENAI_SECRET_KEY, disallowed_special=()),
        collection=collection,
        index_name=vector_search_index,
    )

    [ docs, document_ids ] = process_all_nav_documents()

    # Add chunks to the Vector Store
    vector_store.add_documents(documents=docs)

    for doc_id in document_ids:
        # Instantiate Atlas Vector Search as a retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                "score_threshold": 0.75,
                "pre_filter": {"document_id": doc_id}
            }
        )
        vector_search_results = retriever.invoke("obligations")

        extraction_builder = StateGraph(ExtractionState)
        extraction_builder.add_node("extract_obligations", extract_obligations)

        extraction_builder.add_edge(START, "extract_obligations")
        extraction_builder.add_edge("extract_obligations", END)

        extraction_graph = extraction_builder.compile()
        extractions = await extraction_graph.ainvoke({"vector_search": vector_search_results})
        print(extractions)

        if extractions:
            # Update the document with the obligation
            document_collection = client[db_name][settings.DOCUMENTS_COLLECTION]
            obligations = extractions["extractions"].obligations

            formatted_obligations = []

            for obligation in obligations:
                formatted_obligation = {
                "obligation_type": obligation.obligation_type,
                "extraction": obligation.extraction,
                "due_date_applicable": obligation.due_date_applicable,
                "due_date": obligation.due_date,
                "actions_needed": obligation.actions_needed,
                }

                formatted_obligations.append(formatted_obligation)

            update_result = document_collection.update_one(
                {"document_id": doc_id},
                {
                    "$set": {
                        "obligations": formatted_obligations
                    },
                }
            )

    return "Completed Obligation Extraction"