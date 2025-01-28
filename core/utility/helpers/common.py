import uuid

from langchain_core.documents import Document
from pymongo import MongoClient

from config import BaseConfig
from core.utility.helpers.converse import create_chunks_with_recursive_split

settings = BaseConfig()

def process_all_nav_documents():
    client = MongoClient(settings.DB_URL, uuidRepresentation="standard")
    db_name = settings.DB_NAME
    document_collection = client[db_name][settings.DOCUMENTS_COLLECTION]

    # Fetch all the stored Documents
    mongo_documents = list(document_collection.find({}, {"_id:": 0}))

    # Convert Mongo Documents to Langchain Documents for processing
    langchain_documents = []
    document_ids = []
    for doc in mongo_documents:
        document_ids.append(doc.get("document_id", "Document ID not found"))

        langchain_document = Document(page_content=doc.get("document_text", ""),
                                      metadata={
                                          "document_id": doc.get("document_id", str(uuid.uuid4())),
                                          "version": len(doc.get("versions", 1)),
                                          "file_name": doc.get("navigator_extractions", {}).get("file_name",
                                                                                                "UnknownFile.pdf"),
                                          "file_type": doc.get("navigator_extractions", {}).get("category",
                                                                                                "Unknown Category")
                                      })

        langchain_documents.append(langchain_document)

    # Recursively chunk to prevent Context Window Breach & better vector search results
    chunks = create_chunks_with_recursive_split(langchain_documents, 1000, 200)

    return [chunks, document_ids]


