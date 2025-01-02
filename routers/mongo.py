from typing import List

from fastapi import APIRouter
from fastapi import Request
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
from pathlib import Path
import os


from config import BaseConfig
from utils.vector_utils import get_embedding

router = APIRouter()
settings = BaseConfig()


'''
Loads the documents from Pdf Location
Creates Embeddings from the documents 
Stores into the MongoDB database
'''
@router.get("/pdf/document/process")
async def load_documents_endpoint(request: Request):
    docs = await process_documents()

    # Insert documents into the database
    db = request.app.db
    collection = db[settings.VECTOR_COLLECTION_NAME]
    added_doc = await collection.insert_many(docs)

    return {"inserted_ids": [str(doc_id) for doc_id in added_doc.inserted_ids]}

async def process_documents():
    docs = await load_documents()
    chunks = create_chunks_with_recursive_split(docs, 100, 0)


    # Prepare documents for insertion
    docs_to_insert = [
        {
            "text": chunk.page_content,
            "embedding": get_embedding(chunk.page_content),
        }
        for chunk in chunks
    ]

    return docs_to_insert

async def load_documents():
    folder_path = "./SampleAgreements"
    # Find all PDF files recursively in the given folder
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):  # Ensure case-insensitive match for .pdf
                pdf_files.append(Path(root) / file)

    # Load each PDF file and collect pages
    all_pages = []
    for file_path in pdf_files:
        loader = PyPDFLoader(str(file_path))
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        all_pages.extend(pages)  # Add pages from this document to the overall list

    return all_pages


def create_chunks_with_fixed_token_split(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Fixed token chunking
    Args:
        docs (List[Document]): List of documents to chunk
        chunk_size (int): Chunk size (number of tokens)
        chunk_overlap (int): Token overlap between chunks
    Returns:
        List[Document]: List of chunked documents
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def create_chunks_with_recursive_split(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    return texts



@router.get("/searchIndex/create")
async def create_vector_search_index(request: Request):

    # Define the search index model
    index_name = "vector_test_index"
    search_index_model = {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 768, # TODO: Add the correct number of Dimension (size Vector / Docs for the Embedding Model)
                    "path": "embedding",
                    "similarity": "cosine"
                }
            ]
        }
    }

    # Create the search index
    db = request.app.db
    collection = db[settings.VECTOR_COLLECTION_NAME]
    await collection.create_search_index(search_index_model)
    print("Search index creation initiated.")
    return {"result" : "success"}

@router.get("/vector/search")
async def search_vector_db(request: Request):
    # Define the pipeline
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_test_index',
                'path': 'embedding',
                'queryVector': get_embedding("loan data"),  # Ensure get_embedding is compatible
                'numCandidates': 100,
                'limit': 10
            }
        },
        {
            '$project': {
                '_id': 0,
                'text': 1,
                'score': {
                    '$meta': 'vectorSearchScore'
                }
            }
        }
    ]

    # Access the collection
    db = request.app.db
    collection = db[settings.VECTOR_COLLECTION_NAME]

    # Run the aggregation pipeline
    results = collection.aggregate(pipeline)

    # Collect results asynchronously
    search_results = []
    async for res in results:
        search_results.append(res)

    # Return results
    return {"Found Results": search_results}