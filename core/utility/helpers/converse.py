import os
import uuid
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        document_id = uuid.uuid4()
        async for page in loader.alazy_load():
            page.metadata["document_id"] = document_id
            pages.append(page)
        all_pages.extend(pages)  # Add pages from this document to the overall list

    return all_pages

def create_chunks_with_recursive_split(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(docs)
    return texts

def is_valid_uuid(string):
    try:
        uuid_obj = uuid.UUID(string, version=4)
        return True
    except ValueError:
        return False