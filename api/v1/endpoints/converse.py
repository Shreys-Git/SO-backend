import io
import time
import uuid
from typing import List, Optional

from PyPDF2 import PdfReader
from fastapi import APIRouter, Form, UploadFile
from fastapi.params import File
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_mongodb import MongoDBAtlasVectorSearch, MongoDBChatMessageHistory
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

from config import BaseConfig
from core.llm.openai import llm
from core.utility.helpers.converse import load_documents, create_chunks_with_recursive_split, is_valid_uuid
from schemas.converse import UserChatMessage

router = APIRouter()
settings = BaseConfig()

@router.post("/upload")
async def upload_additional_docs(additional_docs: List[UploadFile] = File([])):
    # Access the collection
    client = MongoClient(settings.DB_URL, uuidRepresentation="standard")
    db_name = settings.DB_NAME
    collection = client[db_name][settings.VECTOR_SEARCH_COLLECTION]

    # Search Index for the chunked agreement data
    vector_search_index = "converse_search_index"

    # Initialise the vector store
    vector_store = MongoDBAtlasVectorSearch(
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_SECRET_KEY, disallowed_special=()),
        collection=collection,
        index_name=vector_search_index,
    )

    # Convert Additional Documents(Pdf/Word) to LangChain Documents
    docs = []
    document_ids = []
    for doc in additional_docs:
        file_content = await doc.read()
        file_bytes = io.BytesIO(file_content)
        pdf_reader = PdfReader(file_bytes)

        document_id = uuid.uuid4()
        document_ids.append(str(document_id))
        for page in pdf_reader.pages:
            text = page.extract_text()
            page_doc = Document(page_content=text, metadata={"document_id": document_id})
            docs.append(page_doc)

    # Chunk (if applicable)
    chunks = create_chunks_with_recursive_split(docs, 1000, 200)
    # Add chunks to the Vector Store
    vector_store.add_documents(documents=chunks)

    # Sleep, so that the MongoDB can initialize the Index,
    # should ideally query Mongo to check the status / webhook it - but ran out of time : )
    time.sleep(5)
    return document_ids

@router.post("/rag")
async def chat_llm(user_message: UserChatMessage):
    print("The document Ids recieved are: ", user_message.document_ids)
    # Validate the document Ids
    for current_id in user_message.document_ids:
        if not is_valid_uuid(current_id): return False

    # Access the collection
    client = MongoClient(settings.DB_URL, uuidRepresentation="standard")
    db_name = settings.DB_NAME
    collection = client[db_name][settings.VECTOR_SEARCH_COLLECTION]

    # Search Index for the chunked agreement data
    vector_search_index = "converse_search_index"

    # Initialise the vector store
    vector_store = MongoDBAtlasVectorSearch(
        embedding=OpenAIEmbeddings(api_key=settings.OPENAI_SECRET_KEY, disallowed_special=()),
        collection=collection,
        index_name=vector_search_index,
    )
    # Instantiate Atlas Vector Search as a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.75,
            "pre_filter": {
                "document_id": {"$in": [uuid.UUID(id) for id in user_message.document_ids]}
            }
        }
    )
    print("closest results", retriever.invoke(user_message.message))

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    chat_history = MongoDBChatMessageHistory(
        session_id=user_message.session_id,  # Unique session identifier
        connection_string=settings.DB_URL,  # Atlas cluster or local MongoDB instance URI
        database_name=db_name,  # Database to store the chat history
        collection_name=settings.CONVERSATION_HISTORY_COLLECTION  # Collection to store the chat history
    )

    ai_msg = rag_chain.invoke({"input": user_message.message, "chat_history": chat_history.messages})

    chat_history.add_user_message(user_message.message)
    chat_history.add_ai_message(ai_msg["answer"])

    return ai_msg

@router.get("/index/setup")
async def chat_llm():
    # Access the collection
    client = MongoClient(settings.DB_URL, uuidRepresentation="standard")
    db_name = settings.DB_NAME
    document_collection = client[db_name][settings.DOCUMENTS_COLLECTION]

    mongo_documents = list(document_collection.find({}, {"_id:": 0}))
    langchain_documents = []
    for doc in mongo_documents:
        langchain_document = Document(page_content=doc.get("document_text", ""),
                                      metadata={
                                          "document_id" : doc.get("document_id", str(uuid.uuid4())),
                                          "version": len(doc.get("versions", 1)),
                                          "file_name": doc.get("navigator_extractions", {}).get("file_name", "UnknownFile.pdf"),
                                          "file_type": doc.get("navigator_extractions", {}).get("category", "Unknown Category")
                                      })

        langchain_documents.append(langchain_document)

    chunks = create_chunks_with_recursive_split(langchain_documents, 1000, 200)

    search_collection = client[db_name][settings.VECTOR_SEARCH_COLLECTION]
    vector_search_index = "converse_search_index"

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(api_key = settings.OPENAI_SECRET_KEY, disallowed_special=()),
        collection=search_collection,
        index_name=vector_search_index
    )

    vector_store.create_vector_search_index(
        dimensions=1536,
        filters=["document_id"]
    )

    return "Index has been setup"



