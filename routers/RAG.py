import pprint
import uuid
from typing import List

from fastapi import APIRouter, Query
from huggingface_hub import InferenceClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel
from fastapi import Request
from pymongo import MongoClient

from config import BaseConfig
from routers.mongo import load_documents, create_chunks_with_recursive_split
from utils.vector_utils import get_embedding
from bson.codec_options import CodecOptions

rag_router = APIRouter()
settings = BaseConfig()

chat_history =[]

@rag_router.get("/llm/chat")
async def chat_llm(request: Request):
    input_query = "loan data"

    # Define the pipeline
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_test_index',
                'path': 'embedding',
                'queryVector': get_embedding(input_query),  # Ensure get_embedding is compatible
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

    # Authenticate to Hugging Face and access the model
    llm = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.3",
        token= settings.HF_ACCESS_TOKEN)

    # Prompt the LLM (this code varies depending on the model you use)
    output = llm.chat_completion(
        messages=[{"role": "user", "content": input_query}], #TODO: Send the context as a string + query Input in a formatted way to the LLM (RAG)
        max_tokens=150
    )


    # Return results
    return {"LLM Results": output.choices[0].message.content}


class UserChatMessage(BaseModel):
    message: str
    agreement_id : List[str]

@rag_router.post("/llm/chat")
async def chat_llm(request: Request, use_chat_message: UserChatMessage):
    input_query = use_chat_message.message
    print("User Query is: " + input_query)
    # Define the pipeline
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_test_index',
                'path': 'embedding',
                'queryVector': get_embedding(input_query),  # Ensure get_embedding is compatible
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

    # Authenticate to Hugging Face and access the model
    llm = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.3",
        token= settings.HF_ACCESS_TOKEN)

    # Prompt the LLM (this code varies depending on the model you use)
    output = llm.chat_completion(
        messages=[{"role": "user", "content": input_query}], #TODO: Send the context as a string + query Input in a formatted way to the LLM (RAG)
        max_tokens=150
    )

    print("The AI Response is: " + output.choices[0].message.content)
    # Return results
    return {"AIResponse": output.choices[0].message.content}

@rag_router.post("/llm/chat")
async def chat_llm(request: Request, use_chat_message: UserChatMessage):
    input_query = use_chat_message.message
    print("User Query is: " + input_query)
    # Define the pipeline
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_test_index_30',
                'path': 'embedding',
                'queryVector': get_embedding(input_query),  # Ensure get_embedding is compatible
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

    # Authenticate to Hugging Face and access the model
    llm = InferenceClient(
        "mistralai/Mistral-7B-Instruct-v0.3",
        token= settings.HF_ACCESS_TOKEN)

    # Prompt the LLM (this code varies depending on the model you use)
    output = llm.chat_completion(
        messages=[{"role": "user", "content": input_query}], #TODO: Send the context as a string + query Input in a formatted way to the LLM (RAG)
        max_tokens=150
    )

    print("The AI Response is: " + output.choices[0].message.content)
    # Return results
    return {"AIResponse": output.choices[0].message.content}

@rag_router.post("/llm/chat/v2")
async def chat_llm(request: Request, use_chat_message: UserChatMessage):
    input_query = use_chat_message.message
    print("User Query is: " + input_query)
    # Define the pipeline
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_test_index_30',
                'path': 'embedding',
                'queryVector': get_embedding(input_query),  # Ensure get_embedding is compatible
                'numCandidates': 20,
                'limit': 2
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
        search_results.append(res.get("text", ""))

    llm_context = " ".join(search_results)

    template = f"""Question: {input_query}
    Answer using the following context: {llm_context}. 
    If you don't find the answer, say I don't know please."""

    prompt = PromptTemplate.from_template(template)

    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.5,
        huggingfacehub_api_token= settings.HF_ACCESS_TOKEN,
    )

    llm_chain = prompt | llm
    response = llm_chain.invoke({"input_query": input_query, "llm_context" : llm_context})
    print(response)

    # Return results
    return {"AIResponse": response}


@rag_router.post("/llm/chat/v3")
async def chat_llm(request: Request, use_chat_message: UserChatMessage):
    input_query = use_chat_message.message
    print("User Query is: " + input_query)

    # # One time set up for the Vector Store create
    folder_path = "./SampleAgreements/Sample Documents for Navigator/Others"
    # docs = await load_documents(folder_path)
    # chunks = create_chunks_with_recursive_split(docs, 1000, 200)

    # Create the vector store
    hf_embeddings = HuggingFaceEndpointEmbeddings(
        model= "sentence-transformers/all-mpnet-base-v2",
        model_kwargs = {"trust_remote_code": True},
        task="feature-extraction",
        huggingfacehub_api_token=settings.HF_ACCESS_TOKEN,
    )

    # Access the collection
    client = MongoClient(settings.DB_URL)
    db_name = settings.DB_NAME
    collection_name = "ShrsingTestVSCollection"
    collection = client[db_name][collection_name]

    # Create the vector store asynchronously
    # vector_store = MongoDBAtlasVectorSearch.from_documents(
    #     documents=chunks,
    #     embedding=hf_embeddings,
    #     collection=collection,
    #     index_name="vector_test_index_30"
    # )
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=hf_embeddings,
        index_name="vector_test_index_30",
    )
    # document_1 = Document(page_content="foo", metadata={"baz": "bar"})
    # document_2 = Document(page_content="thud", metadata={"bar": "baz"})
    # document_3 = Document(page_content="i will be deleted :(")
    #
    # documents = [document_1, document_2, document_3]

    # Documentation on CRUD: https://api.python.langchain.com/en/latest/vectorstores/langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch.html
    # ids = vector_store.add_documents(documents=chunks)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

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


    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.5,
        huggingfacehub_api_token= settings.HF_ACCESS_TOKEN,
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

    ai_msg = rag_chain.invoke({"input": input_query, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=input_query),
            AIMessage(content=ai_msg["answer"]),
        ]
    )

    return {"response" : ai_msg}



@rag_router.get("/llm/chat/v4")
async def chat_llm(use_chat_message: UserChatMessage):
    docs = await load_documents()
    chunks = create_chunks_with_recursive_split(docs, 1000, 200)

    # Access the collection
    client = MongoClient(settings.DB_URL, uuidRepresentation="standard")
    db_name = settings.DB_NAME
    collection_name = "ShrsingCollection20012025"
    collection = client[db_name][collection_name]

    vector_search_index = "vector_index_20012025"

    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(api_key = settings.OPENAI_SECRET_KEY, disallowed_special=()),
        collection=collection,
        index_name=vector_search_index
    )

    vector_store.create_vector_search_index(
        dimensions=1536,
        filters=["document_id"]
    )

    return "success"

@rag_router.get("/llm/chat/v4/search")
async def chat_llm(use_chat_message: UserChatMessage):
    # Access the collection
    client = MongoClient(settings.DB_URL, uuidRepresentation="standard")
    db_name = settings.DB_NAME
    collection_name = "ShrsingCollection20012025"
    collection = client[db_name][collection_name]

    vector_search_index = "vector_index_20012025"

    vector_store = MongoDBAtlasVectorSearch(
        embedding=OpenAIEmbeddings(api_key = settings.OPENAI_SECRET_KEY, disallowed_special=()),
        collection=collection,
        index_name=vector_search_index,
    )

    # Instantiate Atlas Vector Search as a retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "score_threshold": 0.75,
            "pre_filter": {"document_id": {"$eq": uuid.UUID(use_chat_message.agreement_id[0])}}
        }
    )

    # Process Chat history

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        max_retries=1,
        api_key=settings.OPENAI_SECRET_KEY
    )

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

    ai_msg = rag_chain.invoke({"input": use_chat_message.message, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=use_chat_message.message),
            AIMessage(content=ai_msg["answer"]),
        ]
    )

    return {"response": ai_msg}


