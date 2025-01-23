from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor import motor_asyncio
from starlette.middleware.cors import CORSMiddleware
from routers.RAG import rag_router
from routers.contract import contract_router
from routers.documents import documents_router
from routers.google import google_router
from routers.langgraph_agent import agent_router
from routers.mongo import router as mongo_router
from routers.docusign import docusign_router
from config import BaseConfig

settings = BaseConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.client = motor_asyncio.AsyncIOMotorClient(settings.DB_URL)
    app.db = app.client[settings.DB_NAME]

    try:
        app.client.admin.command("ping")
        print("Success in pinging your db")
    except Exception as e:
        print(e)

    yield
    app.client.close()


app = FastAPI(lifespan=lifespan)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"]
)

app.include_router(mongo_router, prefix="/mongo", tags =["mongo"])
app.include_router(docusign_router, prefix="/docusign", tags=["docusign"])
app.include_router(google_router, prefix="/google", tags=["google"])
app.include_router(rag_router, prefix="/RAG", tags=["RAG"])
app.include_router(agent_router, prefix="/agent", tags=["agent"])
app.include_router(contract_router, prefix="/contract", tags=["contract"])
app.include_router(documents_router, prefix="/documents", tags=["documents"])