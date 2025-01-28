from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor import motor_asyncio
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.v1.endpoints.documents import router as documents_router_v1
from api.v1.endpoints.users import router as users_router
from api.v1.endpoints.converse import router as converse_router
from api.v1.endpoints.calendar import router as calender_router
from api.v1.endpoints.tasks import router as task_router

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

app.add_middleware(SessionMiddleware, secret_key="secret-key")

app.include_router(documents_router_v1, prefix="/v1/documents")
app.include_router(users_router, prefix="/v1/users")
app.include_router(converse_router, prefix="/v1/converse")
app.include_router(calender_router, prefix="/v1/calendar")
app.include_router(task_router, prefix="/v1/tasks")

