from langchain_openai import ChatOpenAI

from config import BaseConfig

settings = BaseConfig()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    max_retries=1,
    api_key=settings.OPENAI_SECRET_KEY
)