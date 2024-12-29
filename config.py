from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class BaseConfig(BaseSettings):
    # MongoDB Configs
    DB_URL: Optional[str]
    DB_NAME: Optional[str]
    VECTOR_COLLECTION_NAME: Optional[str]
    NAV_COLLECTION_NAME: Optional[str]

    # Docusign Configs
    INTEGRATION_KEY: Optional[str]
    CLIENT_SECRET: Optional[str]
    REDIRECT_URL: Optional[str]
    DEV_BASE_PATH: Optional[str]
    API_ACCOUNT_ID: Optional[str]
    TEMPLATE_ID: Optional[str]
    model_config =  SettingsConfigDict(env_file=".env", extra="ignore")

    # Hugging Face LLM
    HF_ACCESS_TOKEN: Optional[str]

    # Google Configs
    GOOGLE_CLIENT_ID: Optional[str]
    GOOGLE_PROJECT_ID: Optional[str]
    GOOGLE_AUTH_URI: Optional[str]
    GOOGLE_TOKEN_URI: Optional[str]
    GOOGLE_AUTH_PROVIDER_CERT: Optional[str]
    GOOGLE_CLIENT_SECRET: Optional[str]
    GOOGLE_REDIRECT_URI: Optional[str]

