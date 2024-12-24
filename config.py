from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class BaseConfig(BaseSettings):
    # MongoDB Configs
    DB_URL: Optional[str]
    DB_NAME: Optional[str]

    # Docusign Configs
    INTEGRATION_KEY: Optional[str]
    CLIENT_SECRET: Optional[str]
    REDIRECT_URL: Optional[str]
    DEV_BASE_PATH: Optional[str]
    API_ACCOUNT_ID: Optional[str]
    TEMPLATE_ID: Optional[str]
    model_config =  SettingsConfigDict(env_file=".env", extra="ignore")

