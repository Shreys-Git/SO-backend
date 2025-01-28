from pymongo import MongoClient

from config import BaseConfig

settings = BaseConfig()

# Fetch MongoDB Database
client = MongoClient(settings.DB_URL)
db_name = settings.DB_NAME

# Calendar Collection, contains data about upcoming events
calendar_collection = settings.CALENDAR_COLLECTION
calendar = client[db_name][calendar_collection]

# Documents Collection, contains document text, Nav API extracts & other metadata
documents_collection = settings.DOCUMENTS_COLLECTION
documents = client[db_name][documents_collection]

# Tasks Collection, Fetches all the Tasks for the Kanban Board
tasks_collection = settings.TASKS_COLLECTION
tasks = client[db_name][tasks_collection]