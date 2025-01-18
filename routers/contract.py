from fastapi import APIRouter
from pydantic import BaseModel

from config import BaseConfig
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI

contract_router = APIRouter()
settings = BaseConfig()

class UserPrompt(BaseModel):
    prompt: str

@contract_router.post("/langgraph/llm")
def langgraph_contract_agent(user_prompt: UserPrompt):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        max_retries=1,
        api_key=settings.OPENAI_SECRET_KEY
    )

    prompt = "You're a Contract Drafting agent. Goal is to draft a very good contract in great depth based on instruction : " + user_prompt.prompt
    response = llm.invoke(prompt)

    return {"llmContent" : response }

class EditInput(BaseModel):
    change: str
    document: str

@contract_router.post("/langgraph/magicEdit")
def langgraph_contract_agent(edit_input: EditInput):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        max_retries=1,
        api_key=settings.OPENAI_SECRET_KEY
    )

    # prompt = (f"You're a lawyer at the top law firm and your goal is, given a contractual document, and the corresponding "
    #           "requested change, you need to go through the document and return the changes in a key-value format containing "
    #           " the specific lines being changed in the text as original_text and the updated text as updated_text. "
    #           "You can change as many lines as you'd like to ful fill the request changes but only return the key-value with original_text and updated_text"
    #           ", nothing else. Only add text to the key-value where text is actually updated. "
    #           " Change: {change} Document: {document}").format(change = edit_input.change, document = edit_input.document)
    #
    # response = llm.invoke(prompt)
    # prompt_format = (f"Given this string, remove any of the entries that contain the same value for the original_text and the updated_text and "
    #                  f"in the form of key value pairs of original_text and updated_text"
    #                  f"Only return the key-value, don't need explanation."
    #                  f"Dont' apply any format. Simply return as key-value pairs. String {response}")

    # formatted_response = llm.invoke(prompt_format)

    # return {"llmContent" : formatted_response.content}
    return {"data" : [
    {
        "original_text": "James Joseph Rose",
        "updated_text": "Amelia Hart Gold"
    }
]
}