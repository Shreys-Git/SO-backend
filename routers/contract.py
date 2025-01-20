from fastapi import APIRouter
from pydantic import BaseModel

from config import BaseConfig
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import asyncio
import operator
from typing_extensions import TypedDict
from typing import  Annotated, List, Optional, Literal
from pydantic import BaseModel, Field

from tavily import TavilyClient, AsyncTavilyClient

from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage
from difflib import unified_diff

from langchain_core.runnables import RunnableConfig
from IPython.display import Markdown

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langsmith import traceable
from difflib import Differ
import re


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
    prompt: str
    agreement: str

class AIEdit(BaseModel):
  original_agreement_text: str = Field(
        description="Contains the exact original legal agreememt provided by the user",
    )
  updated_agreement_text: str = Field(
        description="Contains the updated response",
    )
  update_summary: str = Field(
        description="Summary of the changes made described using bullet points",
    )

class SectionState(TypedDict):
    section: AIEdit
    prompt: str
    agreement_text: str
    updated_agreement_text: str

# Magic Edit instructions
magic_edit_instructions="""You are an expert law analyst editing a legal agreement based on the instructions given to you.

Change Instructions:
{prompt}

Legal Agreement:
{agreement}

Guidelines for writing:

1. Technical Accuracy:
- Use technical terminology precisely

2. Length and Style:
- Keep the suggested changes word length similar to the original document
- No marketing language
- Technical focus
- Write in simple, clear language

3. Writing Approach:
- Use concrete details over general statements
- Make every word count
- No preamble prior to creating the section content
- Focus on your single most important point

4. Quality Checks:
- No preamble prior to creating the section content"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    max_retries=1,
    api_key=settings.OPENAI_SECRET_KEY,
)

def update_agreement(state: SectionState):
    """ Update a section of the report """

    # Get state
    agreement = state["agreement_text"]
    prompt = state["prompt"]

    # Format system instructions
    system_instructions = magic_edit_instructions.format(prompt=prompt, agreement=agreement)

    # Update section
    section_content = llm.with_structured_output(AIEdit).invoke([SystemMessage(content=system_instructions)]+[HumanMessage(content="Update the given documents based on the provided input")])
    return {"response": section_content}

def find_differences(original_text, updated_text):
    lines1 = original_text.splitlines()
    lines2 = updated_text.splitlines()

    # Create a Differ object and compare the lines
    differ = Differ()
    diff = differ.compare(lines1, lines2)

    formatted_diffs = []
    for line in diff:
        if not line.startswith("?"):
            formatted_diffs.append(line)
        if line.strip() == "":
            # Replace spaces after the first one with '\n'
            formatted_diffs.append(line[0] + re.sub(r' +', '\n', line[1:]))

    return formatted_diffs

@contract_router.post("/langgraph/magicEdit")
def langgraph_contract_agent(edit_input: EditInput):
    print("Agreement: \n\n", edit_input.agreement)

    updated_response = update_agreement({
    "agreement_text": edit_input.agreement,
    "prompt": edit_input.prompt
    })

    ai_response = updated_response["response"]
    original_agreement = ai_response.original_agreement_text
    updated_agreement = ai_response.updated_agreement_text
    update_summary = ai_response.update_summary

    differences = find_differences(original_agreement, updated_agreement)

    formatted_response = {
        "updated_agreement": updated_agreement,
        "update_summary": update_summary,
        "differences": differences,
    }

    return formatted_response



