import base64
import re
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pymongo import MongoClient
from starlette.requests import Request
from starlette.responses import RedirectResponse

from config import BaseConfig
from core.llm.openai import llm
from core.utility.constants import Versions, SignStatus
from core.utility.helpers.documents import update_agreement, find_differences, generate_report_plan, generate_insight, \
    search_web, generate_queries, fetch_agreements, build_redirect_url, format_nav_extractions, send_envelope, \
    get_envelope_status, get_access_code
from routers.docusign import agreements_cache
from schemas.documents import UserPrompt, EditInput, InsightState, InsightAgreement, Document, SignEmail

router = APIRouter()
settings = BaseConfig()


agreements_cache={}
tokens = {}

@router.get("/files/{document_id}/{version}")
async def fetch_mongo_docs(request:Request, document_id: str, version: str):
    client = MongoClient(settings.DB_URL)
    db_name = settings.DB_NAME
    collection_name = settings.DOCUMENTS_COLLECTION
    collection = client[db_name][collection_name]

    if document_id != "ALL":
        search_query = {
            "document_id": document_id,
        }

        result = collection.find_one(search_query, {"_id": 0})

        # Update the document
        file_text = result["document_text"]
        if version != Versions.LATEST.name:
            file_text = result["versions"][int(version)-1]

        result["document_text"] = re.sub(r'\\\\n', '\n', file_text)

        # Poll Docusign and update the Signing Status
        envelope_id = result["signature_metadata"]["envelope_id"]
        sign_status = result["signature_metadata"]["signature_status"]

        status = SignStatus.REVIEW.name
        if envelope_id and sign_status:
            # If the Envelope has been sent for signing before, poll the status
            envelope_status = get_envelope_status(result["signature_metadata"]["envelope_id"],
                                                  tokens["sign_access_token"])
            if envelope_status == SignStatus.COMPLETED.name.lower():
                status = SignStatus.COMPLETED.name
            else:
                status = SignStatus.REVIEW.name

        result["signature_metadata"]["signature_status"] = status

        signature_metadata = {
            "envelope_id": result["signature_metadata"]["envelope_id"],
            "signature_status": result["signature_metadata"]["signature_status"]
        }
        collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "signature_metadata": signature_metadata
                },
            }
        )

        return result
    else:
        results = list(collection.find({}, {"_id": 0}))
        for result in results:
            # Update the document
            file_text = result["document_text"]
            if version != Versions.LATEST.name:
                file_text = result["versions"][int(version) - 1]

            result["document_text"] = re.sub(r'\\\\n', '\n', file_text)

            # Poll Docusign and update the Signing Status
            envelope_id = result["signature_metadata"]["envelope_id"]
            sign_status = result["signature_metadata"]["signature_status"]

            status = SignStatus.REVIEW.name
            if envelope_id and sign_status:
                # If the Envelope has been sent for signing before, poll the status
                envelope_status = get_envelope_status(result["signature_metadata"]["envelope_id"], tokens["sign_access_token"])
                if envelope_status == SignStatus.COMPLETED.name.lower():
                    status = SignStatus.COMPLETED.name
                else:
                    status = SignStatus.REVIEW.name

            result["signature_metadata"]["signature_status"] = status

            signature_metadata = {
                "envelope_id": result["signature_metadata"]["envelope_id"],
                "signature_status": result["signature_metadata"]["signature_status"]
            }
            collection.update_one(
                {"document_id": document_id},
                {
                    "$set": {
                        "signature_metadata": signature_metadata
                    },
                }
            )

        return results


@router.post("/files/{document_id}")
async def fetch_mongo_docs(document_id: str, document: Document):
    client = MongoClient(settings.DB_URL)
    db = client[settings.DB_NAME]
    collection = db[settings.DOCUMENTS_COLLECTION]

    if document_id != "NEW":
        # Find the document by ID
        prev_document = collection.find_one({"document_id": document_id})
        if not prev_document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update the document
        update_result = collection.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_text": document.document_text
                },
                "$push": {
                    "versions": document.document_text
                }
            }
        )
        if update_result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Failed to update the document")
        return {"message": "Document updated successfully"}
    else:
        # Create a new document
        new_document_id = str(uuid.uuid4())
        new_document = {
            "document_id": new_document_id,
            "document_text": document.document_text,
            "navigator_extractions": {},
            "obligations": [],
            "clauses": [],
            "versions": [document.document_text]
        }
        collection.insert_one(new_document)
        return new_document_id

@router.get("/docusign/login")
async def get_doc_access_token():
    # TODO: Combine the two endpoints later & send a query param to tell which access token is needed
    return RedirectResponse(url=build_redirect_url(True))


@router.get("/login")
async def get_nav_access_token():
    # TODO: Combine the two endpoints later & send a query param to tell which access token is needed
    return RedirectResponse(url=build_redirect_url(False))

@router.get("/docusign/callback")
async def callback(request: Request):
    # Extract the authorization code from the URL
    code = request.query_params.get("code")
    if not code:
        return {"error": "No code returned"}
    else:
        print("Received Code " + code)

    token_data = await get_access_code(code)

    # Extract and return the access token
    access_token = token_data.get("access_token")
    if not access_token:
        return {"error": "Could not retrieve access token"}
    else:
        print("Got the final access_token")

    # Cache the token in session
    tokens["sign_access_token"] = access_token
    print(tokens)
    return RedirectResponse(url=f"http://localhost:5173/documents/files")

@router.get("/callback")
async def callback(request: Request):
    # Extract the authorization code from the URL
    code = request.query_params.get("code")
    if not code:
        return {"error": "No code returned"}
    else:
        print("Received Code " )

    token_data = await get_access_code(code)

    # Extract and return the access token
    access_token = token_data.get("access_token")
    if not access_token:
        return {"error": "Could not retrieve access token"}
    else:
        print("Got the final access_token" )

    # Cache the token in session
    tokens["nav_access_token"] = access_token
    print(tokens)
    return RedirectResponse(url=f"http://localhost:5173/documents/files")

@router.post("/sign")
def form_submit(request: Request, email: SignEmail):
    client = MongoClient(settings.DB_URL)
    db_name = settings.DB_NAME
    collection_name = settings.DOCUMENTS_COLLECTION
    collection = client[db_name][collection_name]

    envelope_id = send_envelope(email, tokens["sign_access_token"])
    signature_metadata = {
        "envelope_id": envelope_id,
        "signature_status": SignStatus.SIGN.name
    }
    collection.update_one(
        {"document_id": email.document_id},
        {"$set": {"signature_metadata": signature_metadata}}
    )
    return {"envelope_id": envelope_id}

@router.get("/navigator/{agreement_id}")
async def get_nav_agreements(request: Request, agreement_id: str = None):
    print("Test Print: ", tokens["nav_access_token"])
    agreements = await fetch_agreements(agreement_id, tokens["nav_access_token"])
    if "agreements" not in agreements_cache:
        agreements_cache["agreements"] = agreements
    return agreements


@router.post("/draft")
def langgraph_contract_agent(user_prompt: UserPrompt):
    prompt = "You're a Contract Drafting agent. Goal is to draft a very good contract in great depth based on instruction : " + user_prompt.prompt
    response = llm.invoke(prompt)

    return {"llmContent" : response }

@router.post("/magicEdit")
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

@router.post("/insights")
async def langgraph_contract_agent(insight_input: InsightAgreement):
    report_state = {
        "number_of_queries" : 2,
        "tavily_topic" : "general",
        "tavily_days" : None,
        "insight_type" : insight_input.insight_type,
        "agreement" : insight_input.agreement
    }
    report_plan = await generate_report_plan(report_state)

    # Add nodes and edges
    insight_builder = StateGraph(InsightState)
    insight_builder.add_node("generate_queries", generate_queries)
    insight_builder.add_node("search_web", search_web)
    insight_builder.add_node("generate_insight", generate_insight)

    insight_builder.add_edge(START, "generate_queries")
    insight_builder.add_edge("generate_queries", "search_web")
    insight_builder.add_edge("search_web", "generate_insight")
    insight_builder.add_edge("generate_insight", END)

    insight_builder_graph = insight_builder.compile()

    final_state = None
    for index in range(len(report_plan["insights"])):
        insight = report_plan["insights"][index]
        report_state["insight"] = insight
        print("Initial State is: ", report_state)
        final_state = await insight_builder_graph.ainvoke(report_state)

    return final_state["completed_insights"]

'''
Fetches all the docs processed by Nav API, formats them 
and stores them in MongoDB.
N.B: Needs the user to be logged in
'''
@router.get("/db/setup")
async def process_nav_agreements(request: Request):
    agreements_response = await fetch_agreements("ALL", tokens["nav_access_token"])
    agreements = format_nav_extractions(agreements_response)

    # Insert documents into the database
    db = request.app.db
    collection = db[settings.DOCUMENTS_COLLECTION]
    added_doc = await collection.insert_many(agreements)

    return {"inserted_ids": [str(doc_id) for doc_id in added_doc.inserted_ids]}
