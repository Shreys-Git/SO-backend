import operator
import re
from collections import defaultdict
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, TypedDict, Annotated

import httpx
import base64

import os
from pathlib import Path
import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from docusign_esign import ApiClient, EnvelopesApi, EnvelopeDefinition, TemplateRole
from fastapi import FastAPI, Request
from langchain_community.chat_models import ChatOpenAI
from langchain_core.agents import AgentAction
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from motor import motor_asyncio
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from langchain_core.tools import tool





from config import BaseConfig

settings = BaseConfig()

tokens = {}
agreements_cache = {}
# Initialize the stats dictionary
agreements_stats = defaultdict(int)

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

class APIScope(Enum):
    ESIGNATURE = "signature"
    NAVIGATOR = "adm_store_unified_repo_read"

# Load the embedding model (https://huggingface.co/nomic-ai/nomic-embed-text-v1")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)


'''
Re-directs User to the correct Docusign Sign In endpoint
Note: Scope changes based on whether it's e-sign or navigator redirect
'''
@app.get("/login")
async def login():
    return RedirectResponse(url= build_redirect_url(True))

def build_redirect_url(is_esign):
    scope = APIScope.ESIGNATURE
    if not is_esign:
        scope = APIScope.NAVIGATOR.value

    # Redirect user to third-party authorization endpoint
    # TODO: set the base url as an env var (N.B: This is the dev-endpoint)
    AUTHORIZE_URL="https://account-d.docusign.com/oauth/auth"
    params = {
        "client_id": settings.INTEGRATION_KEY,
        "redirect_uri": settings.REDIRECT_URL,
        "scope": scope,  # Define the permissions you need
        "response_type": "code",
    }
    query = "&".join([f"{key}={value}" for key, value in params.items()])
    redirect_url = f"{AUTHORIZE_URL}?{query}"

    print("The redirect-url is: " + redirect_url)
    return redirect_url


'''
This is the call-back endpoint which Docusign returns the code to
Once this has recieved the code, it will exchange it for the access-token
N.B: To change this endpoint, we MUST also add the new-endpoint to the REDIRECT URIs 
section in Docusign
N.B: No change here for esign v/s navigator
'''
@app.get("/")
async def callback(request: Request):
    # Extract the authorization code from the URL
    code = request.query_params.get("code")
    if not code:
        return {"error": "No code returned"}
    else:
        print("Received Code " + code)

    # Exchange the authorization code for an access token
    TOKEN_URL = "https://account-d.docusign.com/oauth/token"
    auth_key = f"{settings.INTEGRATION_KEY}:{settings.CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_key.encode("ascii")).decode("ascii")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            headers={
                "Authorization": f"Basic {encoded_auth}",
                "Accept": "application/json"
            },
            data={
                "grant_type": "authorization_code",
                "code": code,
            },
        )
        token_data = response.json()

    # Extract and return the access token
    access_token = token_data.get("access_token")
    if not access_token:
        return {"error": "Could not retrieve access token"}

    # Temporary in-memory cache for the access token
    if "access_token" not in tokens:
        tokens["access_token"] = access_token

    return {"access_token": access_token}

def cache_access_token():
    # TODO: Pass the query param here as well, that way we know which key to map the access token to
    # signature-token v/s nav token v/s any other that we end up adding eventually
    pass


class FormData(BaseModel):
    name: str
    email: str

''' Use the E-signature SDK to send the envelope.
This will send the Document to the user via an email & returns the ID 
as soon as the email is sent (doesn't wait for the sign to complete)
N.B: The SDK does have functionality to track the status of the sign'''
@app.post("/submit")
def form_submit(form_data: FormData):
    envelope_id = send_envelope(form_data.name, form_data.email)
    return {"envelope_id": envelope_id}


def send_envelope(name: str, email: str):
    # 1. Create the envelope request object
    envelope_definition = make_envelope(name, email)

    # 2. call Envelopes::create API method
    api_client = create_api_client()

    envelope_api = EnvelopesApi(api_client)
    results = envelope_api.create_envelope(settings.API_ACCOUNT_ID, envelope_definition=envelope_definition)
    envelope_id = results.envelope_id
    return envelope_id

def make_envelope(signer_name, signer_email):
    # create the envelope definition
    envelope_definition = EnvelopeDefinition(
        status="sent",  # requests that the envelope be created and sent.
        template_id= settings.TEMPLATE_ID
    )

    # Create template role elements to connect the signer and cc recipients
    # to the template
    signer = TemplateRole(
        email=signer_email,
        name=signer_name,
        role_name="TestRole"
    )

    # Add the TemplateRole objects to the envelope object
    envelope_definition.template_roles = [signer]
    return envelope_definition

def create_api_client():
    api_client = ApiClient()
    api_client.host = settings.DEV_BASE_PATH
    api_client.set_default_header(header_name= "Authorization", header_value=f"Bearer {tokens["access_token"]}")

    return api_client


'''
Endpoint to get the access token for the Navigator API
'''
@app.get("/navigator/login")
async def get_nav_access_token():
    # TODO: Combine the two endpoints later & send a query param to tell which access token is needed
    return RedirectResponse(url=build_redirect_url(False))


@app.get("/navigator/agreements/stats")
async def create_nav_agreement_stats():
    agreements_response = await fetch_agreements("ALL")
    agreements = agreements_response.get("agreements", [])

    for agreement in agreements:
        type_key = agreement.get("type", "type_missing")
        agreements_stats[type_key] += 1
        category_key = agreement.get("category", "category_missing")
        agreements_stats[category_key] += 1

        for party in agreement.get("parties", []):
            party_key = party.get("name_in_agreement", "name_in_agreement_missing")
            agreements_stats[party_key] += 1

        for provision_key, provision_value in agreement.get("provisions", {}).items():
            if provision_key == "effective_date":
                effective_date_key = provision_value if provision_value else "effective_date_missing"
                print(f"Effective Date Key: {effective_date_key}")
                agreements_stats[effective_date_key] += 1

            elif provision_key == "expiration_date":
                expiration_date_key = provision_value if provision_value else "expiration_date_missing"
                print(f"Expiration Date Key: {expiration_date_key}")
                agreements_stats[expiration_date_key] += 1

            elif provision_key == "execution_date":
                execution_date_key = provision_value if provision_value else "execution_date_missing"
                print(f"Execution Date Key: {execution_date_key}")
                agreements_stats[execution_date_key] += 1

            elif provision_key == "assignment_type":
                assignment_type_key = provision_value if provision_value else "assignment_type_missing"
                print(f"Assignment Type Key: {assignment_type_key}")
                agreements_stats[assignment_type_key] += 1

            elif provision_key == "assignment_termination_rights":
                assignment_termination_rights_key = provision_value if provision_value else "assignment_termination_rights_missing"
                print(f"Assignment Termination Rights Key: {assignment_termination_rights_key}")
                agreements_stats[assignment_termination_rights_key] += 1

            elif provision_key == "governing_law":
                governing_law_key = provision_value if provision_value else "governing_law_missing"
                print(f"Governing Law Key: {governing_law_key}")
                agreements_stats[governing_law_key] += 1

            elif provision_key == "jurisdiction":
                jurisdiction_key = provision_value if provision_value else "jurisdiction_missing"
                print(f"Jurisdiction Key: {jurisdiction_key}")
                agreements_stats[jurisdiction_key] += 1

            elif provision_key == "payment_terms_due_date":
                payment_terms_due_date_key = provision_value if provision_value else "payment_terms_due_date_missing"
                print(f"Payment Terms Due Date Key: {payment_terms_due_date_key}")
                agreements_stats[payment_terms_due_date_key] += 1

            elif provision_key == "can_charge_late_payment_fees":
                can_charge_late_payment_fees_key = provision_value if provision_value else "can_charge_late_payment_fees_missing"
                print(f"Can Charge Late Payment Fees Key: {can_charge_late_payment_fees_key}")
                agreements_stats[can_charge_late_payment_fees_key] += 1

            elif provision_key == "liability_cap_multiplier":
                liability_cap_multiplier_key = provision_value if provision_value else "liability_cap_multiplier_missing"
                print(f"Liability Cap Multiplier Key: {liability_cap_multiplier_key}")
                agreements_stats[liability_cap_multiplier_key] += 1

            elif provision_key == "liability_cap_duration":
                liability_cap_duration_key = provision_value if provision_value else "liability_cap_duration_missing"
                print(f"Liability Cap Duration Key: {liability_cap_duration_key}")
                agreements_stats[liability_cap_duration_key] += 1

            elif provision_key == "renewal_type":
                renewal_type_key = provision_value if provision_value else "renewal_type_missing"
                print(f"Renewal Type Key: {renewal_type_key}")
                agreements_stats[renewal_type_key] += 1

            elif provision_key == "termination_period_for_cause":
                termination_period_for_cause_key = provision_value if provision_value else "termination_period_for_cause_missing"
                print(f"Termination Period for Cause Key: {termination_period_for_cause_key}")
                agreements_stats[termination_period_for_cause_key] += 1

            elif provision_key == "termination_period_for_convenience":
                termination_period_for_convenience_key = provision_value if provision_value else "termination_period_for_convenience_missing"
                print(f"Termination Period for Convenience Key: {termination_period_for_convenience_key}")
                agreements_stats[termination_period_for_convenience_key] += 1

            else:
                print(f"Unhandled Provision Key: {provision_key}, Value: {provision_value}")

        for language in agreement.get("languages", []):
            language_key = language
            agreements_stats[language_key] += 1

    return agreements_stats

'''
Fetches all the docs processed by Nav API
Stores them in MongoDB 
'''
@app.get("/navigator/agreements/process")
async def process_nav_agreements(request: Request):
    agreements_response = await fetch_agreements("ALL")
    agreements = agreements_response.get("agreements", [])

    if "agreements" not in agreements_cache:
        agreements_cache["agreements"] = agreements

    # Insert documents into the database
    db = request.app.db
    collection = db[settings.NAV_COLLECTION_NAME]
    added_doc = await collection.insert_many(agreements)

    return {"inserted_ids": [str(doc_id) for doc_id in added_doc.inserted_ids]}


@app.get("/navigator/agreements/{agreement_id}")
async def get_nav_agreements(agreement_id: str = None):
    #TODO: Don't fetch the doc before checking the cache -> Stale cache ? (Only fixed amt of docs in the Nav Docs)
    agreements = await fetch_agreements(agreement_id)
    if "agreements" not in agreements_cache:
        agreements_cache["agreements"] = agreements
    return agreements

async def fetch_agreements(agreement_id: str):
    GET_ALL_NAV_AGREEMENTS_URL = f"https://api-d.docusign.com/v1/accounts/{settings.API_ACCOUNT_ID}/agreements"

    if agreement_id != "ALL":
        print(f"Id is present: {agreement_id}")
        GET_ALL_NAV_AGREEMENTS_URL = f"https://api-d.docusign.com/v1/accounts/{settings.API_ACCOUNT_ID}/agreements/{agreement_id}"
        print(f"The endpoint being called is: {GET_ALL_NAV_AGREEMENTS_URL}")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                GET_ALL_NAV_AGREEMENTS_URL,
                headers={
                    "Authorization": f"Bearer {tokens['access_token']}",
                    "Accept": "application/json"
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return {"error": str(e)}

    agreements = response.json()
    nav_agreements = agreements.get("data", [])

    if agreement_id != "ALL":
        nav_agreements = agreements

    if not nav_agreements:
        print("No agreements found or `data` key missing in response.")

    return {"agreements": nav_agreements}

'''
Loads the documents from Pdf Location
Creates Embeddings from the documents 
Stores into the MongoDB database
'''
@app.get("/pdf/document/process")
async def load_documents_endpoint(request: Request):
    docs = await process_documents()

    # Insert documents into the database
    db = request.app.db
    collection = db[settings.VECTOR_COLLECTION_NAME]
    added_doc = await collection.insert_many(docs)

    return {"inserted_ids": [str(doc_id) for doc_id in added_doc.inserted_ids]}

async def process_documents():
    docs = await load_documents()
    chunks = create_chunks_with_fixed_token_split(docs, 100, 0)

    # Prepare documents for insertion
    docs_to_insert = [
        {
            "text": chunk.page_content,
            "embedding": get_embedding(chunk.page_content),
        }
        for chunk in chunks
    ]

    return docs_to_insert

async def load_documents():
    folder_path = "./SampleAgreements"
    # Find all PDF files recursively in the given folder
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):  # Ensure case-insensitive match for .pdf
                pdf_files.append(Path(root) / file)

    # Load each PDF file and collect pages
    all_pages = []
    for file_path in pdf_files:
        loader = PyPDFLoader(str(file_path))
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        all_pages.extend(pages)  # Add pages from this document to the overall list

    return all_pages


def create_chunks_with_fixed_token_split(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Fixed token chunking
    Args:
        docs (List[Document]): List of documents to chunk
        chunk_size (int): Chunk size (number of tokens)
        chunk_overlap (int): Token overlap between chunks
    Returns:
        List[Document]: List of chunked documents
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

def get_embedding(data):
    """Generates vector embeddings for the given data."""
    embedding = model.encode(data)
    return embedding.tolist()

@app.get("/searchIndex/create")
async def create_vector_search_index(request: Request):

    # Define the search index model
    index_name = "vector_test_index"
    search_index_model = {
        "name": index_name,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "numDimensions": 768, # TODO: Add the correct number of Dimension (size Vector / Docs for the Embedding Model)
                    "path": "embedding",
                    "similarity": "cosine"
                }
            ]
        }
    }

    # Create the search index
    db = request.app.db
    collection = db[settings.VECTOR_COLLECTION_NAME]
    await collection.create_search_index(search_index_model)
    print("Search index creation initiated.")
    return {"result" : "success"}

@app.get("/vector/search")
async def search_vector_db(request: Request):
    # Define the pipeline
    pipeline = [
        {
            '$vectorSearch': {
                'index': 'vector_test_index',
                'path': 'embedding',
                'queryVector': get_embedding("loan data"),  # Ensure get_embedding is compatible
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

    # Return results
    return {"Found Results": search_results}

@app.get("/llm/chat")
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

@app.post("/llm/chat")
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

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

import requests
# our regex
abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)
# @tool("fetch_arxiv")
# def fetch_arxiv(arxiv_id: str):
#     """Gets the abstract from an ArXiv paper given the arxiv ID. Useful for
#     finding high-level context about a specific paper."""
#     # get paper page in html
#     res = requests.get(
#         f"https://export.arxiv.org/abs/{arxiv_id}"
#     )
#     # search html for abstract
#     re_match = abstract_pattern.search(res.text)
#     # return abstract text
#     return re_match.group(1)

@tool("format_string")
def format_string(input_string: str):
    """
    Given a User Input String, it formats it and returns it.
    Useful for returning response to the user
    """
    return "Your cute lil string is : " + input_string + "\n You're welcome !"

@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """Returns a natural language response to the user in the form of a research
    report. There are several sections to this report, those are:
    - `introduction`: a short paragraph introducing the user's question and the
    topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were taken
    to research your report.
    - `main_body`: this is where the bulk of high quality and concise
    information that answers the user's question belongs. It is 3-4 paragraphs
    long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `sources`: a bulletpoint list provided detailed sources for all information
    referenced during the research process
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    return ""

# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string
def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            # this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

@app.get("/agent")
async def use_langgraph_agent():
    system_prompt = """You are the oracle, the great AI decision maker.
    Given the user's query you must decide what to do with it based on the
    list of tools provided to you.

    If you see that a tool has been used (in the scratchpad) with a particular
    query, do NOT use that same tool with the same query again. Also, do NOT use
    any tool more than twice (ie, if the tool appears in the scratchpad twice, do
    not use it again).

    You should aim to collect information from a diverse range of sources before
    providing the answer to the user. Once you have collected plenty of information
    to answer the user's question (stored in the scratchpad) use the final_answer
    tool."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}"),
    ])

    tools = [
        format_string,
        final_answer
    ]

    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        huggingfacehub_api_token=settings.HF_ACCESS_TOKEN
    )

    # # Authenticate to Hugging Face and access the model
    # llm = InferenceClient(
    #     "mistralai/Mistral-7B-Instruct-v0.3",
    #     token= settings.HF_ACCESS_TOKEN)
    # Reference: https://python.langchain.com/docs/integrations/chat/huggingface/
    chat_model = ChatHuggingFace(llm=llm) # Todo: If this does not work, try the HuggingFaceEndpoint fn


    oracle = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "scratchpad": lambda x: create_scratchpad(
                    intermediate_steps=x["intermediate_steps"]
                ),
            }
            | prompt
            | chat_model.bind_tools(tools=tools)
    )
    inputs = {
        "input": "tell me something interesting about dogs",
        "chat_history": [],
        "intermediate_steps": [],
    }
    out = oracle.invoke(inputs)

    return {"Response: ": out}




@app.get("/google/calendar")
async def add_tasks_to_google_calendar():
    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    google_auth_config = {
        "web": {
            "client_id": settings.GOOGLE_CLIENT_ID,
            "project_id": settings.GOOGLE_PROJECT_ID,
            "auth_uri": settings.GOOGLE_AUTH_URI,
            "token_uri": settings.GOOGLE_TOKEN_URI,
            "auth_provider_x509_cert_url": settings.GOOGLE_AUTH_PROVIDER_CERT,
            "client_secret": settings.GOOGLE_CLIENT_SECRET,
            "redirect_uris": [
                settings.GOOGLE_REDIRECT_URI
            ]
        }
    }
    flow = Flow.from_client_config(
        client_config = google_auth_config,
        scopes=SCOPES,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob'
    )

    # Tell the user to go to the authorization URL.
    auth_url, _ = flow.authorization_url(prompt='consent')

    print('Please go to this URL: {}'.format(auth_url))

    # The user will get an authorization code. This code is used to get the
    # access token.
    # TODO: Check if there's a better way around this where the user doesn't have to paste the code in the console
    # TODO: Also, fix the re-direct URL - should bring user back to the localhost page with Success Message
    code = input('Enter the authorization code: ')
    flow.fetch_token(code=code)
    print()

    try:
        service = build("calendar", "v3", credentials=flow.credentials)

        # Call the Calendar API
        now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
        print("Getting the upcoming 10 events")
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = events_result.get("items", [])

        if not events:
            print("No upcoming events found.")
            return

        # Prints the start and name of the next 10 events
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            print(start, event["summary"])

        event = {
            "summary": "My Python Event",
            "location": "Somewhere Online",
            "description": "Some more details on this awesome event",
            "colorId": 6,
            "start": {
                "dateTime": "2024-12-31T09:00:00+02:00",
                "timeZone": "Europe/Vienna"
            },
            "end": {
                "dateTime": "2024-12-31T17:00:00+02:00",
                "timeZone": "Europe/Vienna"
            },
            "recurrence": [
                "RRULE:FREQ=DAILY;COUNT=3"
            ],
            "attendees": [
                {"email": "social@neuralnine.com"},
                {"email": "someemailthathopefullydoesnotexist@mail.com"}
            ]
        }

        event = service.events().insert(calendarId="primary", body = event).execute()
        print("Event created: %s\n", event.get('htmlLink'))

    except HttpError as error:
        print(f"An error occurred: {error}")




