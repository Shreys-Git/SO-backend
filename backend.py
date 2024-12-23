from contextlib import asynccontextmanager
from enum import Enum

import httpx
import base64

from docusign_esign import ApiClient, EnvelopesApi, EnvelopeDefinition, TemplateRole
from fastapi import FastAPI, Request
from motor import motor_asyncio
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from config import BaseConfig

settings = BaseConfig()

tokens = {}

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

@app.get("/navigator/agreements")
async def get_nav_agreements():
    # TODO: Make a post request to the Nav Endpoint

    GET_ALL_NAV_AGREEMENTS_URL = f"https://api-d.docusign.com/v1/accounts/{settings.API_ACCOUNT_ID}/agreements"
    async with httpx.AsyncClient() as client:
        response = await client.get(
            GET_ALL_NAV_AGREEMENTS_URL,
            headers={
                "Authorization": f"Bearer {tokens["access_token"]}",
                "Accept": "application/json"
            },
        )
        agreements = response.json()

    # Extract the data, which a list of the documents
    nav_agreements = agreements.get("data")
    print(nav_agreements)
    return nav_agreements


