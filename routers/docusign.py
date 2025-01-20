import base64
from collections import defaultdict
from enum import Enum

import httpx
from docusign_esign import ApiClient, TemplateRole, EnvelopeDefinition, EnvelopesApi
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import RedirectResponse
from fastapi import Request
from config import BaseConfig


docusign_router = APIRouter()
settings = BaseConfig()

tokens = {}
agreements_cache = {}
agreements_stats = defaultdict(int)

class APIScope(Enum):
    ESIGNATURE = "signature"
    NAVIGATOR = "adm_store_unified_repo_read"

'''
Re-directs User to the correct Docusign Sign In endpoint
Note: Scope changes based on whether it's e-sign or navigator redirect
'''
@docusign_router.get("/login")
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
@docusign_router.get("/callback")
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
    return RedirectResponse(url=f"http://localhost:5173")


class FormData(BaseModel):
    name: str
    email: str

''' Use the E-signature SDK to send the envelope.
This will send the Document to the user via an email & returns the ID 
as soon as the email is sent (doesn't wait for the sign to complete)
N.B: The SDK does have functionality to track the status of the sign'''
@docusign_router.post("/submit")
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
@docusign_router.get("/navigator/login")
async def get_nav_access_token():
    # TODO: Combine the two endpoints later & send a query param to tell which access token is needed
    return RedirectResponse(url=build_redirect_url(False))


@docusign_router.get("/navigator/agreements/stats")
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
@docusign_router.get("/navigator/agreements/process")
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


@docusign_router.get("/navigator/agreements/{agreement_id}")
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

    return nav_agreements
