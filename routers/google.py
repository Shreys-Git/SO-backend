import datetime

from fastapi import APIRouter
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import Flow

from config import BaseConfig

google_router = APIRouter()
settings = BaseConfig()

@google_router.get("/google/calendar")
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