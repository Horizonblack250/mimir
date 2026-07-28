import os
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Read-only, on purpose. Mimir should never be able to send, delete, or modify email.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

BASE_DIR = os.path.dirname(__file__)
CREDENTIALS_FILE = os.path.join(BASE_DIR, "..", "credentials.json")
TOKEN_FILE = os.path.join(BASE_DIR, "..", "token.json")


def _get_service():
    """Handles login/auth. First run opens a browser for you to approve access.
    After that, it reuses a saved token so you don't have to log in every time."""
    creds = None

    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_unread_summary(max_results=5):
    """Fetches basic info (sender, subject, snippet) for the most recent unread
    emails. Read-only -- never marks as read, never sends, never deletes."""
    try:
        service = _get_service()
        results = service.users().messages().list(
            userId="me", labelIds=["UNREAD", "INBOX"], maxResults=max_results
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return "SUMMARY: 0 unread emails.\nYou have no unread emails."

        lines = [f"SUMMARY: {len(messages)} unread email(s) shown below."]
        for i, msg_ref in enumerate(messages, start=1):
            msg = service.users().messages().get(
                userId="me", id=msg_ref["id"], format="metadata",
                metadataHeaders=["From", "Subject"]
            ).execute()

            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            sender = headers.get("From", "Unknown sender")
            subject = headers.get("Subject", "(no subject)")
            snippet = msg.get("snippet", "")

            lines.append(f"{i}. From: {sender} | Subject: {subject} | Preview: {snippet}")

        return "\n".join(lines)

    except FileNotFoundError:
        return "Gmail isn't connected yet -- credentials.json is missing."
    except Exception as e:
        return f"Couldn't reach Gmail right now ({type(e).__name__}). Try again in a moment."