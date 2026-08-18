import os
import datetime
from google.cloud import firestore
from google.oauth2 import service_account

CRED_PATH = os.path.join(os.path.dirname(__file__), "..", "firestore-credentials.json")

_db = None


def _get_db():
    global _db
    if _db is None:
        creds = service_account.Credentials.from_service_account_file(CRED_PATH)
        _db = firestore.Client(credentials=creds, project=creds.project_id)
    return _db


def sync_task(task_id, task_text, due_iso, reminder_offset_minutes, done=False):
    """Pushes a lightweight reminder record to Firestore, so the cloud
    reminder-checker can see it even when this PC is off. Fails silently --
    a sync failure should never break local task creation."""
    if not due_iso:
        return  # nothing to remind about, nothing to sync
    try:
        db = _get_db()
        db.collection("mimir_reminders").document(task_id).set({
            "task": task_text,
            "due": due_iso,
            "reminder_offset_minutes": reminder_offset_minutes or 15,
            "done": done,
            "notified": False,
            "synced_at": datetime.datetime.now().isoformat()
        })
    except Exception:
        pass


def sync_done(task_id):
    """Marks a task done in Firestore so the cloud checker stops reminding
    about it -- also fails silently."""
    try:
        db = _get_db()
        db.collection("mimir_reminders").document(task_id).update({"done": True})
    except Exception:
        pass