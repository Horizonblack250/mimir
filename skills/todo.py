import json
import os
import datetime
import uuid

TASKS_FILE = os.path.join(os.path.dirname(__file__), "..", "tasks.json")


def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r") as f:
        tasks = json.load(f)

    migrated = False
    for t in tasks:
        if "id" not in t:
            t["id"] = uuid.uuid4().hex[:8]
            migrated = True
        if "reminder_offset_minutes" not in t:
            t["reminder_offset_minutes"] = None
            migrated = True

    if migrated:
        save_tasks(tasks)

    return tasks


def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)


def _normalize(text):
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


def _parse(due_str):
    try:
        return datetime.datetime.fromisoformat(due_str)
    except ValueError:
        return datetime.datetime.fromisoformat(due_str + "T23:59")


def add_task(description, due=None):
    tasks = load_tasks()
    new_norm = _normalize(description)

    for t in tasks:
        if _normalize(t["task"]) == new_norm and not t["done"]:
            due_str = f" (due: {t['due']})" if t.get("due") else ""
            return f"That task already exists and is still pending: {t['task']}{due_str}. Not adding a duplicate.", t["id"]

    task_id = uuid.uuid4().hex[:8]
    tasks.append({
        "id": task_id,
        "task": description,
        "done": False,
        "due": due,
        "reminder_offset_minutes": None,
        "notified_upcoming": False,
        "notified_overdue": False
    })
    save_tasks(tasks)
    if due:
        return f"Added task: {description} (due: {due})", task_id
    return f"Added task: {description} (no due date/time)", task_id


def get_task_by_id(task_id):
    tasks = load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            return t
    return None


def update_task(task_id, due=None, reminder_offset_minutes=None, new_description=None):
    """Modifies an existing task in place -- this is what Conversation Focus
    resolves into, instead of accidentally creating a duplicate task."""
    tasks = load_tasks()
    target = None
    for t in tasks:
        if t["id"] == task_id:
            target = t
            break

    if target is None:
        return "I couldn't find that task anymore -- it may have been completed or removed."

    changes = []
    if due is not None:
        target["due"] = due
        target["notified_upcoming"] = False
        target["notified_overdue"] = False
        changes.append(f"due date to {due}")
    if reminder_offset_minutes is not None:
        target["reminder_offset_minutes"] = reminder_offset_minutes
        target["notified_upcoming"] = False
        changes.append(f"reminder to {reminder_offset_minutes} minutes before")
    if new_description is not None:
        target["task"] = new_description
        changes.append(f"description to '{new_description}'")

    if not changes:
        return "Nothing to update -- no changes were specified."

    save_tasks(tasks)
    return f"Updated '{target['task']}': " + ", ".join(changes) + "."


def delete_task(task_id):
    tasks = load_tasks()
    target = None
    for t in tasks:
        if t["id"] == task_id:
            target = t
            break

    if target is None:
        return "I couldn't find that task anymore -- it may already be gone."

    tasks = [t for t in tasks if t["id"] != task_id]
    save_tasks(tasks)
    return f"Removed task: {target['task']}."


def list_tasks(filter_mode="all"):
    tasks = load_tasks()
    now = datetime.datetime.now()
    today = now.date().isoformat()

    numbered = list(enumerate(tasks, start=1))

    if filter_mode == "today":
        numbered = [(i, t) for i, t in numbered if t.get("due") and t["due"].startswith(today)]
    elif filter_mode == "overdue":
        numbered = [
            (i, t) for i, t in numbered
            if t.get("due") and not t["done"] and _parse(t["due"]) < now
        ]

    pending_count = sum(1 for _, t in numbered if not t["done"])
    done_count = sum(1 for _, t in numbered if t["done"])

    if not numbered:
        if filter_mode == "today":
            return "SUMMARY: 0 pending, 0 done, 0 tasks due today.\nYou have no tasks due today."
        elif filter_mode == "overdue":
            return "SUMMARY: 0 overdue tasks.\nYou have no overdue tasks."
        return "SUMMARY: 0 pending, 0 done. You have no tasks at all."

    lines = [f"SUMMARY: {pending_count} pending, {done_count} done (out of {len(numbered)} shown below)."]
    for i, t in numbered:
        status = "DONE" if t["done"] else "PENDING"
        due_str = f" | due: {t['due']}" if t.get("due") else ""
        reminder_str = f" | reminds {t['reminder_offset_minutes']}min before" if t.get("reminder_offset_minutes") else ""
        lines.append(f"{i}. [{status}] {t['task']}{due_str}{reminder_str}  (id: {t['id']})")
    return "\n".join(lines)


def complete_task(index):
    tasks = load_tasks()
    if index < 1 or index > len(tasks):
        return "That task number doesn't exist.", None
    tasks[index - 1]["done"] = True
    save_tasks(tasks)
    return f"Marked task {index} as done.", tasks[index - 1]["id"]


def complete_all():
    tasks = load_tasks()
    if not tasks:
        return "You have no tasks to complete."
    count = sum(1 for t in tasks if not t["done"])
    for t in tasks:
        t["done"] = True
    save_tasks(tasks)
    if count == 0:
        return "All tasks were already marked done. Nothing changed."
    return f"Marked {count} task(s) as done. All tasks are now complete."