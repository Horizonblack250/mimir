import json
import os
import datetime

TASKS_FILE = os.path.join(os.path.dirname(__file__), "..", "tasks.json")


def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r") as f:
        return json.load(f)


def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)


def _normalize(text):
    """Lowercase and strip punctuation/whitespace so similar phrasings match."""
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).strip()


def add_task(description, due=None):
    tasks = load_tasks()
    new_norm = _normalize(description)

    for t in tasks:
        if _normalize(t["task"]) == new_norm and not t["done"]:
            due_str = f" (due: {t['due']})" if t.get("due") else ""
            return f"That task already exists and is still pending: {t['task']}{due_str}. Not adding a duplicate."

    tasks.append({"task": description, "done": False, "due": due})
    save_tasks(tasks)
    if due:
        return f"Added task: {description} (due: {due})"
    return f"Added task: {description} (no due date)"


def list_tasks(filter_mode="all"):
    tasks = load_tasks()
    today = datetime.date.today().isoformat()

    numbered = list(enumerate(tasks, start=1))

    if filter_mode == "today":
        numbered = [(i, t) for i, t in numbered if t.get("due") == today]
    elif filter_mode == "overdue":
        numbered = [(i, t) for i, t in numbered if t.get("due") and t["due"] < today and not t["done"]]

    if not numbered:
        if filter_mode == "today":
            return "You have no tasks due today."
        elif filter_mode == "overdue":
            return "You have no overdue tasks."
        return "You have no tasks."

    lines = []
    for i, t in numbered:
        status = "DONE" if t["done"] else "PENDING"
        due_str = f" | due: {t['due']}" if t.get("due") else ""
        lines.append(f"{i}. [{status}] {t['task']}{due_str}")
    return "\n".join(lines)


def complete_task(index):
    tasks = load_tasks()
    if index < 1 or index > len(tasks):
        return "That task number doesn't exist."
    tasks[index - 1]["done"] = True
    save_tasks(tasks)
    return f"Marked task {index} as done."