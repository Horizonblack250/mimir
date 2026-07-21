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

def add_task(description, due=None):
    tasks = load_tasks()
    tasks.append({"task": description, "done": False, "due": due})
    save_tasks(tasks)
    if due:
        return f"Added task: {description} (due: {due})"
    return f"Added task: {description} (no due date)"

def list_tasks(filter_mode="all"):
    tasks = load_tasks()
    today = datetime.date.today().isoformat()

    if filter_mode == "today":
        tasks = [t for i, t in enumerate(tasks) if t.get("due") == today]
    elif filter_mode == "overdue":
        tasks = [t for t in tasks if t.get("due") and t["due"] < today and not t["done"]]

    if not tasks:
        if filter_mode == "today":
            return "You have no tasks due today."
        elif filter_mode == "overdue":
            return "You have no overdue tasks."
        return "You have no tasks."

    lines = []
    for i, t in enumerate(tasks, start=1):
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