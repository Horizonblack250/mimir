import datetime
from plyer import notification
from skills import todo

def get_due_and_overdue():
    tasks = todo.load_tasks()
    today = datetime.date.today().isoformat()
    relevant = [
        t for t in tasks
        if not t["done"] and t.get("due") and t["due"] <= today
    ]
    return relevant

def main():
    tasks = get_due_and_overdue()

    if not tasks:
        return  # nothing to say, stay silent, don't spam a notification for no reason

    lines = []
    for t in tasks:
        marker = "OVERDUE" if t["due"] < datetime.date.today().isoformat() else "DUE TODAY"
        lines.append(f"[{marker}] {t['task']}")

    message = "\n".join(lines)

    notification.notify(
        title="Mimir — Task Check-in",
        message=message,
        app_name="Mimir",
        timeout=15
    )

if __name__ == "__main__":
    main()