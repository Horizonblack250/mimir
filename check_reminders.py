import datetime
from plyer import notification
from skills import todo


def main():
    tasks = todo.load_tasks()
    now = datetime.datetime.now()
    changed = False

    for t in tasks:
        if t["done"] or not t.get("due"):
            continue

        due_time = todo._parse(t["due"])
        minutes_until = (due_time - now).total_seconds() / 60

        # Use the task's own configured reminder offset (e.g. "remind me an
        # hour before" -> 60), falling back to a default 15 min if the user
        # never specified one for this particular task.
        reminder_window = t.get("reminder_offset_minutes") or 15

        if 0 <= minutes_until <= reminder_window and not t.get("notified_upcoming"):
            notification.notify(
                title="Mimir — Upcoming Task",
                message=f"Due in {int(minutes_until)} min: {t['task']}",
                app_name="Mimir",
                timeout=15
            )
            t["notified_upcoming"] = True
            changed = True

        elif minutes_until < 0 and not t.get("notified_overdue"):
            notification.notify(
                title="Mimir — Overdue Task",
                message=f"Overdue: {t['task']}",
                app_name="Mimir",
                timeout=15
            )
            t["notified_overdue"] = True
            changed = True

    if changed:
        todo.save_tasks(tasks)


if __name__ == "__main__":
    main()