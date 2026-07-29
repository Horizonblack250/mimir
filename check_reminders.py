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
        print(f"DEBUG: task='{t['task']}' due={due_time} now={now} minutes_until={minutes_until:.1f} "
              f"notified_upcoming={t.get('notified_upcoming')} notified_overdue={t.get('notified_overdue')}")

        if 0 <= minutes_until <= 15 and not t.get("notified_upcoming"):
            print("DEBUG: firing UPCOMING notification")
            notification.notify(
                title="Mimir — Upcoming Task",
                message=f"Due in {int(minutes_until)} min: {t['task']}",
                app_name="Mimir",
                timeout=15
            )
            t["notified_upcoming"] = True
            changed = True

        elif minutes_until < 0 and not t.get("notified_overdue"):
            print("DEBUG: firing OVERDUE notification")
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
        print("DEBUG: tasks.json updated")
    else:
        print("DEBUG: no changes made")


if __name__ == "__main__":
    main()