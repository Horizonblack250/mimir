import json
import os
import datetime

USAGE_FILE = os.path.join(os.path.dirname(__file__), "..", "usage_stats.json")


def _load_usage():
    if not os.path.exists(USAGE_FILE):
        return {}
    with open(USAGE_FILE, "r") as f:
        content = f.read().strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _save_usage(data):
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def record_usage(model_name, prompt_tokens, completion_tokens):
    """Records real token usage from an actual API response. This is the only
    source of truth for usage numbers -- never let the model guess or
    estimate these, always report from what's actually been recorded here."""
    today = datetime.date.today().isoformat()
    data = _load_usage()

    if today not in data:
        data[today] = {"total_tokens": 0, "total_requests": 0, "by_model": {}}

    total = (prompt_tokens or 0) + (completion_tokens or 0)
    data[today]["total_tokens"] += total
    data[today]["total_requests"] += 1

    if model_name not in data[today]["by_model"]:
        data[today]["by_model"][model_name] = {"requests": 0, "tokens": 0}
    data[today]["by_model"][model_name]["requests"] += 1
    data[today]["by_model"][model_name]["tokens"] += total

    _save_usage(data)


def get_usage_summary(scope="today"):
    """Returns a grounded, real summary of usage -- never estimated."""
    data = _load_usage()
    today = datetime.date.today().isoformat()

    if scope == "today":
        day_data = data.get(today)
        if not day_data:
            return "No usage recorded yet today."
        lines = [f"Today's usage: {day_data['total_requests']} requests, {day_data['total_tokens']} total tokens."]
        for model, stats in day_data["by_model"].items():
            lines.append(f"  - {model}: {stats['requests']} requests, {stats['tokens']} tokens")
        return "\n".join(lines)

    # all-time
    if not data:
        return "No usage recorded yet."
    total_tokens = sum(day["total_tokens"] for day in data.values())
    total_requests = sum(day["total_requests"] for day in data.values())
    days_tracked = len(data)
    return (
        f"All-time usage across {days_tracked} day(s) of tracking: "
        f"{total_requests} requests, {total_tokens} total tokens."
    )