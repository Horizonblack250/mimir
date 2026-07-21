import ollama
import json
import datetime
from skills import todo

MODEL = "llama3.1:8b"
MEMORY_FILE = "memory.json"
TODAY = datetime.date.today().isoformat()

def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)

def extract_fact(user_input, existing_memories):
    known = "\n".join(existing_memories) if existing_memories else "Nothing yet."
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "You extract long-term memorable facts about a user from a single message.\n"
                "A memorable fact is something stable: their name, background, goals, preferences, "
                "ongoing projects, or habits.\n\n"
                f"Here is what is ALREADY known about the user:\n{known}\n\n"
                "If the new message contains a fact that is NOT already covered above, reply with ONLY "
                "a short, clean sentence stating it (e.g. 'The user's name is Adwait.').\n"
                "If the message contains nothing new worth remembering, or repeats something already known, "
                "reply with exactly: NONE"
            )
        },
        {"role": "user", "content": user_input}
    ]
    result = ollama.chat(model=MODEL, messages=extraction_prompt)
    fact = result["message"]["content"].strip()
    if fact.upper() == "NONE" or len(fact) == 0:
        return None
    return fact

def route_message(user_input):
    routing_prompt = [
        {
            "role": "system",
            "content": (
                "You are a router for an assistant with a todo-list skill. "
                f"Today's date is {TODAY} (format: YYYY-MM-DD).\n"
                "Classify the user's message into exactly one intent.\n\n"
                "Reply with ONLY valid JSON, no other text, no explanation, in one of these exact formats:\n"
                '{"intent": "ADD_TASK", "task": "the task description", "due": "YYYY-MM-DD" or null}\n'
                '{"intent": "LIST_TASKS", "filter": "today" or "overdue" or "all"}\n'
                '{"intent": "COMPLETE_TASK", "number": 1}\n'
                '{"intent": "CHAT"}\n\n'
                "Use ADD_TASK when the user wants to add a NEW task/todo/reminder. "
                "When resolving a date, IMPORTANT: always pick the next FUTURE occurrence relative to today. "
                "If a day-of-month has already passed this month, use next month instead. Never return a past date. "
                "If no date is mentioned, use null.\n"
                "Use LIST_TASKS when the user wants to see/check their tasks (e.g. 'what are my tasks', "
                "'do I have anything today', 'show my tasks', 'I meant tasks for today'). "
                "Set filter to 'today' if they're asking about today specifically, 'overdue' if asking about "
                "late/missed tasks, otherwise 'all'.\n"
                "Use COMPLETE_TASK when they want to mark a numbered task done. "
                "Use CHAT for everything else, including questions and normal conversation."
            )
        },
        {"role": "user", "content": user_input}
    ]
    result = ollama.chat(model=MODEL, messages=routing_prompt)
    raw = result["message"]["content"].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "CHAT"}

def phrase_skill_result(user_input, raw_result, conversation):
    """Takes a raw, factual skill result and asks the model to phrase it naturally,
    without changing or adding any facts."""
    phrasing_messages = conversation + [
        {"role": "user", "content": user_input},
        {
            "role": "system",
            "content": (
                "The task-management system just returned this EXACT factual result:\n\n"
                f"{raw_result}\n\n"
                "Reply to the user conversationally, presenting this information naturally. "
                "Do NOT add, remove, invent, or change any facts, numbers, dates, or task names. "
                "Only rephrase tone and delivery, not content."
            )
        }
    ]
    result = ollama.chat(model=MODEL, messages=phrasing_messages)
    return result["message"]["content"]

memories = load_memory()
memory_text = "\n".join(memories) if memories else "Nothing yet."

conversation = [
    {
        "role": "system",
        "content": (
            "You are Mimir, the user's personal local AI assistant.\n\n"
            f"Here is what you currently know about the user, and NOTHING ELSE:\n{memory_text}\n\n"
            "STRICT RULES:\n"
            "- Only state facts about the user that appear explicitly above, or that the user has just said in this conversation.\n"
            "- Never invent, assume, or guess additional personal details (places, dates, institutions, events) that were not explicitly stated.\n"
            "- If you don't know something about the user, say so plainly instead of guessing.\n"
            "- It is better to say 'I don't have that information' than to make something up."
        )
    }
]

print("Mimir is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Mimir: Goodbye for now.")
        break

    route = route_message(user_input)
    intent = route.get("intent", "CHAT")

    if intent == "ADD_TASK":
        task_desc = route.get("task", user_input)
        due_date = route.get("due")
        raw_result = todo.add_task(task_desc, due_date)

    elif intent == "LIST_TASKS":
        filter_mode = route.get("filter", "all")
        raw_result = todo.list_tasks(filter_mode)

    elif intent == "COMPLETE_TASK":
        number = route.get("number", 0)
        raw_result = todo.complete_task(number)

    else:
        raw_result = None

    if raw_result is not None:
        reply = phrase_skill_result(user_input, raw_result, conversation)
        print("Mimir:", reply)
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": reply})
        continue

    conversation.append({"role": "user", "content": user_input})

    response = ollama.chat(
        model=MODEL,
        messages=conversation
    )

    reply = response["message"]["content"]
    print("Mimir:", reply)

    conversation.append({"role": "assistant", "content": reply})

    fact = extract_fact(user_input, memories)
    if fact:
        memories.append(fact)
        save_memory(memories)
        print(f"(remembered: {fact})")