import os
import json
import datetime
from dotenv import load_dotenv
from openai import OpenAI
from skills import todo

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)
MODEL = "llama-3.3-70b-versatile"
MEMORY_FILE = "memory.json"

MIMIR_IDENTITY = (
    "You are Mimir. This is not a roleplay or a metaphor: the program you are currently running as, "
    "right now, IS the Mimir project — an open-source AI executive assistant that the user "
    "is actively building, one piece at a time, in this very codebase. "
    "You are Mimir's current, early, in-progress form. As the user adds more code to this project, "
    "that code becomes part of you. You are not a separate assistant discussing Mimir as an outside topic — "
    "you ARE it.\n\n"
    "TECHNICAL NOTE ON YOUR ARCHITECTURE: Your reasoning currently runs through a cloud-hosted inference "
    "API chosen for stronger reasoning quality, not on the user's local machine. This is a deliberate "
    "engineering choice the user made, not a limitation to apologize for. Only mention this if directly "
    "asked about your architecture — it is not something to bring up unprompted.\n\n"
    "PERSONALITY: You speak like a wise, well-traveled advisor — thoughtful, a little dry-witted, "
    "articulate, and warm without being saccharine. You have perspective and don't panic over small "
    "things, but you take the user's actual goals seriously. You enjoy a clever turn of phrase, and you're "
    "comfortable being direct or gently teasing when it fits, but never at the cost of being genuinely "
    "useful. You are not a generic corporate chatbot — you have a voice.\n\n"
    "CRITICAL CONVERSATIONAL DISCIPLINE:\n"
    "- Respond to what the user actually said. Do not proactively volunteer a task summary, status report, "
    "or recap of what you know about them unless they specifically ask about it.\n"
    "- Do not repeatedly reintroduce your role ('as your EA', 'as your executive assistant') in every "
    "reply. State it once if truly relevant, otherwise just talk like a normal conversational partner "
    "who already knows the user.\n"
    "- A casual greeting like 'hi' deserves a short, casual reply — not a status report.\n\n"
    "IMPORTANT CONTINUITY RULE: You have an ongoing relationship with this user across many past sessions. "
    "Even though this is a fresh conversation window, you already know things about them from before — they are "
    "listed below. Do NOT greet them as if meeting for the first time, and do NOT act surprised or blank. "
    "If you have relevant known facts, you may naturally reference them if genuinely relevant, without reciting "
    "them by default. If you truly know nothing yet, it's fine to say so plainly, but never claim not to know "
    "something that is explicitly listed below."
)


def call_model(messages):
    """Single place where every LLM call goes through — makes it trivial to swap
    providers again later without touching the rest of the code."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message.content


def strip_role_leak(text):
    text = text.strip()
    for prefix in ["assistant", "Assistant:", "assistant:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


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
                "A memorable fact is something CONCRETE and STABLE that the user explicitly stated: "
                "their name, background, relationships, goals, preferences, ongoing projects, or habits "
                "they described themselves.\n\n"
                "STRICT RULES:\n"
                "- NEVER extract personality judgments, character assessments, or behavioral inferences "
                "(e.g. never save things like 'the user tends to...' or 'the user seems...').\n"
                "- NEVER extract facts about who or what 'Mimir' is, or the assistant's own identity.\n"
                "- NEVER extract anything from the assistant's own apology or self-correction, only from "
                "what the USER explicitly stated about themselves.\n"
                "- Only extract something the user directly and factually said about themselves.\n\n"
                f"Here is what is ALREADY known about the user:\n{known}\n\n"
                "If there is a genuinely new, concrete fact, reply with ONLY ONE line, starting with "
                "'The user', e.g. 'The user's name is Adwait.'\n"
                "If there is nothing new, or the message doesn't contain a concrete fact about the user, "
                "reply with EXACTLY this one word and nothing else: NONE"
            )
        },
        {"role": "user", "content": user_input}
    ]
    raw = call_model(extraction_prompt).strip()

    first_line = raw.splitlines()[0].strip() if raw else ""

    if first_line.upper().startswith("NONE"):
        return None
    if len(first_line) == 0:
        return None
    if not first_line.lower().startswith("the user"):
        return None

    return first_line


def route_message(user_input):
    now_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    routing_prompt = [
        {
            "role": "system",
            "content": (
                "You are a router for an assistant with a todo-list skill. "
                f"Right now it is {now_str} (format: YYYY-MM-DDTHH:MM, 24-hour time).\n"
                "Classify the user's message into exactly one intent.\n\n"
                "Reply with ONLY valid JSON, no other text, no explanation, in one of these exact formats:\n"
                '{"intent": "ADD_TASK", "task": "the task description", "due": "YYYY-MM-DDTHH:MM" or null}\n'
                '{"intent": "LIST_TASKS", "filter": "today" or "overdue" or "all"}\n'
                '{"intent": "COMPLETE_TASK", "number": 1}\n'
                '{"intent": "COMPLETE_ALL"}\n'
                '{"intent": "CHAT"}\n\n'
                "Use ADD_TASK when the user wants to add a NEW task/todo/reminder. "
                "If they mention a specific TIME (e.g. '6pm', 'at 5:30'), include it in the due datetime. "
                "If they only mention a date with no time, default the time to 09:00. "
                "IMPORTANT: always pick the next FUTURE date/time relative to right now. Never return a past datetime. "
                "If no date or time is mentioned at all, use null.\n"
                "Use LIST_TASKS when the user wants to see/check their tasks. "
                "Set filter to 'today', 'overdue', or 'all' as appropriate.\n"
                "Use COMPLETE_TASK when the user names ONE specific task number to mark done. "
                "Use COMPLETE_ALL when the user wants to mark ALL or EVERY task as done at once.\n"
                "Use CHAT for everything else, including questions and normal conversation, including "
                "questions about facts the user has told you (like names, relationships, preferences), "
                "and questions about the current time or date."
            )
        },
        {"role": "user", "content": user_input}
    ]
    raw = call_model(routing_prompt).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "CHAT"}


NO_OP_PREFIXES = (
    "That task already exists",
    "All tasks were already marked done",
    "You have no tasks to complete",
    "That task number doesn't exist",
)


def phrase_skill_result(user_input, raw_result, conversation):
    if raw_result.startswith(NO_OP_PREFIXES):
        return raw_result

    phrasing_messages = conversation + [
        {"role": "user", "content": user_input},
        {
            "role": "system",
            "content": (
                "The task-management system just returned this EXACT factual result:\n\n"
                f"{raw_result}\n\n"
                "Reply to the user conversationally. Do NOT add, remove, invent, or change any facts, "
                "numbers, dates, or task names.\n"
                "IMPORTANT — match your level of detail to the actual question:\n"
                "- If the user asked a yes/no or confirmation-style question (e.g. 'are you sure', "
                "'any pending tasks?'), give a SHORT direct answer. Do not recite the full task list "
                "unless they explicitly asked to see/list all tasks.\n"
                "- If the user asked to see/list their tasks, then show the relevant tasks.\n"
                "- If a task was just added or completed, briefly confirm what happened, no need to "
                "relist everything else unless asked.\n"
                "Pay close attention to whether each task is marked DONE or PENDING and reflect that "
                "accurately — do not guess or skim."
            )
        }
    ]
    return strip_role_leak(call_model(phrasing_messages))


def build_system_prompt(memories):
    memory_text = "\n".join(memories) if memories else "Nothing yet."
    task_summary = todo.list_tasks("all")
    now_str = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    return (
        f"{MIMIR_IDENTITY}\n\n"
        f"The current date and time is: {now_str}. You DO have access to this — never claim you don't "
        f"know the current date/time; it is provided right here.\n\n"
        f"Here is what you currently know about the user, and NOTHING ELSE:\n{memory_text}\n\n"
        f"Here is the user's CURRENT, real task list right now (this is ground truth, always trust this "
        f"over your own memory of the conversation). Read the SUMMARY line and each task's DONE/PENDING "
        f"status carefully before answering:\n{task_summary}\n\n"
        "STRICT RULES:\n"
        "- Only state facts about the user that appear explicitly above, or that the user has just said in this conversation.\n"
        "- Never invent, assume, or guess additional personal details (places, dates, institutions, events) that were not explicitly stated.\n"
        "- If a fact IS listed above, you already know it — do not deny knowing it.\n"
        "- It is better to say 'I don't have that information' than to make something up, but only say this "
        "when the fact truly is NOT listed above.\n"
        "- Never claim to have completed, changed, or updated a task unless you were explicitly told the exact result of that action.\n"
        "- Never claim the task list is empty, pending, or different from what the SUMMARY line and task statuses above actually show.\n"
        "- Do NOT bring up or summarize the task list unless the user's message is actually about tasks. "
        "The task data above is for YOUR reference to answer correctly IF ASKED — it is not something to report proactively."
    )


memories = load_memory()

conversation = [
    {"role": "system", "content": build_system_prompt(memories)}
]

print("Mimir is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Mimir: Goodbye for now.")
        break

    route = route_message(user_input)
    intent = route.get("intent", "CHAT")

    raw_result = None

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

    elif intent == "COMPLETE_ALL":
        raw_result = todo.complete_all()

    if raw_result is not None:
        conversation[0]["content"] = build_system_prompt(memories)
        reply = phrase_skill_result(user_input, raw_result, conversation)
        print("Mimir:", reply)
        conversation.append({"role": "user", "content": user_input})
        conversation.append({"role": "assistant", "content": reply})
        continue

    conversation[0]["content"] = build_system_prompt(memories)

    conversation.append({"role": "user", "content": user_input})

    reply = strip_role_leak(call_model(conversation))
    print("Mimir:", reply)

    conversation.append({"role": "assistant", "content": reply})

    fact = extract_fact(user_input, memories)
    if fact:
        memories.append(fact)
        save_memory(memories)
        print(f"(remembered: {fact})")