import ollama
import json

MEMORY_FILE = "memory.json"

def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)

def extract_fact(user_input, existing_memories):
    """Silently asks the model whether this message contains a NEW fact worth remembering."""
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

    result = ollama.chat(model="llama3.2", messages=extraction_prompt)
    fact = result["message"]["content"].strip()

    if fact.upper() == "NONE" or len(fact) == 0:
        return None
    return fact

memories = load_memory()
memory_text = "\n".join(memories) if memories else "Nothing yet."

conversation = [
    {"role": "system", "content": f"Here is what you currently remember about the user:\n{memory_text}"}
]

print("Mimir is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Mimir: Goodbye for now.")
        break

    conversation.append({"role": "user", "content": user_input})

    response = ollama.chat(
        model="llama3.2",
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