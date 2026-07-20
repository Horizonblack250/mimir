import ollama
import json

MEMORY_FILE = "memory.json"

def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)

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

    if user_input.lower().startswith("remember that"):
        memories.append(user_input)
        save_memory(memories)
        print("(saved to memory)")

    conversation.append({"role": "user", "content": user_input})

    response = ollama.chat(
        model="llama3.2",
        messages=conversation
    )

    reply = response["message"]["content"]
    print("Mimir:", reply)

    conversation.append({"role": "assistant", "content": reply})