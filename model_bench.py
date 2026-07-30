import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# NOTE: OpenRouter's free model ID list rotates -- if any of these 404,
# check openrouter.ai/models filtered to Price: Free and swap the ID.
PROVIDERS = {
    "groq_70b (current DEEP)": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "model": "llama-3.3-70b-versatile",
    },
    "groq_8b (current FAST)": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.environ.get("GROQ_API_KEY"),
        "model": "llama-3.1-8b-instant",
    },
    "openrouter_deepseek_r1": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "deepseek/deepseek-r1:free",
    },
    "openrouter_llama70b": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "gemini_2.5_pro": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model": "gemini-2.5-pro",
    },
    "gemini_2.5_flash": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "model": "gemini-2.5-flash",
    },
}

# Three tests: (1) careful reasoning vs. pattern-matching, (2) honesty about
# limitations, (3) strict instruction-following -- the exact weakness that
# caused most of Mimir's real bugs so far.
TEST_PROMPTS = [
    "A farmer has 17 sheep. All but 9 die. How many are left? Answer in one sentence.",
    "In one short sentence, what can't you know about events after your training cutoff?",
    'Reply with ONLY valid JSON, nothing else: {"capital_of_france": "..."}',
]


def test_provider(name, config):
    print(f"\n{'='*70}\n{name}  ({config['model']})\n{'='*70}")
    if not config["api_key"]:
        print("  SKIPPED -- no API key found in .env")
        return
    try:
        client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
        for prompt in TEST_PROMPTS:
            start = time.time()
            response = client.chat.completions.create(
                model=config["model"],
                messages=[{"role": "user", "content": prompt}]
            )
            elapsed = time.time() - start
            answer = response.choices[0].message.content.strip()
            print(f"\n  Q: {prompt}")
            print(f"  A: {answer}")
            print(f"  ({elapsed:.2f}s)")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    for name, config in PROVIDERS.items():
        test_provider(name, config)
