import os
import subprocess

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")

OWN_SOURCE_FILES = [
    "chat.py",
    "skills/todo.py",
    "skills/conversation_log.py",
    "skills/gmail_reader.py",
    "skills/usage_tracker.py",
]

# Even a tiny change touching any of these ALWAYS requires human confirmation,
# regardless of how few lines it changes -- these are the places where a
# small mistake has an outsized, hard-to-notice impact.
SAFETY_CRITICAL_MARKERS = [
    "verify_challenged_claim", "_shares_grounding", "NO_OP_PREFIXES",
    "PROVIDER_CHAINS", "delete_task", "delete_all_pending", "os.remove",
    "shutil.rmtree", "subprocess", "eval(", "exec(", "credentials.json",
    "token.json", ".env", "SCOPES", "api_key",
]


def read_own_source(filename):
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def is_safety_critical(text):
    return any(marker in text for marker in SAFETY_CRITICAL_MARKERS)


def count_changed_lines(old_code, new_code):
    return max(len(old_code.splitlines()), len(new_code.splitlines()))


def apply_fix(filename, old_code, new_code):
    """Applies a code change directly to a real source file. Refuses rather
    than guessing if the exact old code can't be found, or if it's ambiguous
    (appears more than once) -- accuracy over convenience here."""
    path = os.path.join(PROJECT_ROOT, filename)
    if not os.path.exists(path):
        return False, f"File {filename} doesn't exist."

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if old_code not in content:
        return False, "Couldn't find the exact code to replace -- refusing to apply rather than guess."
    if content.count(old_code) > 1:
        return False, "That code appears more than once in the file -- too ambiguous to safely auto-replace."

    new_content = content.replace(old_code, new_code, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True, "Applied."


def git_commit(message):
    """Every applied self-fix becomes its own git commit -- this is the real
    undo mechanism. If a fix turns out to be wrong, 'git revert' undoes it
    cleanly."""
    try:
        subprocess.run(["git", "add", "."], cwd=PROJECT_ROOT, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], cwd=PROJECT_ROOT, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False