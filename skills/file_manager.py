import os
import shutil
import datetime

TRASH_DIR = os.path.join(os.path.expanduser("~"), "MimirTrash")


def _ensure_trash():
    os.makedirs(TRASH_DIR, exist_ok=True)


def list_directory(path):
    if not path or not os.path.isdir(path):
        return f"'{path}' is not a valid directory."
    try:
        entries = os.listdir(path)
    except PermissionError:
        return f"Permission denied trying to list '{path}'."
    if not entries:
        return f"'{path}' is empty."
    lines = [f"Contents of {path}:"]
    for e in sorted(entries):
        full = os.path.join(path, e)
        marker = "[DIR]" if os.path.isdir(full) else "[FILE]"
        lines.append(f"  {marker} {e}")
    return "\n".join(lines)


def read_file_preview(path, max_chars=3000):
    if not path or not os.path.isfile(path):
        return f"'{path}' is not a valid file."
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
    except PermissionError:
        return f"Permission denied trying to read '{path}'."
    except Exception as e:
        return f"Couldn't read '{path}': {type(e).__name__}"
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...(truncated, file is longer)"
    return f"Contents of {path}:\n{content}"


def _unique_destination(dst):
    """Never silently overwrites -- auto-renames on collision instead."""
    if not os.path.exists(dst):
        return dst
    base, ext = os.path.splitext(dst)
    counter = 1
    new_dst = f"{base} ({counter}){ext}"
    while os.path.exists(new_dst):
        counter += 1
        new_dst = f"{base} ({counter}){ext}"
    return new_dst


def copy_item(src, dst):
    if not src or not os.path.exists(src):
        return f"'{src}' doesn't exist -- nothing to copy."
    if not dst:
        return "No destination given for the copy."
    safe_dst = _unique_destination(dst)
    try:
        if os.path.isdir(src):
            shutil.copytree(src, safe_dst)
        else:
            shutil.copy2(src, safe_dst)
    except Exception as e:
        return f"Couldn't copy: {type(e).__name__}: {e}"
    if safe_dst != dst:
        return f"Copied to '{safe_dst}' (renamed automatically since '{dst}' already existed)."
    return f"Copied '{src}' to '{safe_dst}'."


def move_item(src, dst):
    if not src or not os.path.exists(src):
        return f"'{src}' doesn't exist -- nothing to move."
    if not dst:
        return "No destination given for the move."
    safe_dst = _unique_destination(dst)
    try:
        shutil.move(src, safe_dst)
    except Exception as e:
        return f"Couldn't move: {type(e).__name__}: {e}"
    if safe_dst != dst:
        return f"Moved to '{safe_dst}' (renamed automatically since '{dst}' already existed)."
    return f"Moved '{src}' to '{safe_dst}'."


def soft_delete(path):
    """Never permanently deletes -- moves to a Mimir Trash folder instead.
    Fully reversible; this is the only kind of 'delete' that happens without
    confirmation, precisely because it can't actually destroy anything."""
    if not path or not os.path.exists(path):
        return f"'{path}' doesn't exist -- nothing to delete."
    _ensure_trash()
    name = os.path.basename(path.rstrip("\\/"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(TRASH_DIR, f"{timestamp}_{name}")
    try:
        shutil.move(path, dst)
    except Exception as e:
        return f"Couldn't move to trash: {type(e).__name__}: {e}"
    return f"Moved '{path}' to Mimir Trash. Fully recoverable from {TRASH_DIR} if this was a mistake."


def list_trash():
    _ensure_trash()
    entries = os.listdir(TRASH_DIR)
    if not entries:
        return "Mimir Trash is empty."
    lines = ["Mimir Trash contents:"]
    for e in sorted(entries):
        lines.append(f"  {e}")
    return "\n".join(lines)


def empty_trash():
    """PERMANENT and irreversible. Must always be confirmed by the user
    before this is ever called -- this is the one true point of no return
    in the whole file management system."""
    _ensure_trash()
    entries = os.listdir(TRASH_DIR)
    if not entries:
        return "Mimir Trash is already empty."
    count = len(entries)
    for e in entries:
        full = os.path.join(TRASH_DIR, e)
        try:
            if os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
        except Exception:
            pass
    return f"Permanently deleted {count} item(s) from Mimir Trash. This cannot be undone."
