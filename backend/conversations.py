import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT / "data" / "conversations"
CONVERSATIONS_FILE = CONVERSATIONS_DIR / "conversations.json"

_lock = threading.Lock()


def _ensure_dir():
    CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONVERSATIONS_FILE.exists():
        with _lock:
            if not CONVERSATIONS_FILE.exists():
                CONVERSATIONS_FILE.write_text("{}", encoding="utf-8")


def _read_data() -> Dict[str, Dict]:
    _ensure_dir()
    try:
        with _lock:
            content = CONVERSATIONS_FILE.read_text(encoding="utf-8").strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception as exc:
        print(f"[ConversationsStore] Error reading {CONVERSATIONS_FILE}: {exc}")
        return {}


def _write_data(data: Dict[str, Dict]):
    _ensure_dir()
    try:
        with _lock:
            temp_file = CONVERSATIONS_FILE.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            temp_file.replace(CONVERSATIONS_FILE)
    except Exception as exc:
        print(f"[ConversationsStore] Error writing {CONVERSATIONS_FILE}: {exc}")


def list_conversations() -> List[Dict]:
    """Return all conversations ordered by last updated timestamp descending."""
    data = _read_data()
    conversations = list(data.values())
    # Sort descending by updatedAt
    conversations.sort(
        key=lambda c: c.get("updatedAt") or c.get("createdAt") or "",
        reverse=True
    )
    return conversations


def get_conversation(conv_id: str) -> Optional[Dict]:
    """Retrieve a single conversation by its stable ID."""
    if not conv_id:
        return None
    data = _read_data()
    return data.get(conv_id)


def save_conversation(conv: Dict) -> Dict:
    """Create or update a conversation with stable ID and timestamps."""
    data = _read_data()
    now_iso = datetime.now(timezone.utc).isoformat()

    conv_id = conv.get("id")
    if not conv_id:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        conv["id"] = conv_id
        conv["createdAt"] = now_iso

    if "createdAt" not in conv or not conv["createdAt"]:
        conv["createdAt"] = now_iso

    conv["updatedAt"] = now_iso

    if "title" not in conv or not conv["title"]:
        conv["title"] = "New Medical Chat"

    # Safeguard: if conversation already exists with messages and incoming is empty, preserve existing messages
    if conv_id in data:
        existing = data[conv_id]
        if existing.get("messages") and (not conv.get("messages") or len(conv["messages"]) == 0):
            conv["messages"] = existing["messages"]

    if "messages" not in conv or not isinstance(conv["messages"], list):
        conv["messages"] = []

    # Ensure every message has an id and timestamp
    for idx, msg in enumerate(conv["messages"]):
        if not msg.get("id"):
            msg["id"] = f"msg_{conv_id}_{idx}_{uuid.uuid4().hex[:6]}"
        if not msg.get("timestamp"):
            msg["timestamp"] = now_iso

    data[conv_id] = conv
    _write_data(data)
    return conv


def rename_conversation(conv_id: str, new_title: str) -> Optional[Dict]:
    """Rename a specific conversation without changing its ID or messages."""
    data = _read_data()
    if conv_id not in data:
        return None
    cleaned_title = (new_title or "").strip()
    if not cleaned_title:
        return data[conv_id]
    data[conv_id]["title"] = cleaned_title[:80]
    data[conv_id]["updatedAt"] = datetime.now(timezone.utc).isoformat()
    _write_data(data)
    return data[conv_id]


def delete_conversation(conv_id: str) -> bool:
    """Delete only the specified conversation."""
    data = _read_data()
    if conv_id in data:
        del data[conv_id]
        _write_data(data)
        return True
    return False


def append_message_to_conversation(conv_id: str, message_item: Dict) -> Optional[Dict]:
    """Append a user or assistant exchange to an existing or auto-created conversation."""
    data = _read_data()
    now_iso = datetime.now(timezone.utc).isoformat()

    if conv_id not in data:
        conv = {
            "id": conv_id,
            "title": message_item.get("question", "Medical Consultation")[:45],
            "createdAt": now_iso,
            "updatedAt": now_iso,
            "messages": []
        }
        data[conv_id] = conv
    else:
        conv = data[conv_id]

    # Deduplication check: prevent appending duplicate exchange
    msg_id = message_item.get("id")
    msg_q = (message_item.get("question") or "").strip()
    msg_a = (message_item.get("answer") or "").strip()

    for existing_msg in conv.get("messages", []):
        if msg_id and existing_msg.get("id") == msg_id:
            return conv
        if msg_q and existing_msg.get("question") == msg_q and msg_a and existing_msg.get("answer") == msg_a:
            return conv

    if not message_item.get("id"):
        message_item["id"] = f"msg_{conv_id}_{len(conv['messages'])}_{uuid.uuid4().hex[:6]}"
    if not message_item.get("timestamp"):
        message_item["timestamp"] = now_iso

    # Auto-title on first message
    if len(conv["messages"]) == 0 and message_item.get("question"):
        q = message_item["question"].strip()
        conv["title"] = q[:45] + ("..." if len(q) > 45 else "")

    conv["messages"].append(message_item)
    conv["updatedAt"] = now_iso
    data[conv_id] = conv
    _write_data(data)
    return conv
