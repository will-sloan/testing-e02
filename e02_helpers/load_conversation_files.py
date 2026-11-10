import json
from pathlib import Path


def load_conversation_files(folder_path, max_conversations=None):
    if max_conversations is not None and max_conversations < 1:
        raise ValueError("max_conversations must be at least 1.")
    path = Path(folder_path)
    paths = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    conversations = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            if path.suffix == ".jsonl":
                for line in file:
                    if line.strip():
                        conversations.append(json.loads(line))
                        if max_conversations is not None and len(conversations) >= max_conversations:
                            return conversations
            elif path.suffix == ".json":
                conversations.append(json.load(file))
            else:
                raise ValueError("Use a JSON file, JSONL file, or folder containing them.")
        if max_conversations is not None and len(conversations) >= max_conversations:
            return conversations
    if not conversations:
        raise ValueError("No conversations found.")
    return conversations
