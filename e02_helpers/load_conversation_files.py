import json
from pathlib import Path


def load_conversation_files(folder_path):
    path = Path(folder_path)
    paths = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    conversations = []
    for path in paths:
        with path.open(encoding="utf-8") as file:
            if path.suffix == ".jsonl":
                conversations.extend(json.loads(line) for line in file if line.strip())
            elif path.suffix == ".json":
                conversations.append(json.load(file))
            else:
                raise ValueError("Use a JSON file, JSONL file, or folder containing them.")
    if not conversations:
        raise ValueError("No conversations found.")
    return conversations
