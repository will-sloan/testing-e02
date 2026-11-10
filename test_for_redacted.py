import argparse
import json
from pathlib import Path


def read_conversations(path):
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise ValueError("No JSON or JSONL files found.")
    for file_path in files:
        with file_path.open(encoding="utf-8-sig") as file:
            if file_path.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(file, start=1):
                    if line.strip():
                        source = f"{file_path.name}, line {line_number}"
                        try:
                            conversation = json.loads(line)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON in {source}") from None
                        yield source, conversation
            elif file_path.suffix.lower() == ".json":
                try:
                    conversation = json.load(file)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON in {file_path.name}") from None
                yield file_path.name, conversation
            else:
                raise ValueError("Use a JSON file, JSONL file, or folder containing them.")


def count_redacted(path):
    counts = dict(conversations=0, conversations_redacted=0, samples=0,
                  targets_redacted=0, samples_redacted=0)
    for source, conversation in read_conversations(path):
        messages = conversation.get("messages") if isinstance(conversation, dict) else conversation
        if not isinstance(messages, list):
            raise ValueError(f"{source}: expected a message list or an object with a messages list.")
        history_redacted = False
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"{source}, message {index + 1}: expected an object.")
            fields = {key.lower(): value for key, value in message.items()}
            role = fields.get("speaker", fields.get("role", ""))
            text = fields.get("text", fields.get("content"))
            if not isinstance(role, str) or role.strip().lower() not in {"clinician", "assistant", "user"}:
                raise ValueError(f"{source}, message {index + 1}: unsupported speaker or role.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{source}, message {index + 1}: expected nonempty text.")
            redacted = "[redacted]" in text.casefold()
            # Match training: an assistant reply needs at least one preceding message.
            if role.strip().lower() in {"clinician", "assistant"} and index > 0:
                counts["samples"] += 1
                counts["targets_redacted"] += int(redacted)
                counts["samples_redacted"] += int(history_redacted or redacted)
            history_redacted = history_redacted or redacted
        counts["conversations"] += 1
        counts["conversations_redacted"] += int(history_redacted)
    return counts


def print_counts(label, total, with_redacted):
    without_redacted = total - with_redacted
    percent = without_redacted / total * 100 if total else 0
    print(f"\n{label}")
    print(f"  Total: {total:,}")
    print(f"  With [REDACTED]: {with_redacted:,}")
    print(f"  Without [REDACTED]: {without_redacted:,} ({percent:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=str(Path(__file__).parent / "real_conversations"))
    args = parser.parse_args()
    try:
        counts = count_redacted(args.path)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print("Checked all records; no training sample limit was applied.")
    print("Matching [REDACTED] regardless of capitalization. Source files were not changed.")
    print_counts("Training targets (clinician reply only)", counts["samples"], counts["targets_redacted"])
    print_counts("Training examples (preceding context plus target reply)", counts["samples"], counts["samples_redacted"])
    print_counts("Whole conversations (any message)", counts["conversations"], counts["conversations_redacted"])
    print("\nFor filtering only target replies, use the first section's 'Without' count.")


if __name__ == "__main__":
    main()
