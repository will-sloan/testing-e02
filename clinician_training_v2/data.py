"""Local transcript preparation, with participant separation and auditable exclusions.

This module uses only the standard library. It never rewrites clinical meaning,
creates summaries, or sends transcripts to a service. IDs are stable hashes for
bookkeeping, not a promise of de-identification.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re


ROLES = {
    "user": "user", "patient": "user", "participant": "user",
    "clinician": "assistant", "assistant": "assistant",
    "dietitian": "assistant", "provider": "assistant",
}
_NONVERBAL = re.compile(
    r"[\[(](?:laugh(?:s|ter|ing)?|sigh(?:s|ing)?|cough(?:s|ing)?|"
    r"silence|pause|inaudible|unintelligible|crosstalk|background noise)[\])]",
    re.IGNORECASE,
)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _fields(value, location):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{location}: expected an object with string keys.")
    result = {}
    for key, item in value.items():
        lowered = key.casefold()
        if lowered in result:
            raise ValueError(f"{location}: duplicate case-insensitive field {lowered!r}.")
        result[lowered] = item
    return result


def _read_records(input_path):
    path = Path(input_path)
    if not path.exists():
        raise ValueError(f"Transcript input does not exist: {path}")
    paths = sorted(p for p in path.iterdir() if p.suffix.casefold() in {".json", ".jsonl"}) if path.is_dir() else [path]
    if not paths:
        raise ValueError("No JSON or JSONL transcript files found.")
    for file in paths:
        if file.suffix.casefold() not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported transcript extension: {file.suffix}")
        with file.open(encoding="utf-8-sig") as stream:
            if file.suffix.casefold() == ".jsonl":
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(f"Invalid JSON in {file.name}, line {line_number}.") from error
                        yield record, f"{file.name}:{line_number}", file.name
            else:
                try:
                    record = json.load(stream)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in {file.name}.") from error
                yield record, file.name, file.name


def _normalize(record, source, source_file, require_participant_id):
    source_id = "source_" + _digest(source)[:20]
    if isinstance(record, list):
        fields = {"messages": record}
    else:
        fields = _fields(record, source_id)
    raw_messages = fields.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"{source_id}: messages must be a nonempty list.")
    participant = fields.get("participant_id")
    if participant is not None and (isinstance(participant, bool) or not isinstance(participant, (str, int))):
        raise ValueError(f"{source_id}: participant_id must be a nonempty string or integer.")
    if isinstance(participant, str):
        participant = participant.strip()
    missing_id = participant is None or participant == ""
    if missing_id and require_participant_id:
        raise ValueError(
            f"{source_id}: missing participant_id. Add a stable pseudonymous participant_id "
            "shared by all sessions for the same person. Set require_participant_id=false "
            "only if each source file belongs to a different patient."
        )
    conversation = fields.get("conversation_id", source)
    if isinstance(conversation, bool) or not isinstance(conversation, (str, int)) or not str(conversation).strip():
        raise ValueError(f"{source_id}: conversation_id must be a nonempty string or integer.")
    messages = []
    source_indices = []
    for index, message in enumerate(raw_messages):
        location = f"{source_id} message {index}"
        message = _fields(message, location)
        role = message.get("speaker", message.get("role"))
        text = message.get("text", message.get("content"))
        if not isinstance(role, str) or role.strip().casefold() not in ROLES:
            raise ValueError(f"{location}: unsupported speaker; explicitly map speakers to patient or clinician.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{location}: text/content must be a nonempty string.")
        if any(marker in text.casefold() for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
            raise ValueError(f"{location}: reserved model control token embedded in transcript text; review the source.")
        if "speaker" in message and "role" in message:
            alternate = message["role"]
            if not isinstance(alternate, str) or ROLES.get(alternate.strip().casefold()) != ROLES[role.strip().casefold()]:
                raise ValueError(f"{location}: speaker and role disagree.")
        if "text" in message and "content" in message:
            alternate = message["content"]
            if not isinstance(alternate, str) or " ".join(alternate.split()) != " ".join(text.split()):
                raise ValueError(f"{location}: text and content disagree.")
        role = ROLES[role.strip().casefold()]
        text = " ".join(text.split())
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += " " + text
            source_indices[-1].append(index)
        else:
            messages.append({"role": role, "content": text})
            source_indices.append([index])
    return {
        "conversation_id": "conversation_" + _digest([participant, conversation])[:20],
        "participant_key": ["file", source_file] if missing_id else ["participant", str(participant)],
        "source_file": source_file,
        "source_id": source_id, "missing_participant_id": missing_id,
        "messages": messages, "source_indices": source_indices,
        "raw_message_count": len(raw_messages),
        "fingerprint": _digest(messages),
    }


def target_rejection(text, min_words=1, max_words=500):
    """Return a quality flag, not a judgment of clinical correctness."""
    if re.search(r"[\[<]\s*redacted\s*[\]>]", text, flags=re.IGNORECASE):
        return "redacted_target"
    if not re.search(r"\w", _NONVERBAL.sub("", text), flags=re.UNICODE):
        return "nonverbal_target"
    words = re.findall(r"\b[\w'-]+\b", text.casefold())
    if len(words) < min_words:
        return "short_target"
    if len(words) > max_words:
        return "long_target"
    # Quarantine repeated ASR loops; retain the original text for manual review.
    for size in range(1, min(9, len(words) // 3 + 1)):
        repeats = 4 if size == 1 else 3
        for start in range(len(words) - size * repeats + 1):
            chunk = words[start:start + size]
            if words[start:start + size * repeats] == chunk * repeats:
                return "repeated_phrase_target"
    return None


def _validate_config(config):
    if not isinstance(config.get("system_prompt"), str) or not config["system_prompt"].strip():
        raise ValueError("system_prompt must be a nonempty string.")
    for name, default in [("seed", 42), ("min_target_words", 1), ("max_target_words", 500), ("max_examples_per_conversation", 100)]:
        value = config.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer.")
    if config.get("min_target_words", 1) > config.get("max_target_words", 500):
        raise ValueError("min_target_words cannot exceed max_target_words.")
    validation = config.get("validation_fraction", 0.15)
    test = config.get("test_fraction", 0.15)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1 for value in (validation, test)) or validation + test >= 1:
        raise ValueError("validation_fraction and test_fraction must be positive and sum to less than 1.")
    if not isinstance(config.get("require_participant_id", False), bool):
        raise ValueError("require_participant_id must be true or false.")


def prepare_records(input_path: Path, config: dict) -> dict:
    """Build leakage-controlled train/validation/test examples from transcripts.

    The returned examples still contain sensitive transcript text. Keep all
    prepared files local and protected just like the source data.
    """
    _validate_config(config)
    records = [_normalize(record, source, source_file, config.get("require_participant_id", False))
               for record, source, source_file in _read_records(input_path)]
    if not records:
        raise ValueError("No conversations found.")

    # Union participant sessions and duplicate transcripts before the split.
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(a, b):
        left, right = find(a), find(b)
        parents[max(left, right)] = min(left, right)

    participant_owner, transcript_owner, file_owner = {}, {}, {}
    missing_id_files = {record["source_file"] for record in records if record["missing_participant_id"]}
    for index, record in enumerate(records):
        keys = [(participant_owner, _digest(record["participant_key"])), (transcript_owner, record["fingerprint"])]
        # The dataset owner confirmed one file per patient. If a file has missing
        # IDs, keep all its records together, including any explicitly tagged ones.
        if record["source_file"] in missing_id_files:
            keys.append((file_owner, record["source_file"]))
        for owners, key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    rejections, groups = [], {}
    seen_transcripts = set()
    raw_candidates = 0
    quality_candidates = 0
    for index, record in enumerate(records):
        group_id = "group_" + _digest(records[find(index)]["participant_key"])[:20]
        if record["fingerprint"] in seen_transcripts:
            rejections.append({"conversation_id": record["conversation_id"], "reason": "duplicate_conversation"})
            continue
        seen_transcripts.add(record["fingerprint"])
        examples = []
        for turn, message in enumerate(record["messages"]):
            if message["role"] != "assistant" or turn == 0 or record["messages"][turn - 1]["role"] != "user":
                continue
            raw_candidates += 1
            example_id = "example_" + _digest([record["source_id"], record["fingerprint"], turn])[:24]
            reason = target_rejection(message["content"], config.get("min_target_words", 1), config.get("max_target_words", 500))
            if reason:
                rejections.append({"id": example_id, "conversation_id": record["conversation_id"],
                                   "source_file": record["source_file"],
                                   "source_message_indices": record["source_indices"][turn],
                                   "turn_index": turn, "reason": reason})
                continue
            quality_candidates += 1
            examples.append({
                "id": example_id, "group_id": group_id, "conversation_id": record["conversation_id"],
                "messages": [{"role": "system", "content": config["system_prompt"]}] + [dict(m) for m in record["messages"][:turn]],
                "target": message["content"],
                "metadata": {"source_id": record["source_id"], "source_file": record["source_file"], "turn_index": turn,
                             "source_message_indices": record["source_indices"][turn],
                             "target_words": len(message["content"].split())},
            })
        cap = config.get("max_examples_per_conversation", 100)
        if len(examples) > cap:
            random.Random(f"{config.get('seed', 42)}:{record['fingerprint']}").shuffle(examples)
            for example in examples[cap:]:
                rejections.append({"id": example["id"], "conversation_id": record["conversation_id"], "reason": "conversation_cap"})
            examples = examples[:cap]
        if examples:
            groups.setdefault(group_id, []).extend(examples)

    group_ids = sorted(groups)
    if len(group_ids) < 3:
        raise ValueError(
            f"Need at least three independent participant groups with eligible replies; found {len(group_ids)} "
            "after quality filtering and duplicate grouping. Add conversations/participant IDs or review target filters."
        )
    random.Random(config.get("seed", 42)).shuffle(group_ids)
    count = len(group_ids)
    n_test = min(max(1, round(count * config.get("test_fraction", 0.15))), count - 2)
    n_validation = min(max(1, round(count * config.get("validation_fraction", 0.15))), count - n_test - 1)
    split_groups = {
        "test": group_ids[:n_test],
        "validation": group_ids[n_test:n_test + n_validation],
        "train": group_ids[n_test + n_validation:],
    }
    output, seen_examples = {}, set()
    for split in ("train", "validation", "test"):
        output[split] = []
        for group_id in split_groups[split]:
            for example in groups[group_id]:
                fingerprint = _digest([example["messages"], example["target"]])
                if fingerprint in seen_examples:
                    rejections.append({"id": example["id"], "conversation_id": example["conversation_id"], "reason": "duplicate_example", "split": split})
                    continue
                seen_examples.add(fingerprint)
                output[split].append(example)
        if not output[split]:
            raise ValueError(f"The {split} split has no unique eligible examples. Add independent data or review duplicate transcripts.")
    random.Random(config.get("seed", 42)).shuffle(output["train"])
    output["rejected"] = rejections
    output["report"] = {
        "conversations_loaded": len(records), "unique_conversations": len(seen_transcripts),
        "independent_groups": count,
        "conversations_missing_participant_id": sum(record["missing_participant_id"] for record in records),
        "raw_messages": sum(record["raw_message_count"] for record in records),
        "merged_turns": sum(len(record["messages"]) for record in records),
        "candidate_replies": raw_candidates, "quality_eligible_replies": quality_candidates,
        "split_examples": {split: len(output[split]) for split in split_groups},
        "split_groups": {split: len(split_groups[split]) for split in split_groups},
        "rejection_counts": dict(sorted(Counter(item["reason"] for item in rejections).items())),
        "seed": config.get("seed", 42),
        "warnings": (["Participant IDs missing: grouped by source file, relying on the confirmed one-file-per-patient convention. Repeated patients across files require participant IDs."]
                     if any(record["missing_participant_id"] for record in records) else []),
    }
    return output
