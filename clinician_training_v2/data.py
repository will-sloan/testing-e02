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

from .settings import load_config, validate_config


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
            "to allow conversation-level splitting when patient IDs are unavailable."
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
        "participant_key": ["conversation_record", source] if missing_id else ["participant", str(participant)],
        "source_file": source_file,
        "source_record": source,
        "source_id": source_id, "missing_participant_id": missing_id,
        "messages": messages, "source_indices": source_indices,
        "raw_message_count": len(raw_messages),
        "fingerprint": _digest(messages),
    }


def target_rejection(text, min_words=None, max_words=None):
    """Return a quality flag, using the central configuration for omitted limits."""
    if min_words is None or max_words is None:
        defaults = load_config()
        min_words = defaults['min_target_words'] if min_words is None else min_words
        max_words = defaults['max_target_words'] if max_words is None else max_words
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


def prepare_records(input_path: Path, config: dict) -> dict:
    """Build leakage-controlled train/validation/test examples from transcripts.

    The returned examples still contain sensitive transcript text. Keep all
    prepared files local and protected just like the source data.
    """
    config = validate_config({**load_config(), **config})
    records = [_normalize(record, source, source_file, config["require_participant_id"])
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

    participant_owner, transcript_owner = {}, {}
    for index, record in enumerate(records):
        keys = [(participant_owner, _digest(record["participant_key"])), (transcript_owner, record["fingerprint"])]
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
        clinician_turns = 0
        for turn, message in enumerate(record["messages"]):
            if message["role"] == "assistant":
                clinician_turns += 1
            if message["role"] != "assistant" or turn == 0 or record["messages"][turn - 1]["role"] != "user":
                continue
            raw_candidates += 1
            example_id = "example_" + _digest([record["source_id"], record["fingerprint"], turn])[:24]
            # Exclude opening clinician turns as targets only; later history stays complete.
            reason = ("initial_clinician_turn"
                      if clinician_turns <= config["skip_first_clinician_turns"] else
                      target_rejection(message["content"], config["min_target_words"], config["max_target_words"]))
            if reason:
                rejections.append({"id": example_id, "conversation_id": record["conversation_id"],
                                   "source_file": record["source_file"],
                                   "source_record": record["source_record"],
                                   "source_message_indices": record["source_indices"][turn],
                                   "turn_index": turn, "reason": reason})
                continue
            quality_candidates += 1
            examples.append({
                "id": example_id, "group_id": group_id, "conversation_id": record["conversation_id"],
                "messages": [{"role": "system", "content": config["system_prompt"]}] + [dict(m) for m in record["messages"][:turn]],
                "target": message["content"],
                "metadata": {"source_id": record["source_id"], "source_file": record["source_file"],
                             "source_record": record["source_record"], "turn_index": turn,
                             "source_message_indices": record["source_indices"][turn],
                             "target_words": len(message["content"].split())},
            })
        cap = config["max_examples_per_conversation"]
        if len(examples) > cap:
            random.Random(f"{config['seed']}:{record['fingerprint']}").shuffle(examples)
            for example in examples[cap:]:
                rejections.append({"id": example["id"], "conversation_id": record["conversation_id"], "reason": "conversation_cap"})
            examples = examples[:cap]
        if examples:
            groups.setdefault(group_id, []).extend(examples)

    group_ids = sorted(groups)
    if len(group_ids) < 3:
        raise ValueError(
            f"Need at least three independent participant groups with eligible replies; found {len(group_ids)} "
            "after opening-turn exclusions, quality filtering and duplicate grouping. "
            "Add conversations/participant IDs or review skip_first_clinician_turns and target filters."
        )
    random.Random(config["seed"]).shuffle(group_ids)
    count = len(group_ids)
    n_test = min(max(1, round(count * config["test_fraction"])), count - 2)
    n_validation = min(max(1, round(count * config["validation_fraction"])), count - n_test - 1)
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
    random.Random(config["seed"]).shuffle(output["train"])
    output["rejected"] = rejections
    output["report"] = {
        "conversations_loaded": len(records), "unique_conversations": len(seen_transcripts),
        "independent_groups": count,
        "conversations_missing_participant_id": sum(record["missing_participant_id"] for record in records),
        "raw_messages": sum(record["raw_message_count"] for record in records),
        "merged_turns": sum(len(record["messages"]) for record in records),
        "candidate_replies": raw_candidates, "quality_eligible_replies": quality_candidates,
        "skip_first_clinician_turns": config["skip_first_clinician_turns"],
        "split_examples": {split: len(output[split]) for split in split_groups},
        "split_groups": {split: len(split_groups[split]) for split in split_groups},
        "rejection_counts": dict(sorted(Counter(item["reason"] for item in rejections).items())),
        "seed": config["seed"],
        "warnings": (["Participant IDs missing: each conversation record is a separate split group. "
                      "If a patient appears in multiple conversations, add a shared participant_id "
                      "to prevent that patient appearing across splits."]
                     if any(record["missing_participant_id"] for record in records) else []),
    }
    return output
