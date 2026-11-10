import random


def split_by_conversation(conversations):
    groups = {}
    for index, conversation in enumerate(conversations):
        participant = conversation.get("participant_id") if isinstance(conversation, dict) else None
        key = ("participant", participant) if participant is not None else ("conversation", index)
        groups.setdefault(key, []).append(conversation)
    groups = list(groups.values())
    if len(groups) < 2:
        raise ValueError("Need at least two independent conversations or participant groups.")
    random.Random(42).shuffle(groups)
    position = max(1, min(int(len(groups) * 0.8), len(groups) - 1))
    train = [conversation for group in groups[:position] for conversation in group]
    test = [conversation for group in groups[position:] for conversation in group]
    return train, test
