def create_examples(conversations, max_examples=None):
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be at least 1.")
    examples = []
    roles = {"clinician": "assistant", "assistant": "assistant", "user": "user"}
    for conversation in conversations:
        messages = conversation.get("messages") if isinstance(conversation, dict) else conversation
        if not isinstance(messages, list):
            raise ValueError("Each conversation must be a message list or an object with a messages list.")
        history = []
        for message in messages:
            fields = {key.lower(): value for key, value in message.items()}
            speaker = fields.get("speaker", fields.get("role", "")).strip().lower()
            text = fields.get("text", fields.get("content"))
            if speaker not in roles or not isinstance(text, str) or not text.strip():
                raise ValueError("Each message needs a clinician/user speaker and nonempty text.")
            role = roles[speaker]
            # Skip redacted targets, but keep their original text in later context.
            if role == "assistant" and history and "[redacted]" not in text.casefold():
                examples.append((history.copy(), text))
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
            history.append({"role": role, "content": text})
    if not examples:
        raise ValueError("No clinician replies with preceding context and without [REDACTED] were found.")
    return examples
