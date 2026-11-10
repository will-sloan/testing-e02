def create_examples(conversations):
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
            # Use reviewed transcripts: every clinician reply after context is a target.
            if role == "assistant" and history:
                examples.append((history.copy(), text))
            history.append({"role": role, "content": text})
    if not examples:
        raise ValueError("No clinician replies with preceding context found.")
    return examples
