from e02_helpers.add_context_prompt import add_context_prompt


def create_examples(conversations, max_examples=None, skip_first_clinician_messages=0):
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be at least 1.")
    if not isinstance(skip_first_clinician_messages, int) or skip_first_clinician_messages < 0:
        raise ValueError("skip_first_clinician_messages must be a nonnegative integer.")
    examples = []
    roles = {"clinician": "assistant", "assistant": "assistant", "user": "user"}
    for conversation in conversations:
        messages = conversation.get("messages") if isinstance(conversation, dict) else conversation
        if not isinstance(messages, list):
            raise ValueError("Each conversation must be a message list or an object with a messages list.")
        history = []
        clinician_messages = 0
        for message in messages:
            fields = {key.lower(): value for key, value in message.items()}
            speaker = fields.get("speaker", fields.get("role", "")).strip().lower()
            text = fields.get("text", fields.get("content"))
            if speaker not in roles or not isinstance(text, str) or not text.strip():
                raise ValueError("Each message needs a clinician/user speaker and nonempty text.")
            role = roles[speaker]
            if role == "assistant":
                clinician_messages += 1
            # Skip opening and redacted targets; retain every message in context.
            if (role == "assistant" and history
                    and clinician_messages > skip_first_clinician_messages
                    and "[redacted]" not in text.casefold()):
                examples.append((add_context_prompt(history), text))
                if max_examples is not None and len(examples) >= max_examples:
                    return examples
            history.append({"role": role, "content": text})
    if not examples:
        raise ValueError("No eligible replies remain after skipping opening and redacted targets. Try more conversations or a smaller skip setting.")
    return examples
