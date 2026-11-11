def fit_context(tokenizer, examples, max_new_tokens, split_name):
    max_input_tokens = 2048 - max_new_tokens
    if max_input_tokens < 1:
        raise ValueError("The reply limit must be less than 2048 tokens.")
    fitted = []
    for index, (original_history, reply) in enumerate(examples, start=1):
        history = [message.copy() for message in original_history]
        keep_start = 1 if history and history[0]["role"] == "system" else 0
        while True:
            tokens = tokenizer.apply_chat_template(
                history, tokenize=True, return_dict=False, add_generation_prompt=True,
            )
            if len(tokens) <= max_input_tokens:
                break
            # Remove oldest whole turns, restarting at a user message.
            next_user = next(
                (i for i in range(keep_start + 1, len(history)) if history[i]["role"] == "user"),
                None,
            )
            if next_user is None:
                raise ValueError(
                    f"{split_name} example {index}: system prompt and latest turn exceed "
                    f"{max_input_tokens} input tokens. Review this example or reduce the reply limit."
                )
            del history[keep_start:next_user]

        # Never truncate the reference reply to make a training example fit.
        full_tokens = tokenizer.apply_chat_template(
            history + [{"role": "assistant", "content": reply}],
            tokenize=True, return_dict=False, add_generation_prompt=False,
        )
        if len(full_tokens) > 2048:
            raise ValueError(
                f"{split_name} example {index}: context plus full reference reply exceeds 2048 tokens. "
                "Review this example; its target has not been truncated."
            )
        removed = len(original_history) - len(history)
        if removed:
            print(
                f"{split_name} example {index}: removed {removed} oldest context messages; "
                f"using {len(tokens)}/{max_input_tokens} input tokens.", flush=True,
            )
        fitted.append((history, reply))
    return fitted
