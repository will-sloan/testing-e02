def fit_context(tokenizer, examples, max_new_tokens, split_name, *, max_context_tokens):
    max_input_tokens = max_context_tokens - max_new_tokens
    if max_input_tokens < 1:
        raise ValueError(f"The reply limit must be less than {max_context_tokens} tokens.")
    fitted = []
    for index, (original_history, reply) in enumerate(examples, start=1):
        history = [message.copy() for message in original_history]
        keep_start = 1 if history and history[0]["role"] == "system" else 0
        while True:
            tokens = tokenizer.apply_chat_template(
                history, tokenize=True, return_dict=False, add_generation_prompt=True,
            )
            # The reference may be longer than the generation allowance.
            # Keep it complete, including the template's end-of-message marker.
            full_tokens = tokenizer.apply_chat_template(
                history + [{"role": "assistant", "content": reply}],
                tokenize=True, return_dict=False, add_generation_prompt=False,
            )
            if len(tokens) <= max_input_tokens and len(full_tokens) <= max_context_tokens:
                break
            # Remove oldest whole turns, restarting at a user message.
            next_user = next(
                (i for i in range(keep_start + 1, len(history)) if history[i]["role"] == "user"),
                None,
            )
            if next_user is None:
                if len(full_tokens) > max_context_tokens:
                    raise ValueError(
                        f"{split_name} example {index}: system prompt, latest turn, and full "
                        f"reference reply require {len(full_tokens)} tokens, exceeding {max_context_tokens} "
                        "even after removing older context. Review or split this example; "
                        "its target has not been truncated."
                    )
                raise ValueError(
                    f"{split_name} example {index}: system prompt and latest turn exceed "
                    f"{max_input_tokens} input tokens. Review this example or reduce the reply limit."
                )
            del history[keep_start:next_user]

        removed = len(original_history) - len(history)
        if removed:
            print(
                f"{split_name} example {index}: removed {removed} oldest context messages; "
                f"using {len(tokens)}/{max_input_tokens} input tokens.", flush=True,
            )
        fitted.append((history, reply))
    return fitted
