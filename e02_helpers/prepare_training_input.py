def prepare_training_input(tokenizer, conversation_so_far, clinician_reply, train_on_reply_only=True):
    import torch

    prompt = tokenizer.apply_chat_template(conversation_so_far, tokenize=True, return_dict=False, add_generation_prompt=True)
    messages = conversation_so_far + [{"role": "assistant", "content": clinician_reply}]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=False)
    if tokens[:len(prompt)] != prompt:
        raise ValueError("Chat template does not preserve the prompt prefix.")
    if len(tokens) > 2048:
        raise ValueError("Example exceeds 2048 tokens. Shorten its context before training.")
    if tokenizer.eos_token_id not in tokens[len(prompt):]:
        raise ValueError("The target must include an end-of-message token.")
    input_ids = torch.tensor([tokens], dtype=torch.long)
    labels = input_ids.clone()
    if train_on_reply_only:
        labels[:, :len(prompt)] = -100
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}
