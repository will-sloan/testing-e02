def test_model(model, tokenizer, test_examples, *, max_new_tokens, do_sample,
               temperature, top_p, top_k):
    import torch

    generation_settings = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    # Sampling controls only apply when sampling is enabled.
    if do_sample:
        generation_settings.update(temperature=temperature, top_p=top_p, top_k=top_k)

    model.eval()
    results = []
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        for history, reference in test_examples:
            tokens = tokenizer.apply_chat_template(history, tokenize=True, return_dict=False, add_generation_prompt=True)
            if len(tokens) + max_new_tokens > 2048:
                raise ValueError(f"Test context is too long to reserve {max_new_tokens} reply tokens.")
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            output = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                pad_token_id=tokenizer.pad_token_id,
                **generation_settings,
            )
            reply = output[0, len(tokens):]
            results.append({
                "reference": reference,
                "reply": tokenizer.decode(reply, skip_special_tokens=True),
                "ended": tokenizer.eos_token_id in reply.tolist(),
            })
    return results
