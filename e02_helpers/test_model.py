def test_model(model, tokenizer, test_examples):
    import torch

    model.eval()
    results = []
    device = model.get_input_embeddings().weight.device
    with torch.no_grad():
        for history, reference in test_examples:
            tokens = tokenizer.apply_chat_template(history, tokenize=True, return_dict=False, add_generation_prompt=True)
            if len(tokens) + 192 > 2048:
                raise ValueError("Test context is too long to reserve 192 reply tokens.")
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            output = model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                max_new_tokens=192, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            reply = output[0, len(tokens):]
            results.append({
                "reference": reference,
                "reply": tokenizer.decode(reply, skip_special_tokens=True),
                "ended": tokenizer.eos_token_id in reply.tolist(),
            })
    return results
