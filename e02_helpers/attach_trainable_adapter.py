def attach_trainable_adapter(qwen):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules="all-linear", bias="none",
    )
    return get_peft_model(qwen, config)
