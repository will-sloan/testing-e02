def load_qwen_in_4bit(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("Run this training script on the lab computer with an NVIDIA GPU.")
    revision = "cdbee75f17c01a7cc42f958dc650907174af0554"
    torch.manual_seed(22)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, quantization_config=config,
        dtype=dtype, device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return model, tokenizer
