"""GPU-dependent operations; preparation/tests do not import ML packages."""
import importlib.metadata
import math
import platform


def environment():
    names = ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}
    if versions['torch']:
        import torch
        info.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda)
        if torch.cuda.is_available():
            info.update(gpu=torch.cuda.get_device_name(0),
                        vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                        bf16=torch.cuda.is_bf16_supported())
    return info


def load_tokenizer(config):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'])
    if tokenizer.eos_token_id is None:
        raise ValueError('Model tokenizer has no end-of-message token.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    return tokenizer


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('Training/evaluation needs the Windows NVIDIA CUDA computer. '
                           'Preparation, review and unit tests can run without a GPU.')
    set_seed(config['seed'])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], revision=config['model_revision'], quantization_config=quant,
        dtype=dtype, device_map={'': 0}, attn_implementation='sdpa',
    )
    if config['max_context_tokens'] > model.config.max_position_embeddings:
        raise ValueError('Requested context exceeds the base model configuration.')
    return model


def add_adapter(model, config):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import set_seed
    set_seed(config['seed'])
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    adapter = LoraConfig(task_type='CAUSAL_LM', target_modules='all-linear',
        r=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'], use_rslora=True, bias='none')
    model = get_peft_model(model, adapter)
    model.print_trainable_parameters()
    return model


def make_trainer(model, tokenizer, config, train_rows, validation_rows, output_path):
    import torch
    from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
    from .tokenization import CompletionCollator
    microbatches = math.ceil(len(train_rows) / config['train_batch_size'])
    steps_per_epoch = math.ceil(microbatches / config['gradient_accumulation_steps'])
    total_steps = steps_per_epoch * config['epochs']
    eval_steps = max(1, steps_per_epoch // config['validation_checks_per_epoch'])
    bf16 = torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=config['train_batch_size'],
        per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        num_train_epochs=config['epochs'], learning_rate=config['learning_rate'],
        lr_scheduler_type='cosine', warmup_steps=math.ceil(total_steps * config['warmup_ratio']),
        weight_decay=config['weight_decay'], max_grad_norm=config['max_grad_norm'],
        optim='adamw_torch', bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps', eval_steps=eval_steps, save_strategy='steps', save_steps=eval_steps,
        save_total_limit=config['save_total_limit'], load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False,
        prediction_loss_only=True, logging_steps=config['logging_steps'], report_to='none',
        seed=config['seed'], data_seed=config['seed'], dataloader_num_workers=0,
        remove_unused_columns=False, label_names=['labels'],
    )
    return Trainer(model=model, args=args, train_dataset=train_rows,
        eval_dataset=validation_rows, data_collator=CompletionCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'])])
