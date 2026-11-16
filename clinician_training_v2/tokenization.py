"""Explicit target-only labels; no packing and no implicit tokenizer truncation."""


class NoFittingExamples(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('No examples fit the context budget. Inspect the token audit, raise '
                         'the context limit, or explicitly choose context_policy=recent. '
                         'No targets were truncated.')


def encode_example(tokenizer, example, config):
    text_fields = [m['content'] for m in example['messages']] + [example['target']]
    if any(marker in text.casefold() for text in text_fields
           for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
        raise ValueError('Reserved model control token in example text; review the source.')
    history = [m.copy() for m in example['messages']]
    if not history or history[-1]['role'] != 'user':
        raise ValueError(f"{example['id']}: generation prompt must end with a patient turn.")
    original_count = len(history)
    while True:
        prompt = tokenizer.apply_chat_template(history, tokenize=True, return_dict=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            history + [{'role': 'assistant', 'content': example['target']}],
            tokenize=True, return_dict=False, add_generation_prompt=False,
        )
        if full[:len(prompt)] != prompt:
            raise ValueError('Chat-template prefix mismatch; refusing to guess the label boundary.')
        completion = full[len(prompt):]
        if tokenizer.eos_token_id not in completion:
            raise ValueError('The target is missing the tokenizer end-of-message token.')
        # Stop supervision at EOS, without scoring any template newline after it.
        full = full[:len(prompt) + completion.index(tokenizer.eos_token_id) + 1]
        if (len(full) <= config['max_context_tokens'] and
                len(prompt) + config['max_new_tokens'] <= config['max_context_tokens']):
            return {
                'input_ids': full, 'attention_mask': [1] * len(full),
                'labels': [-100] * len(prompt) + full[len(prompt):],
            }, {**example, 'messages': history}, {
                'id': example['id'], 'prompt_tokens': len(prompt),
                'target_tokens': len(full) - len(prompt),
                'removed_messages': original_count - len(history),
            }
        if config['context_policy'] == 'skip':
            return None, None, {'id': example['id'], 'reason': 'over_context_budget',
                                'prompt_tokens': len(prompt), 'target_tokens': len(full) - len(prompt)}
        keep_start = int(history[0]['role'] == 'system')
        next_user = next((i for i in range(keep_start + 1, len(history))
                          if history[i]['role'] == 'user'), None)
        if next_user is None:
            return None, None, {'id': example['id'], 'reason': 'latest_turn_and_target_too_long'}
        del history[keep_start:next_user]


def encode_split(tokenizer, records, config):
    batches, kept, audit = [], [], []
    for row in records:
        batch, example, entry = encode_example(tokenizer, row, config)
        audit.append(entry)
        if batch is not None:
            batches.append(batch)
            kept.append(example)
    if not kept:
        raise NoFittingExamples(audit)
    return batches, kept, audit


class CompletionCollator:
    """Mask padding by position, never by token ID (EOS may also be PAD)."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        length = max(len(f['input_ids']) for f in features)
        batch = {key: [] for key in ('input_ids', 'attention_mask', 'labels')}
        for feature in features:
            padding = length - len(feature['input_ids'])
            batch['input_ids'].append(feature['input_ids'] + [self.pad_token_id] * padding)
            batch['attention_mask'].append(feature['attention_mask'] + [0] * padding)
            batch['labels'].append(feature['labels'] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
