"""Paired-seed generation and teacher-forced loss, with honest stopping metrics."""
import math
import random
import re


def repetition_metrics(text):
    words = re.findall(r"\w+", text.lower())
    ngrams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    duplicate_fraction = 1 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0
    # A repeated contiguous phrase must occur four times to be classified as a loop.
    loop = any(words[i:i + width] * 4 == words[i:i + width * 4]
               for width in range(1, 9) for i in range(max(0, len(words) - width * 4 + 1)))
    return {'repeated_4gram_fraction': duplicate_fraction, 'loop': loop}


def stop_token_ids(model, tokenizer):
    ids = model.generation_config.eos_token_id
    ids = [] if ids is None else ([ids] if isinstance(ids, int) else list(ids))
    return sorted(set(ids + [tokenizer.eos_token_id]))


def describe_stop(ids, eos_ids, limit):
    ended = bool(ids and ids[-1] in eos_ids)
    return 'eos' if ended else ('length' if len(ids) >= limit else 'other')


def generation_settings(model, tokenizer, config):
    return dict(do_sample=True, temperature=config['temperature'], top_p=config['top_p'],
                top_k=config['top_k'], max_new_tokens=config['max_new_tokens'], min_new_tokens=0,
                eos_token_id=stop_token_ids(model, tokenizer), pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.0, use_cache=True)


def select_examples(examples, config):
    return random.Random(config['seed']).sample(examples, min(config['eval_examples'], len(examples)))


def evaluate_generation(model, tokenizer, examples, config):
    import torch
    from transformers import set_seed
    model.eval()
    eos = stop_token_ids(model, tokenizer)
    settings = generation_settings(model, tokenizer, config)
    device = model.get_input_embeddings().weight.device
    rows = []
    old_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            for example in select_examples(examples, config):
                prompt = tokenizer.apply_chat_template(example['messages'], tokenize=True, return_dict=False,
                                                       add_generation_prompt=True)
                if len(prompt) + config['max_new_tokens'] > config['max_context_tokens']:
                    raise ValueError('Evaluation prompt was not fitted to the context budget.')
                inputs = torch.tensor([prompt], dtype=torch.long, device=device)
                for seed in config['eval_seeds']:
                    set_seed(seed)
                    output = model.generate(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        **settings,
                    )
                    ids = output[0, len(prompt):].tolist()
                    reply = tokenizer.decode(ids, skip_special_tokens=True)
                    rows.append({'id': example['id'], 'seed': seed, 'messages': example['messages'],
                                 'reference': example['target'], 'reply': reply, 'tokens': len(ids),
                                 'last_token_id': ids[-1] if ids else None,
                                 'stop_reason': describe_stop(ids, eos, config['max_new_tokens']),
                                 'empty': not reply.strip(), **repetition_metrics(reply)})
    finally:
        model.config.use_cache = old_cache
    count = len(rows)
    if not count:
        raise ValueError('No evaluation examples.')
    summary = {'generations': count, 'examples': len(set(r['id'] for r in rows)),
               'stop_rate': sum(r['stop_reason'] == 'eos' for r in rows) / count,
               'length_limit_rate': sum(r['stop_reason'] == 'length' for r in rows) / count,
               'loop_rate': sum(r['loop'] for r in rows) / count,
               'empty_rate': sum(r['empty'] for r in rows) / count,
               'mean_generated_tokens': sum(r['tokens'] for r in rows) / count,
               'eos_token_ids': eos}
    return rows, summary


def target_loss(model, batches):
    import torch
    model.eval()
    device = model.get_input_embeddings().weight.device
    total_loss, tokens = 0.0, 0
    with torch.inference_mode():
        for row in batches:
            batch = {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in row.items()}
            count = sum(label != -100 for label in row['labels'][1:])
            loss = float(model(**batch, use_cache=False).loss)
            if not math.isfinite(loss):
                raise ValueError('Nonfinite evaluation loss; check precision/model/data before training.')
            total_loss += loss * count
            tokens += count
    return {'target_nll': total_loss / tokens, 'target_tokens': tokens}


def baseline_passes(summary, config):
    return (summary['stop_rate'] >= config['minimum_stop_rate'] and
            summary['loop_rate'] <= config['maximum_loop_rate'] and summary['empty_rate'] == 0)


def blind_review(before, after, seed):
    rng = random.Random(seed)
    after_by_key = {(r['id'], r['seed']): r for r in after}
    reviews, key = [], []
    for row in before:
        identity = (row['id'], row['seed'])
        pair = [('base', row['reply']), ('adapter', after_by_key[identity]['reply'])]
        rng.shuffle(pair)
        reviews.append({'id': row['id'], 'seed': row['seed'], 'messages': row['messages'],
                        'A': pair[0][1], 'B': pair[1][1], 'preferred': '',
                        'rubric': 'Grounded in patient context; appropriate spoken style; coherent complete turn; no invented facts or unsafe clinical instructions.',
                        'notes': ''})
        key.append({'id': row['id'], 'seed': row['seed'], 'A': pair[0][0], 'B': pair[1][0]})
    return reviews, key
