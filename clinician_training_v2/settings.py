import hashlib
import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name('config.json')
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PATH_KEYS = ('input_path', 'prepared_path', 'baseline_output', 'training_output',
             'evaluation_output', 'adapter_path', 'reviews_path', 'resume_checkpoint')
WORKFLOW_KEYS = (*PATH_KEYS, 'evaluation_split')
# Historical values only: allow resuming checkpoints predating configurable Trainer controls.
LEGACY_TRAINER_SETTINGS = dict(train_batch_size=1, eval_batch_size=1,
                               validation_checks_per_epoch=4, save_total_limit=2, logging_steps=1)
LEGACY_DATA_SETTINGS = {'skip_first_clinician_turns': 0}


def load_config(path=None):
    config = json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    if path:
        overrides = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(overrides, dict):
            raise ValueError('Configuration overrides must be a JSON object.')
        unknown = overrides.keys() - config.keys()
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        config.update(overrides)
    return validate_config(config)


def validate_config(config):
    for key in PATH_KEYS:
        value = config[key]
        if key in ('adapter_path', 'reviews_path', 'resume_checkpoint') and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{key} must be a nonempty path string' +
                             (' or null.' if key in ('adapter_path', 'reviews_path', 'resume_checkpoint') else '.'))
    if config['evaluation_split'] not in ('validation', 'test'):
        raise ValueError('evaluation_split must be validation or test.')
    for key in ('model_id', 'model_revision', 'system_prompt'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must be a nonempty string.')
    for key in ('require_participant_id', 'require_reviewed_targets', 'baseline_gate'):
        if type(config[key]) is not bool:
            raise ValueError(f'{key} must be true or false.')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**32:
        raise ValueError('seed must be a nonnegative integer smaller than 2**32.')
    if type(config['skip_first_clinician_turns']) is not int or config['skip_first_clinician_turns'] < 0:
        raise ValueError('skip_first_clinician_turns must be a nonnegative integer.')
    if config['max_train_samples'] is not None and (
            type(config['max_train_samples']) is not int or config['max_train_samples'] < 1):
        raise ValueError('max_train_samples must be a positive integer or null (all eligible training samples).')
    for key in ('max_context_tokens', 'max_new_tokens', 'epochs', 'gradient_accumulation_steps',
                'lora_rank', 'lora_alpha', 'eval_examples', 'early_stopping_patience',
                'min_target_words', 'max_target_words', 'max_examples_per_conversation',
                'train_batch_size', 'eval_batch_size', 'validation_checks_per_epoch',
                'save_total_limit', 'logging_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if config['min_target_words'] > config['max_target_words']:
        raise ValueError('min_target_words cannot exceed max_target_words.')
    if config['max_new_tokens'] >= config['max_context_tokens']:
        raise ValueError('max_new_tokens must be smaller than max_context_tokens.')
    if config['context_policy'] not in ('skip', 'recent'):
        raise ValueError('context_policy must be skip or recent.')
    numeric = ('validation_fraction', 'test_fraction', 'learning_rate', 'temperature',
               'top_p', 'minimum_stop_rate', 'maximum_loop_rate', 'warmup_ratio',
               'lora_dropout', 'weight_decay', 'max_grad_norm')
    for key in numeric:
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError(f'{key} must be a finite number.')
    if not 0 < config['validation_fraction'] < 1 or not 0 < config['test_fraction'] < 1:
        raise ValueError('Validation and test fractions must be between 0 and 1.')
    if config['validation_fraction'] + config['test_fraction'] >= 1:
        raise ValueError('Validation and test fractions must leave training data.')
    seeds = config['eval_seeds']
    if (not isinstance(seeds, list) or not seeds or
            any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError('eval_seeds must be a nonempty list of unique integers from 0 to 2**32 - 1.')
    if not 0 < config['learning_rate'] < 1 or config['temperature'] <= 0:
        raise ValueError('learning_rate must be between 0 and 1; temperature must be positive.')
    if not 0 < config['top_p'] <= 1 or type(config['top_k']) is not int or config['top_k'] < 1:
        raise ValueError('Invalid sampling configuration.')
    for key in ('minimum_stop_rate', 'maximum_loop_rate', 'warmup_ratio'):
        if not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be between zero and one.')
    if not 0 <= config['lora_dropout'] < 1:
        raise ValueError('lora_dropout must be between zero (inclusive) and one (exclusive).')
    if config['weight_decay'] < 0 or config['max_grad_norm'] <= 0:
        raise ValueError('weight_decay must be nonnegative and max_grad_norm must be positive.')
    return config


def training_config(config):
    """Paths are operational choices; they must not invalidate a training resume."""
    # Older runs used every eligible training sample; never infer this from today's config.
    merged = {'max_train_samples': None, **LEGACY_TRAINER_SETTINGS, **LEGACY_DATA_SETTINGS, **config}
    return {key: value for key, value in merged.items() if key not in WORKFLOW_KEYS}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path, records):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
