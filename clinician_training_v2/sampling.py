"""Reproducible selection from training examples that already fit the token budget."""
import random


def select_training_examples(batches, examples, config):
    """Limit training rows while preserving their alignment with tokenized inputs."""
    if len(batches) != len(examples):
        raise ValueError('Training batches and examples must have matching lengths.')
    if not examples:
        raise ValueError('No eligible training examples are available for selection.')
    limit = config['max_train_samples']
    indices = range(len(examples))
    if limit is not None and limit < len(examples):
        # Sample without replacement, retaining the prepared order for selected rows.
        indices = sorted(random.Random(config['seed']).sample(indices, limit))
    selected_batches = [batches[i] for i in indices]
    selected_examples = [examples[i] for i in indices]
    report = {
        'requested_limit': limit,
        'available_examples': len(examples),
        'selected_examples': len(selected_examples),
        'seed': config['seed'],
        'example_ids': [row['id'] for row in selected_examples],
        'patient_groups': len({row['group_id'] for row in selected_examples}),
    }
    return selected_batches, selected_examples, report
