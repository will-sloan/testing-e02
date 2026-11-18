import argparse
import copy
from pathlib import Path
import json

from .settings import (load_config, digest, read_jsonl, write_json, write_jsonl,
                       PROJECT_ROOT, PATH_KEYS, LEGACY_DATA_SETTINGS, training_config)

SPLITS = ('train', 'validation', 'test')
DATA_SCHEMA_VERSION = 2
DATA_KEYS = ('seed', 'system_prompt', 'require_participant_id', 'validation_fraction',
             'test_fraction', 'min_target_words', 'max_target_words', 'max_examples_per_conversation',
             'skip_first_clinician_turns')
MODEL_KEYS = ('model_id', 'model_revision', 'system_prompt', 'max_context_tokens',
              'max_new_tokens', 'context_policy')


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def review_hash(row):
    return row.get('metadata', {}).get('source_hash') or digest(
        {key: row[key] for key in ('id', 'messages', 'target')})


def apply_reviews(prepared, path):
    reviews = read_jsonl(path) if path else []
    originals = {row['id']: row for split in SPLITS for row in prepared[split]}
    decisions = {}
    for review in reviews:
        identity = review.get('id')
        if identity not in originals or identity in decisions:
            raise ValueError('Review IDs must be unique and match this prepared dataset.')
        if review.get('source_hash') != review_hash(originals[identity]):
            raise ValueError(f'{identity}: stale review; regenerate the review queue.')
        if review.get('decision') not in ('pending', 'approve', 'reject'):
            raise ValueError('Review decision must be pending, approve or reject.')
        decisions[identity] = review
    for split in SPLITS:
        kept = []
        for row in prepared[split]:
            row['metadata']['source_hash'] = review_hash(row)
            review = decisions.get(row['id'], {})
            if review.get('decision') == 'reject':
                prepared['rejected'].append({'id': row['id'], 'reason': 'review_rejected', 'split': split})
                continue
            approved = review.get('decision') == 'approve'
            if approved:
                target = review.get('target', row['target'])
                if not isinstance(target, str) or not target.strip():
                    raise ValueError(f"{row['id']}: reviewed target must be nonempty text.")
                if any(marker in target.casefold() for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>')):
                    raise ValueError('Reviewed targets cannot contain model control tokens.')
                row['target'] = ' '.join(target.split())
            row['metadata']['reviewed'] = approved
            row['metadata']['target_words'] = len(row['target'].split())
            kept.append(row)
        if not kept:
            raise ValueError(f'Review decisions left the {split} split empty.')
        prepared[split] = kept
    return prepared


def prepare(args, config):
    from .data import prepare_records
    prepared = prepare_records(Path(args.input), config)
    queue = [{'id': row['id'], 'source_hash': review_hash(row), 'decision': 'pending',
              'messages': row['messages'], 'target': row['target']}
             for split in SPLITS for row in prepared[split]]
    prepared = apply_reviews(prepared, args.reviews)
    # Keep rejected decisions too, so another review pass cannot resurrect them.
    previous_reviews = {r['id']: r for r in read_jsonl(args.reviews)} if args.reviews else {}
    for entry in queue:
        previous = previous_reviews.get(entry['id'], {})
        entry['decision'] = previous.get('decision', 'pending')
        if entry['decision'] == 'approve':
            entry['target'] = ' '.join(previous.get('target', entry['target']).split())
    # Reviewed edits must not introduce identical full prompt/target examples across splits.
    fingerprints = set()
    for split in SPLITS:
        for row in prepared[split]:
            fp = digest([row['messages'], row['target']])
            if fp in fingerprints:
                raise ValueError('Review introduced duplicate prompt/target examples; revise the reviews.')
            fingerprints.add(fp)
    output = fresh_directory(args.output)
    hashes = {}
    for split in SPLITS:
        write_jsonl(output / f'{split}.jsonl', prepared[split])
        hashes[split] = digest(prepared[split])
    write_jsonl(output / 'review_queue.jsonl', queue)
    write_jsonl(output / 'rejected.jsonl', prepared['rejected'])
    prepared['report']['post_review_examples'] = {s: len(prepared[s]) for s in SPLITS}
    prepared['report']['reviewed_examples'] = {s: sum(r['metadata']['reviewed'] for r in prepared[s]) for s in SPLITS}
    write_json(output / 'report.json', prepared['report'])
    write_json(output / 'manifest.json', {'schema_version': DATA_SCHEMA_VERSION, 'config': config, 'split_hashes': hashes})
    print(json.dumps(prepared['report'], indent=2))
    print(f'Prepared local transcripts in {output}. No model training was performed.')


def load_data(path, config):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != DATA_SCHEMA_VERSION:
        raise ValueError('Prepared data uses an older splitting format. Run prepare again into '
                         'a new directory to use conversation-level JSONL grouping.')
    prepared_config = {**LEGACY_DATA_SETTINGS, **manifest['config']}
    if any(prepared_config[key] != config[key] for key in DATA_KEYS):
        raise ValueError('Data preparation settings differ. Use the same --config or prepare a new dataset.')
    splits = {s: read_jsonl(path / f'{s}.jsonl') for s in SPLITS}
    for split, rows in splits.items():
        if digest(rows) != manifest['split_hashes'][split]:
            raise ValueError(f'{split} data changed after preparation. Use --reviews and re-prepare.')
    groups = [set(r['group_id'] for r in splits[s]) for s in SPLITS]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Patient groups overlap between dataset splits.')
    return splits, manifest


def save_run_metadata(output, config, manifest):
    from .runtime import environment
    write_json(output / 'config.json', config)
    write_json(output / 'run_manifest.json', {'config': config, 'split_hashes': manifest['split_hashes'],
                                            'environment': environment()})


def preflight(tokenizer, model, config, output):
    from .evaluation import evaluate_generation
    probes = [
        {'id': 'basic_arithmetic', 'messages': [{'role': 'user', 'content': 'What is 2 plus 2? Answer briefly.'}], 'target': '4.'},
        {'id': 'ordinary_conversation', 'messages': [{'role': 'system', 'content': config['system_prompt']},
          {'role': 'user', 'content': 'I keep forgetting to pack lunch. Can we talk about one small step?'}],
          'target': 'What would make packing lunch easier to remember?'},
    ]
    rows, summary = evaluate_generation(model, tokenizer, probes, config)
    write_jsonl(output / 'base_probes.jsonl', rows)
    write_json(output / 'base_probes_summary.json', summary)
    return summary


def comparison(model, tokenizer, examples, batches, config, output, label):
    from .evaluation import evaluate_generation, target_loss
    rows, summary = evaluate_generation(model, tokenizer, examples, config)
    summary.update(target_loss(model, batches))
    write_jsonl(output / f'{label}.jsonl', rows)
    write_json(output / f'{label}_summary.json', summary)
    print(f'{label}: {json.dumps(summary)}', flush=True)
    return rows, summary


def save_comparison(output, before, after, summaries, config):
    from .evaluation import blind_review, baseline_passes
    review, key = blind_review(before, after, config['seed'])
    write_jsonl(output / 'blind_review.jsonl', review)
    write_jsonl(output / 'blind_review_key.jsonl', key)
    write_json(output / 'comparison.json', {
        'base': summaries[0], 'adapter': summaries[1],
        'adapter_passes_mechanical_checks': baseline_passes(summaries[1], config),
        'note': 'Stopping/loss metrics do not measure clinical correctness or prove improvement. '
                'Review paired generations before choosing this adapter.',
    })


def run_model_command(args, config):
    from .runtime import load_model, load_tokenizer, add_adapter, make_trainer
    from .tokenization import encode_split, NoFittingExamples
    from .evaluation import baseline_passes
    from .sampling import select_training_examples
    splits, manifest = load_data(args.data, config)
    if args.command == 'train' and config['require_reviewed_targets']:
        if any(not r['metadata'].get('reviewed', False) for s in ('train', 'validation') for r in splits[s]):
            raise ValueError('require_reviewed_targets=true: approve training and validation targets '
                             'in review_queue.jsonl and run prepare --reviews into a new directory.')
    if args.command == 'train' and args.resume:
        # Checkpoints live at RUN/checkpoints/checkpoint-N.
        old_manifest = Path(args.resume).parent.parent / 'run_manifest.json'
        previous = json.loads(old_manifest.read_text(encoding='utf-8'))
        if (training_config(previous['config']) != training_config(config) or
                previous['split_hashes'] != manifest['split_hashes']):
            raise ValueError('Resume requires the original configuration and unchanged prepared data.')
    if args.command == 'evaluate':
        previous = json.loads((Path(args.adapter).parent / 'run_manifest.json').read_text(encoding='utf-8'))
        if any(previous['config'][key] != config[key] for key in MODEL_KEYS):
            raise ValueError('Adapter evaluation must use the same model, prompt, and context settings.')
        if previous['split_hashes'] != manifest['split_hashes']:
            raise ValueError('Use the original prepared dataset when evaluating this adapter.')
    output = fresh_directory(args.output)
    save_run_metadata(output, config, manifest)
    tokenizer = load_tokenizer(config)
    split = args.split if args.command == 'evaluate' else 'validation'
    try:
        batches, examples, audit = encode_split(tokenizer, splits[split], config)
    except NoFittingExamples as error:
        write_jsonl(output / f'{split}_token_audit.jsonl', error.audit)
        raise
    write_jsonl(output / f'{split}_token_audit.jsonl', audit)
    print(f'{split}: {len(examples)}/{len(splits[split])} examples fit the context budget.', flush=True)
    write_json(output / 'coverage.json', {
        'split': split, 'kept': len(examples), 'total': len(splits[split]),
        'references_longer_than_generation_limit': sum(a.get('target_tokens', 0) > config['max_new_tokens'] for a in audit if 'reason' not in a),
        'note': 'Long reference replies can legitimately require more output tokens; inspect length-limit cases.',
    })
    model = load_model(config)
    probes = preflight(tokenizer, model, config, output)
    before, base_summary = comparison(model, tokenizer, examples, batches, config, output, 'base')
    if args.command == 'baseline':
        print('Inspect base.jsonl and base_probes.jsonl before training; metrics cannot assess clinical quality.')
        return
    if args.command == 'evaluate':
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        after, summary = comparison(model, tokenizer, examples, batches, config, output, 'adapter')
        save_comparison(output, before, after, (base_summary, summary), config)
        return
    if config['baseline_gate'] and (not baseline_passes(base_summary, config) or not baseline_passes(probes, config)):
        raise ValueError('Baseline failed stopping/repetition/empty-output checks. No training was performed. '
                         'Inspect base outputs, stop IDs, target lengths and software versions in the run folder. '
                         'baseline_gate=false is an explicit diagnostic override, not a quality fix.')
    try:
        train_batches, train_examples, train_audit = encode_split(tokenizer, splits['train'], config)
    except NoFittingExamples as error:
        write_jsonl(output / 'train_token_audit.jsonl', error.audit)
        raise
    write_jsonl(output / 'train_token_audit.jsonl', train_audit)
    train_batches, train_examples, sample_report = select_training_examples(train_batches, train_examples, config)
    sample_report['prepared_examples'] = len(splits['train'])
    write_json(output / 'training_samples.json', sample_report)
    print(f'Training: {sample_report["selected_examples"]}/{sample_report["available_examples"]} '
          f'eligible examples selected ({sample_report["prepared_examples"]} before token filtering); '
          f'{sample_report["patient_groups"]} patient groups.', flush=True)
    if len(train_examples) < 100 or len(examples) < 20:
        print('Small dataset: treat this as a pipeline experiment, not evidence of generalization.', flush=True)
    model = add_adapter(model, config)
    trainer = make_trainer(model, tokenizer, config, train_batches, batches, output / 'checkpoints')
    trainer.train(resume_from_checkpoint=args.resume or None)
    # Trainer restores the validation-loss-selected checkpoint, not the last update.
    model = trainer.model
    model.save_pretrained(output / 'adapter')
    tokenizer.save_pretrained(output / 'adapter')
    from .evaluation import generation_settings
    saved_generation = copy.deepcopy(model.generation_config)
    saved_generation.update(**generation_settings(model, tokenizer, config))
    saved_generation.save_pretrained(output / 'adapter')
    trainer.save_state()
    write_json(output / 'selection.json', {'best_checkpoint': trainer.state.best_model_checkpoint,
        'best_validation_loss': trainer.state.best_metric, 'global_step': trainer.state.global_step})
    after, summary = comparison(model, tokenizer, examples, batches, config, output, 'adapter')
    save_comparison(output, before, after, (base_summary, summary), config)
    print(f'Candidate adapter saved to {output / "adapter"}. The test split was not used for training or selection.')


def resolve_command_args(args, config):
    """Fill omitted flags from config and record the effective paths in the snapshot."""
    for key in PATH_KEYS:
        if config[key] is not None:
            path = Path(config[key]).expanduser()
            config[key] = str((PROJECT_ROOT / path).resolve())

    def use(flag, key):
        override = getattr(args, flag, None)
        if override is not None:
            # Preserve the existing CLI's working-directory-relative paths.
            config[key] = str(Path(override).expanduser().resolve())
        setattr(args, flag, config[key])

    if args.command == 'prepare':
        use('input', 'input_path')
        use('reviews', 'reviews_path')
        use('output', 'prepared_path')
    elif args.command != 'doctor':
        use('data', 'prepared_path')
        use('output', {'baseline': 'baseline_output', 'train': 'training_output',
                       'evaluate': 'evaluation_output'}[args.command])
        if args.command == 'train':
            use('resume', 'resume_checkpoint')
        elif args.command == 'evaluate':
            if config['adapter_path'] is None and getattr(args, 'adapter', None) is None:
                config['adapter_path'] = str(Path(config['training_output']) / 'adapter')
            use('adapter', 'adapter_path')
            if args.split is not None:
                config['evaluation_split'] = args.split
            args.split = config['evaluation_split']
    return args


def main():
    parser = argparse.ArgumentParser(description='Transcript-aware QLoRA training, separate from e02.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'baseline', 'train', 'evaluate', 'doctor'):
        command = commands.add_parser(name)
        command.add_argument('--config', help='JSON overrides for config.json')
        if name == 'prepare':
            command.add_argument('--input', help='Override input_path from config.json')
            command.add_argument('--reviews')
        elif name != 'doctor':
            command.add_argument('--data', help='Override prepared_path from config.json')
        if name != 'doctor':
            command.add_argument('--output', help='Override configured output; directory must be new')
        if name == 'train':
            command.add_argument('--resume', help='Existing checkpoint directory, same data/config required')
        if name == 'evaluate':
            command.add_argument('--adapter', help='Override adapter_path (defaults to training_output/adapter)')
            command.add_argument('--split', choices=('validation', 'test'))
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        args = resolve_command_args(args, config)
        if args.command == 'prepare':
            prepare(args, config)
        elif args.command == 'doctor':
            from .runtime import environment
            print(json.dumps(environment(), indent=2))
        else:
            run_model_command(args, config)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
