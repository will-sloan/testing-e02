"""Training-only sample budgets preserve held-out groups and token eligibility."""
import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clinician_training_v2 import __main__ as cli
from clinician_training_v2.data import prepare_records
from clinician_training_v2.sampling import select_training_examples
from clinician_training_v2.settings import load_config, training_config, validate_config, write_jsonl
from clinician_training_v2.tests.test_tokenization import ChatTokenizer
from clinician_training_v2.tokenization import encode_split


class TrainingSampleLimitTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.examples = [
            {'id': f'example-{index}', 'group_id': f'patient-{index // 2}',
             'messages': [{'role': 'user', 'content': f'Question {index}'}],
             'target': f'Reply {index}'}
            for index in range(20)
        ]
        self.batches = [{'input_ids': [index]} for index in range(20)]

    def test_null_uses_every_eligible_example_in_existing_order(self):
        self.assertIsNone(self.config['max_train_samples'])
        batches, examples, report = select_training_examples(self.batches, self.examples, self.config)
        self.assertEqual(batches, self.batches)
        self.assertEqual(examples, self.examples)
        self.assertIsNone(report['requested_limit'])
        self.assertEqual(report['available_examples'], 20)
        self.assertEqual(report['selected_examples'], 20)
        self.assertEqual(report['example_ids'], [row['id'] for row in self.examples])
        self.assertEqual(report['patient_groups'], 10)
        self.assertEqual(report['seed'], self.config['seed'])

    def test_sample_is_reproducible_unique_and_keeps_batches_aligned(self):
        config = {**self.config, 'max_train_samples': 6}
        before = copy.deepcopy((self.batches, self.examples))
        result = select_training_examples(self.batches, self.examples, config)
        self.assertEqual(result, select_training_examples(self.batches, self.examples, config))
        batches, examples, report = result
        self.assertEqual(len(examples), 6)
        self.assertEqual(len({row['id'] for row in examples}), 6)
        for batch, example in zip(batches, examples):
            self.assertEqual(example['id'], f"example-{batch['input_ids'][0]}")
        self.assertEqual(report['requested_limit'], 6)
        self.assertEqual(report['available_examples'], 20)
        self.assertEqual(report['selected_examples'], 6)
        self.assertEqual(report['example_ids'], [row['id'] for row in examples])
        self.assertEqual(report['patient_groups'], len({row['group_id'] for row in examples}))
        self.assertEqual((self.batches, self.examples), before)

    def test_changing_seed_changes_the_selected_subset(self):
        first = select_training_examples(self.batches, self.examples,
                                         {**self.config, 'max_train_samples': 6, 'seed': 17})[2]
        second = select_training_examples(self.batches, self.examples,
                                          {**self.config, 'max_train_samples': 6, 'seed': 43})[2]
        self.assertNotEqual(set(first['example_ids']), set(second['example_ids']))

    def test_limit_equal_to_or_larger_than_available_does_not_duplicate_or_reorder(self):
        for limit in (20, 30):
            with self.subTest(limit=limit):
                batches, examples, report = select_training_examples(
                    self.batches, self.examples, {**self.config, 'max_train_samples': limit})
                self.assertEqual(batches, self.batches)
                self.assertEqual(examples, self.examples)
                self.assertEqual(report['requested_limit'], limit)
                self.assertEqual(report['selected_examples'], 20)

    def test_limit_validation_accepts_only_null_or_positive_integers(self):
        for value in (None, 1, 5000):
            with self.subTest(value=value):
                self.assertEqual(validate_config({**self.config, 'max_train_samples': value})[
                    'max_train_samples'], value)
        for value in (0, -1, True, False, 2.0, '100', 'none', []):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'max_train_samples'):
                    validate_config({**self.config, 'max_train_samples': value})

    def test_legacy_training_configuration_means_all_samples(self):
        legacy = dict(self.config)
        del legacy['max_train_samples']
        self.assertEqual(training_config(legacy), training_config(self.config))
        self.assertIsNone(training_config(legacy)['max_train_samples'])
        self.assertNotEqual(training_config(legacy),
                            training_config({**self.config, 'max_train_samples': 500}))

    def test_ten_percent_validation_groups_and_training_budget_leave_splits_stable(self):
        self.assertEqual(self.config['validation_fraction'], 0.10)
        self.assertEqual(self.config['test_fraction'], 0.15)
        records = [
            {'messages': [
                {'speaker': 'patient', 'text': f'I am preparing lunch for day {index}.'},
                {'speaker': 'clinician', 'text': f'What would help you prepare lunch for day {index}?'},
                {'speaker': 'patient', 'text': f'I can shop before day {index}.'},
                {'speaker': 'clinician', 'text': f'How does shopping fit with your plans for day {index}?'},
            ]}
            for index in range(20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'conversations.jsonl'
            write_jsonl(source, records)
            all_samples = prepare_records(source, self.config)
            limited = prepare_records(source, {**self.config, 'max_train_samples': 2})
        self.assertEqual(limited, all_samples)
        self.assertEqual(all_samples['report']['split_groups'], {'test': 3, 'validation': 2, 'train': 15})
        self.assertEqual(len(all_samples['train']), 30)
        groups = {split: {row['group_id'] for row in all_samples[split]} for split in cli.SPLITS}
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])

    def test_limit_changes_do_not_require_preparing_again(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            splits = {split: [dict(self.examples[index], group_id=split)]
                      for index, split in enumerate(cli.SPLITS)}
            from clinician_training_v2.settings import digest, write_json
            for split, rows in splits.items():
                write_jsonl(folder / f'{split}.jsonl', rows)
            write_json(folder / 'manifest.json', {
                'schema_version': cli.DATA_SCHEMA_VERSION, 'config': self.config,
                'split_hashes': {split: digest(rows) for split, rows in splits.items()},
            })
            loaded, _ = cli.load_data(folder, {**self.config, 'max_train_samples': 1})
            self.assertEqual(loaded, splits)

    def test_command_limits_training_after_token_filtering_and_keeps_validation(self):
        config = {**self.config, 'max_train_samples': 2, 'max_context_tokens': 128,
                  'max_new_tokens': 16, 'baseline_gate': False}
        training = copy.deepcopy(self.examples[:5])
        training[0]['target'] = 'x' * 500
        validation = copy.deepcopy(self.examples[10:13])
        splits = {'train': training, 'validation': validation, 'test': self.examples[15:]}
        tokenizer = ChatTokenizer()
        fitting_batches, fitting_examples, _ = encode_split(tokenizer, training, config)
        validation_batches, _, _ = encode_split(tokenizer, validation, config)
        expected_batches, _, expected_report = select_training_examples(fitting_batches, fitting_examples, config)

        class TrainerReached(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'training'
            args = argparse.Namespace(command='train', data='unused', output=str(output), resume=None)
            with patch.object(cli, 'load_data', return_value=(splits, {'split_hashes': {}})), \
                 patch('clinician_training_v2.runtime.environment', return_value={'packages': {}}), \
                 patch('clinician_training_v2.runtime.load_tokenizer', return_value=tokenizer), \
                 patch('clinician_training_v2.runtime.load_model', return_value=object()), \
                 patch('clinician_training_v2.runtime.add_adapter', return_value=object()), \
                 patch.object(cli, 'preflight', return_value={}), \
                 patch.object(cli, 'comparison', return_value=([], {})), \
                 patch('clinician_training_v2.runtime.make_trainer', side_effect=TrainerReached) as trainer, \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(TrainerReached):
                    cli.run_model_command(args, config)
            self.assertEqual(trainer.call_args.args[3], expected_batches)
            self.assertEqual(trainer.call_args.args[4], validation_batches)
            report = json.loads((output / 'training_samples.json').read_text(encoding='utf-8'))
            self.assertEqual(report['available_examples'], 4)
            self.assertEqual(report['selected_examples'], 2)
            self.assertEqual(report['example_ids'], expected_report['example_ids'])
            self.assertNotIn(training[0]['id'], report['example_ids'])


if __name__ == '__main__':
    unittest.main()
