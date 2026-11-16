"""End-to-end preparation and integrity checks using temporary synthetic transcripts."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from clinician_training_v2 import __main__ as cli
from clinician_training_v2.settings import load_config, read_jsonl, write_jsonl


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'transcripts'
        self.source.mkdir()
        self.config = load_config()
        for index in range(6):
            conversation = {
                'participant_id': f'person-{index}', 'conversation_id': f'visit-{index}',
                'messages': [
                    {'speaker': 'patient', 'text': f'My concern number {index} is packing lunch.'},
                    {'speaker': 'clinician', 'text': f'For concern {index}, what makes packing lunch difficult?'},
                    {'speaker': 'patient', 'text': f'My work schedule number {index} changes each week.'},
                    {'speaker': 'clinician', 'text': f'For schedule {index}, what small preparation would feel manageable?'},
                ],
            }
            (self.source / f'patient-{index}.json').write_text(json.dumps(conversation), encoding='utf-8')

    def prepare(self, name, reviews=None):
        output = self.root / name
        args = argparse.Namespace(input=str(self.source), output=str(output), reviews=reviews)
        with redirect_stdout(io.StringIO()):
            cli.prepare(args, self.config)
        return output

    def write_reviews(self, name, rows):
        path = self.root / name
        write_jsonl(path, rows)
        return str(path)

    def test_prepare_round_trip_manifest_and_separate_patient_groups(self):
        output = self.prepare('prepared')
        splits, manifest = cli.load_data(output, self.config)
        self.assertEqual(set(splits), {'train', 'validation', 'test'})
        self.assertTrue(all(splits.values()))
        groups = {name: {row['group_id'] for row in rows} for name, rows in splits.items()}
        self.assertFalse(groups['train'] & groups['validation'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['validation'] & groups['test'])
        self.assertEqual(sum(len(rows) for rows in splits.values()), 12)
        self.assertEqual(manifest['config'], self.config)
        self.assertEqual(len(read_jsonl(output / 'review_queue.jsonl')), 12)

    def test_single_jsonl_folder_prepares_without_participant_ids(self):
        records = []
        for path in sorted(self.source.glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            del row['participant_id']
            records.append(row)
            path.unlink()
        write_jsonl(self.source / 'all_conversations.jsonl', records)
        output = self.prepare('jsonl-prepared')
        splits, manifest = cli.load_data(output, self.config)
        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(sum(len(rows) for rows in splits.values()), 12)
        self.assertTrue(all(splits.values()))
        self.assertEqual(len({row['group_id'] for rows in splits.values() for row in rows}), 6)

    def test_older_prepared_splits_require_regeneration(self):
        output = self.prepare('old-prepared')
        path = output / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['schema_version'] = 1
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Run prepare again'):
            cli.load_data(output, self.config)

    def test_review_decisions_are_repeatable_and_preserve_split_assignment(self):
        original = self.prepare('original')
        splits, _ = cli.load_data(original, self.config)
        train_rows = splits['train']
        approved_id, rejected_id = train_rows[0]['id'], train_rows[1]['id']
        queue = read_jsonl(original / 'review_queue.jsonl')
        for item in queue:
            if item['id'] == approved_id:
                item.update(decision='approve', target='What would make tomorrow feel a little easier?')
            elif item['id'] == rejected_id:
                item['decision'] = 'reject'
        review_path = self.write_reviews('decisions.jsonl', queue)
        first = self.prepare('reviewed-first', review_path)
        second = self.prepare('reviewed-second', review_path)
        first_splits, first_manifest = cli.load_data(first, self.config)
        second_splits, second_manifest = cli.load_data(second, self.config)
        self.assertEqual(first_splits, second_splits)
        self.assertEqual(first_manifest['split_hashes'], second_manifest['split_hashes'])
        # Feeding the emitted queue back in must retain edits and rejected examples.
        third = self.prepare('reviewed-third', str(first / 'review_queue.jsonl'))
        third_splits, third_manifest = cli.load_data(third, self.config)
        self.assertEqual(first_splits, third_splits)
        self.assertEqual(first_manifest['split_hashes'], third_manifest['split_hashes'])
        original_split = {row['id']: split for split, rows in splits.items() for row in rows}
        for split, rows in first_splits.items():
            for row in rows:
                self.assertEqual(original_split[row['id']], split)
        flat = {row['id']: row for rows in first_splits.values() for row in rows}
        self.assertNotIn(rejected_id, flat)
        self.assertEqual(flat[approved_id]['target'], 'What would make tomorrow feel a little easier?')
        self.assertTrue(flat[approved_id]['metadata']['reviewed'])
        self.assertTrue(any(row['id'] == rejected_id and row['reason'] == 'review_rejected'
                            for row in read_jsonl(first / 'rejected.jsonl')))

    def test_reviewed_queue_can_be_used_for_another_edit_pass(self):
        original = self.prepare('original')
        queue = read_jsonl(original / 'review_queue.jsonl')
        identity = queue[0]['id']
        original_hash = queue[0]['source_hash']
        queue[0].update(decision='approve', target='What could help you take that first step?')
        first = self.prepare('first', self.write_reviews('first-reviews.jsonl', queue))
        next_queue = read_jsonl(first / 'review_queue.jsonl')
        edited_row = next(row for row in next_queue if row['id'] == identity)
        self.assertEqual(edited_row['source_hash'], original_hash)
        edited_row.update(decision='approve', target='What would help you begin tomorrow?')
        second = self.prepare('second', self.write_reviews('second-reviews.jsonl', next_queue))
        splits, _ = cli.load_data(second, self.config)
        target = next(row for rows in splits.values() for row in rows if row['id'] == identity)
        self.assertEqual(target['target'], 'What would help you begin tomorrow?')

    def test_stale_review_hash_and_duplicate_ids_are_rejected(self):
        original = self.prepare('original')
        queue = read_jsonl(original / 'review_queue.jsonl')
        queue[0]['source_hash'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'stale review'):
            self.prepare('stale', self.write_reviews('stale.jsonl', queue))
        self.assertFalse((self.root / 'stale').exists())
        queue = read_jsonl(original / 'review_queue.jsonl')
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.prepare('duplicate', self.write_reviews('duplicate.jsonl', queue + [queue[0]]))

    def test_hand_edited_prepared_data_is_rejected(self):
        output = self.prepare('prepared')
        rows = read_jsonl(output / 'train.jsonl')
        rows[0]['target'] = 'A change made outside the review workflow.'
        write_jsonl(output / 'train.jsonl', rows)
        with self.assertRaisesRegex(ValueError, 'changed after preparation'):
            cli.load_data(output, self.config)

    def test_changed_preparation_settings_are_rejected(self):
        output = self.prepare('prepared')
        with self.assertRaisesRegex(ValueError, 'preparation settings differ'):
            cli.load_data(output, {**self.config, 'seed': self.config['seed'] + 1})

    def test_existing_output_is_never_overwritten(self):
        output = self.prepare('prepared')
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.prepare('prepared')
        self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)

    def test_all_over_budget_failure_still_writes_audit(self):
        from clinician_training_v2.tokenization import NoFittingExamples
        prepared = self.prepare('prepared')
        output = self.root / 'baseline'
        audit = [{'id': 'overlong', 'reason': 'over_context_budget',
                  'prompt_tokens': 10000, 'target_tokens': 50}]
        args = argparse.Namespace(command='baseline', data=str(prepared), output=str(output))
        with patch('clinician_training_v2.runtime.environment', return_value={'packages': {}}), \
             patch('clinician_training_v2.runtime.load_tokenizer', return_value=object()), \
             patch('clinician_training_v2.tokenization.encode_split', side_effect=NoFittingExamples(audit)):
            with self.assertRaisesRegex(ValueError, 'No examples fit'):
                cli.run_model_command(args, self.config)
        self.assertEqual(read_jsonl(output / 'validation_token_audit.jsonl'), audit)

    def test_cli_reports_cuda_failure_without_traceback(self):
        prepared = self.prepare('prepared')
        splits, _ = cli.load_data(prepared, self.config)
        rows = splits['validation']
        output = self.root / 'baseline'
        stderr = io.StringIO()
        argv = ['clinician_training_v2', 'baseline', '--data', str(prepared), '--output', str(output)]
        with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            with patch('clinician_training_v2.runtime.environment', return_value={'packages': {}}), \
                 patch('clinician_training_v2.runtime.load_tokenizer', return_value=object()), \
                 patch('clinician_training_v2.tokenization.encode_split', return_value=([], rows, [])), \
                 patch('clinician_training_v2.runtime.load_model', side_effect=RuntimeError('CUDA computer required')):
                with self.assertRaises(SystemExit) as error:
                    cli.main()
        self.assertEqual(error.exception.code, 1)
        self.assertIn('Error: CUDA computer required', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
