"""Opening-turn exclusions retain context and preserve prepared-data compatibility."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from clinician_training_v2 import __main__ as cli
from clinician_training_v2.data import prepare_records
from clinician_training_v2.settings import load_config, training_config, validate_config, write_jsonl


class SkipClinicianTurnsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'conversations.jsonl'
        self.config = load_config()
        self.records = []
        for index in range(5):
            self.records.append({
                'conversation_id': f'visit-{index}',
                'messages': [
                    {'speaker': 'patient', 'text': f'I am thinking about lunch for day {index}.'},
                    {'speaker': 'clinician', 'text': f'What would work for lunch on day {index}?'},
                    {'speaker': 'patient', 'text': f'I could prepare soup for day {index}.'},
                    {'speaker': 'clinician', 'text': f'What would make soup preparation easier on day {index}?'},
                    {'speaker': 'patient', 'text': f'I can shop the evening before day {index}.'},
                    {'speaker': 'clinician', 'text': f'How does shopping ahead fit your plans for day {index}?'},
                ],
            })
        write_jsonl(self.source, self.records)

    @staticmethod
    def examples(prepared):
        return [row for split in cli.SPLITS for row in prepared[split]]

    def prepare(self, name, skip=0):
        config = {**self.config, 'skip_first_clinician_turns': skip}
        output = self.root / name
        args = argparse.Namespace(input=str(self.source), output=str(output), reviews=None)
        with redirect_stdout(io.StringIO()):
            cli.prepare(args, config)
        return output, config

    def test_zero_preserves_default_examples(self):
        self.assertEqual(self.config['skip_first_clinician_turns'], 0)
        default = prepare_records(self.source, {})
        explicit = prepare_records(self.source, {'skip_first_clinician_turns': 0})
        self.assertEqual(default, explicit)
        self.assertEqual(len(self.examples(default)), 15)
        self.assertEqual(default['report']['candidate_replies'], 15)
        self.assertEqual(default['report']['quality_eligible_replies'], 15)
        self.assertEqual(default['rejected'], [])

    def test_one_excludes_first_target_but_keeps_full_later_history(self):
        baseline = prepare_records(self.source, {'skip_first_clinician_turns': 0})
        skipped = prepare_records(self.source, {'skip_first_clinician_turns': 1})
        expected = {row['id']: row for row in self.examples(baseline)
                    if row['metadata']['turn_index'] > 1}
        actual = {row['id']: row for row in self.examples(skipped)}
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 10)
        for row in actual.values():
            self.assertEqual(row['messages'][2]['role'], 'assistant')
            self.assertTrue(row['messages'][2]['content'].startswith('What would work for lunch'))
        self.assertEqual(skipped['report']['candidate_replies'], 15)
        self.assertEqual(skipped['report']['quality_eligible_replies'], 10)
        self.assertEqual(skipped['report']['rejection_counts'], {'initial_clinician_turn': 5})
        initial = {row['id']: row for row in self.examples(baseline)
                   if row['metadata']['turn_index'] == 1}
        for rejection in skipped['rejected']:
            original = initial[rejection['id']]
            self.assertEqual(rejection['conversation_id'], original['conversation_id'])
            self.assertEqual(rejection['reason'], 'initial_clinician_turn')
            for key in ('source_file', 'source_record', 'source_message_indices', 'turn_index'):
                self.assertEqual(rejection[key], original['metadata'][key])

    def test_adjacent_clinician_fragments_count_as_one_turn(self):
        for record in self.records:
            # The assistant alias and clinician speaker must merge before counting.
            record['messages'].insert(2, {'role': 'assistant', 'content': 'Take your time to think.'})
        write_jsonl(self.source, self.records)
        skipped = prepare_records(self.source, {'skip_first_clinician_turns': 2})
        rows = self.examples(skipped)
        self.assertEqual(len(rows), 5)
        self.assertEqual(skipped['report']['rejection_counts'], {'initial_clinician_turn': 10})
        for row in rows:
            self.assertEqual(row['metadata']['turn_index'], 5)
            self.assertTrue(row['target'].startswith('How does shopping ahead'))
            self.assertTrue(row['messages'][2]['content'].endswith('Take your time to think.'))
        first_rejections = [row for row in skipped['rejected'] if row['turn_index'] == 1]
        self.assertEqual(len(first_rejections), 5)
        self.assertTrue(all(row['source_message_indices'] == [1, 2] for row in first_rejections))

    def test_leading_clinician_greeting_counts_without_becoming_a_candidate(self):
        for index, record in enumerate(self.records):
            record['messages'].insert(0, {'speaker': 'clinician', 'text': f'Welcome to visit {index}.'})
        write_jsonl(self.source, self.records)
        baseline = prepare_records(self.source, {'skip_first_clinician_turns': 0})
        skip_greeting = prepare_records(self.source, {'skip_first_clinician_turns': 1})
        self.assertEqual(self.examples(skip_greeting), self.examples(baseline))
        self.assertEqual(skip_greeting['rejected'], [])
        self.assertEqual(skip_greeting['report']['candidate_replies'], 15)
        skip_first_reply = prepare_records(self.source, {'skip_first_clinician_turns': 2})
        rows = self.examples(skip_first_reply)
        self.assertEqual(len(rows), 10)
        self.assertEqual(skip_first_reply['report']['rejection_counts'], {'initial_clinician_turn': 5})
        for row in rows:
            self.assertTrue(row['messages'][1]['content'].startswith('Welcome to visit'))
            self.assertTrue(row['messages'][3]['content'].startswith('What would work for lunch'))
        self.assertTrue(all(row['turn_index'] == 2 for row in skip_first_reply['rejected']))

    def test_skip_must_be_a_nonnegative_integer(self):
        for invalid in (-1, True, 1.0, '1', None):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, 'skip_first_clinician_turns.*nonnegative integer'):
                    validate_config({**self.config, 'skip_first_clinician_turns': invalid})

    def test_excluding_every_target_reports_no_eligible_groups(self):
        with self.assertRaisesRegex(ValueError, 'eligible replies; found 0.*skip_first_clinician_turns'):
            prepare_records(self.source, {'skip_first_clinician_turns': 3})

    def test_prepared_data_requires_the_same_skip_setting(self):
        output, config = self.prepare('prepared', skip=1)
        splits, manifest = cli.load_data(output, config)
        self.assertEqual(sum(map(len, splits.values())), 10)
        self.assertEqual(manifest['config']['skip_first_clinician_turns'], 1)
        for changed in (0, 2):
            with self.subTest(skip=changed):
                with self.assertRaisesRegex(ValueError, 'prepare a new dataset'):
                    cli.load_data(output, {**config, 'skip_first_clinician_turns': changed})

    def test_legacy_schema_two_manifest_means_zero_skipped_turns(self):
        output, config = self.prepare('legacy')
        path = output / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(manifest['schema_version'], 2)
        del manifest['config']['skip_first_clinician_turns']
        path.write_text(json.dumps(manifest), encoding='utf-8')
        splits, _ = cli.load_data(output, config)
        self.assertEqual(sum(map(len, splits.values())), 15)
        with self.assertRaisesRegex(ValueError, 'prepare a new dataset'):
            cli.load_data(output, {**config, 'skip_first_clinician_turns': 1})

    def test_legacy_training_config_matches_zero_but_not_nonzero(self):
        legacy = dict(self.config)
        del legacy['skip_first_clinician_turns']
        self.assertEqual(training_config(legacy), training_config(self.config))
        self.assertEqual(training_config(legacy)['skip_first_clinician_turns'], 0)
        self.assertNotEqual(training_config(legacy),
                            training_config({**self.config, 'skip_first_clinician_turns': 1}))


if __name__ == '__main__':
    unittest.main()
