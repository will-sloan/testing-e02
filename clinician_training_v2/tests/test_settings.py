import json
from pathlib import Path
import tempfile
import unittest

from clinician_training_v2.settings import digest, load_config, read_jsonl, write_json, write_jsonl


class SettingsTests(unittest.TestCase):
    def load_override(self, overrides):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            path.write_text(json.dumps(overrides), encoding='utf-8')
            return load_config(path)

    def test_defaults_and_partial_overrides(self):
        defaults = load_config()
        self.assertEqual(defaults['max_context_tokens'], 4096)
        self.assertEqual(defaults['context_policy'], 'skip')
        updated = self.load_override({'max_context_tokens': 8192})
        self.assertEqual(updated['max_context_tokens'], 8192)
        self.assertEqual(updated['model_id'], defaults['model_id'])
        self.assertEqual(load_config(), defaults)

    def test_misspelled_setting_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unknown configuration'):
            self.load_override({'max_content_tokens': 4096})

    def test_invalid_integer_settings_fail(self):
        for value in (0, -1, True, 1.5, '4096'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load_override({'max_context_tokens': value})

    def test_output_must_leave_space_for_prompt(self):
        for output in (4096, 5000):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.load_override({'max_new_tokens': output})

    def test_context_policy_is_explicit(self):
        with self.assertRaises(ValueError):
            self.load_override({'context_policy': 'truncate'})
        self.assertEqual(self.load_override({'context_policy': 'recent'})['context_policy'], 'recent')

    def test_invalid_split_and_sampling_settings_fail(self):
        invalid = [
            {'validation_fraction': 0}, {'test_fraction': 1},
            {'validation_fraction': .6, 'test_fraction': .4},
            {'eval_seeds': []}, {'eval_seeds': [True]}, {'eval_seeds': ['42']},
            {'learning_rate': 0}, {'temperature': 0}, {'top_p': 0}, {'top_p': 1.1},
            {'top_k': 0}, {'top_k': True}, {'minimum_stop_rate': 1.1},
            {'maximum_loop_rate': -1}, {'warmup_ratio': 2}, {'lora_dropout': -0.1},
        ]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.load_override(override)

    def test_malformed_config_values_raise_actionable_errors(self):
        invalid = [
            [], 'not-an-object',
            {'baseline_gate': 'false'}, {'require_participant_id': 0},
            {'require_reviewed_targets': None}, {'seed': -1}, {'seed': True},
            {'seed': 2**32}, {'eval_seeds': [1, 1]}, {'eval_seeds': [-1]},
            {'eval_seeds': '17'}, {'model_id': ''}, {'model_revision': None},
            {'system_prompt': '   '}, {'max_grad_norm': 0}, {'weight_decay': -1},
            {'min_target_words': 30, 'max_target_words': 2}, {'lora_dropout': 1},
            {'temperature': float('nan')}, {'learning_rate': float('inf')},
            {'validation_fraction': True}, {'max_grad_norm': 'one'},
        ]
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.load_override(override)

    def test_digests_are_stable_for_key_order_and_sensitive_to_content(self):
        self.assertEqual(digest({'a': 1, 'b': 2}), digest({'b': 2, 'a': 1}))
        self.assertNotEqual(digest({'a': 1}), digest({'a': 2}))

    def test_json_artifact_unicode_round_trip(self):
        records = [{'id': 'one', 'text': 'patient’s café'}, {'id': 'two', 'text': '你好'}]
        with tempfile.TemporaryDirectory() as folder:
            lines_path = Path(folder) / 'records.jsonl'
            write_jsonl(lines_path, records)
            self.assertEqual(read_jsonl(lines_path), records)
            object_path = Path(folder) / 'report.json'
            write_json(object_path, {'rows': records})
            self.assertEqual(json.loads(object_path.read_text(encoding='utf-8')), {'rows': records})


if __name__ == '__main__':
    unittest.main()
