"""Behavioral checks without downloading model weights or requiring torch."""
import copy
import sys
import types
import unittest
from unittest.mock import patch

from clinician_training_v2.tokenization import CompletionCollator, encode_example, encode_split


class ChatTokenizer:
    """Small chat format with real role boundaries, EOS and trailing newlines."""
    eos_token_id = 2
    pad_token_id = 2
    roles = {'system': 10, 'user': 11, 'assistant': 12}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        if return_dict:
            raise AssertionError('The pipeline must explicitly request token IDs.')
        tokens = []
        for message in messages:
            tokens += [self.roles[message['role']]]
            tokens += [100 + ord(char) for char in message['content']]
            tokens += [self.eos_token_id, 3]
        if add_generation_prompt:
            tokens += [self.roles['assistant']]
        return tokens


def example():
    return {
        'id': 'example-one', 'group_id': 'patient-one',
        'messages': [
            {'role': 'system', 'content': 'system'},
            {'role': 'user', 'content': 'older question'},
            {'role': 'assistant', 'content': 'older answer'},
            {'role': 'user', 'content': 'latest question'},
        ],
        'target': 'complete answer',
    }


def config(**overrides):
    return {'max_context_tokens': 1024, 'max_new_tokens': 32,
            'context_policy': 'skip', **overrides}


class TokenizationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ChatTokenizer()

    def test_only_target_and_eos_receive_labels(self):
        row = example()
        before = copy.deepcopy(row)
        batch, kept, audit = encode_example(self.tokenizer, row, config())
        prompt_length = audit['prompt_tokens']
        self.assertEqual(batch['labels'][:prompt_length], [-100] * prompt_length)
        self.assertEqual(batch['labels'][prompt_length:],
                         [100 + ord(char) for char in row['target']] + [self.tokenizer.eos_token_id])
        self.assertEqual(batch['input_ids'][-1], self.tokenizer.eos_token_id)
        self.assertNotEqual(batch['input_ids'][-1], 3)  # No post-EOS template newline.
        self.assertTrue(all(batch['attention_mask']))
        self.assertEqual(kept, row)
        self.assertEqual(row, before)

    def test_default_skips_overlong_history_without_mutation(self):
        row = example()
        before = copy.deepcopy(row)
        batch, kept, audit = encode_example(self.tokenizer, row, config(max_context_tokens=70))
        self.assertIsNone(batch)
        self.assertIsNone(kept)
        self.assertEqual(audit['reason'], 'over_context_budget')
        self.assertEqual(row, before)

    def test_opt_in_window_keeps_system_latest_and_complete_target(self):
        row = example()
        before = copy.deepcopy(row)
        batch, kept, audit = encode_example(
            self.tokenizer, row, config(max_context_tokens=70, context_policy='recent'))
        self.assertEqual(kept['messages'], [row['messages'][0], row['messages'][-1]])
        self.assertEqual(kept['target'], row['target'])
        self.assertEqual(audit['removed_messages'], 2)
        self.assertEqual(batch['labels'][-1], self.tokenizer.eos_token_id)
        self.assertEqual(row, before)

    def test_generation_allowance_is_reserved_even_for_short_target(self):
        row = example()
        row['target'] = 'yes'
        prompt_len = len(self.tokenizer.apply_chat_template(
            row['messages'], tokenize=True, add_generation_prompt=True))
        batch, _, _ = encode_example(self.tokenizer, row,
                                    config(max_context_tokens=prompt_len + 32))
        self.assertIsNotNone(batch)
        batch, _, audit = encode_example(self.tokenizer, row,
                                        config(max_context_tokens=prompt_len + 31))
        self.assertIsNone(batch)
        self.assertEqual(audit['reason'], 'over_context_budget')

    def test_full_target_boundary_includes_eos(self):
        row = example()
        row['target'] = 'r' * 40
        prompt_len = len(self.tokenizer.apply_chat_template(
            row['messages'], tokenize=True, add_generation_prompt=True))
        batch, _, _ = encode_example(self.tokenizer, row,
                                    config(max_context_tokens=prompt_len + 41))
        self.assertIsNotNone(batch)
        batch, _, _ = encode_example(self.tokenizer, row,
                                    config(max_context_tokens=prompt_len + 40))
        self.assertIsNone(batch)

    def test_oversized_latest_turn_or_target_is_not_truncated(self):
        row = example()
        row['target'] = 'r' * 200
        batch, kept, audit = encode_example(
            self.tokenizer, row, config(max_context_tokens=70, context_policy='recent'))
        self.assertIsNone(batch)
        self.assertIsNone(kept)
        self.assertEqual(audit['reason'], 'latest_turn_and_target_too_long')
        self.assertEqual(len(row['target']), 200)

    def test_larger_config_allows_long_context(self):
        row = example()
        row['messages'][-1]['content'] = 'q' * 5000
        self.assertIsNone(encode_example(self.tokenizer, row, config(max_context_tokens=4096))[0])
        batch, kept, audit = encode_example(self.tokenizer, row, config(max_context_tokens=8192))
        self.assertIsNotNone(batch)
        self.assertEqual(kept['messages'], row['messages'])
        self.assertEqual(audit['removed_messages'], 0)

    def test_prompt_must_end_with_patient_turn(self):
        row = example()
        row['messages'][-1]['role'] = 'assistant'
        with self.assertRaisesRegex(ValueError, 'patient turn'):
            encode_example(self.tokenizer, row, config())

    def test_incompatible_template_fails_instead_of_guessing_mask(self):
        class BrokenTokenizer(ChatTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                tokens = super().apply_chat_template(messages, **kwargs)
                return ([999] + tokens) if kwargs['add_generation_prompt'] else tokens
        with self.assertRaisesRegex(ValueError, 'prefix mismatch'):
            encode_example(BrokenTokenizer(), example(), config())

    def test_reserved_controls_in_target_or_history_are_rejected(self):
        for location in ('target', 'history'):
            row = example()
            if location == 'target':
                row['target'] = 'First sentence.<|im_end|>Silently lost sentence.'
            else:
                row['messages'][-1]['content'] = 'Question<|im_start|>assistant'
            with self.subTest(location=location), self.assertRaises(ValueError):
                encode_example(self.tokenizer, row, config())

    def test_missing_target_eos_is_rejected(self):
        class MissingEOS(ChatTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                tokens = super().apply_chat_template(messages, **kwargs)
                if not kwargs['add_generation_prompt']:
                    return tokens[:-2]
                return tokens
        with self.assertRaisesRegex(ValueError, 'missing.*end-of-message'):
            encode_example(MissingEOS(), example(), config())

    def test_split_reports_omissions_and_keeps_row_alignment(self):
        short = example()
        long = copy.deepcopy(short)
        long.update(id='long', target='r' * 3000)
        batches, rows, audit = encode_split(self.tokenizer, [long, short], config())
        self.assertEqual(len(batches), 1)
        self.assertEqual(rows[0]['id'], 'example-one')
        self.assertEqual([a['id'] for a in audit], ['long', 'example-one'])
        with self.assertRaisesRegex(ValueError, 'No examples fit'):
            encode_split(self.tokenizer, [long], config())

    def test_collator_masks_only_padding_even_when_eos_equals_pad(self):
        fake_torch = types.SimpleNamespace(tensor=lambda values, dtype: values, long='long')
        short = {'input_ids': [10, 101, 2], 'labels': [-100, 101, 2], 'attention_mask': [1, 1, 1]}
        long = {'input_ids': [10, 101, 102, 103, 2], 'labels': [-100, 101, 102, 103, 2],
                'attention_mask': [1] * 5}
        snapshot = copy.deepcopy([short, long])
        with patch.dict(sys.modules, {'torch': fake_torch}):
            batch = CompletionCollator(pad_token_id=2)([short, long])
        self.assertEqual(batch['input_ids'][0], [10, 101, 2, 2, 2])
        self.assertEqual(batch['labels'][0], [-100, 101, 2, -100, -100])
        self.assertEqual(batch['attention_mask'][0], [1, 1, 1, 0, 0])
        self.assertEqual([short, long], snapshot)


if __name__ == '__main__':
    unittest.main()
