"""Opt-in test against the exact cached production tokenizer; no weights/network."""
import os
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('RUN_QWEN_TOKENIZER_TEST') == '1',
                     'Set RUN_QWEN_TOKENIZER_TEST=1 after caching the Qwen tokenizer')
class QwenTokenizerTests(unittest.TestCase):
    def test_real_chat_template_and_eos_supervision(self):
        from transformers import AutoTokenizer
        from clinician_training_v2.data import prepare_records
        from clinician_training_v2.settings import load_config
        from clinician_training_v2.tokenization import encode_split
        config = load_config()
        tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
                                                  local_files_only=True)
        samples = Path(__file__).resolve().parents[2] / 'sample_conversations'
        data = prepare_records(samples, config)
        for split in ('train', 'validation', 'test'):
            batches, kept, audit = encode_split(tokenizer, data[split], config)
            self.assertEqual(len(batches), len(data[split]))
            for batch, row in zip(batches, kept):
                target_ids = [label for label in batch['labels'] if label != -100]
                self.assertEqual(target_ids[-1], tokenizer.eos_token_id)
                self.assertEqual(tokenizer.decode(target_ids, skip_special_tokens=True), row['target'])
                prompt = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                                       add_generation_prompt=True)
                self.assertEqual(batch['input_ids'][:len(prompt)], prompt)
                self.assertEqual(batch['labels'][:len(prompt)], [-100] * len(prompt))

    def test_generation_and_loss_with_real_tokenizer_tiny_random_model(self):
        import math
        import torch
        from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
        from clinician_training_v2.data import prepare_records
        from clinician_training_v2.evaluation import evaluate_generation, target_loss
        from clinician_training_v2.settings import load_config
        from clinician_training_v2.tokenization import encode_split
        config = load_config()
        config.update(max_new_tokens=2, eval_examples=1, eval_seeds=[17])
        tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['model_revision'],
                                                  local_files_only=True)
        torch.set_num_threads(1)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, head_dim=8,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=4096, eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
            pad_token_id=tokenizer.pad_token_id))
        samples = Path(__file__).resolve().parents[2] / 'sample_conversations'
        records = prepare_records(samples, config)['validation'][:1]
        batches, examples, _ = encode_split(tokenizer, records, config)
        rows, summary = evaluate_generation(model, tokenizer, examples, config)
        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0]['stop_reason'], ('eos', 'length'))
        self.assertIn(tokenizer.eos_token_id, summary['eos_token_ids'])
        self.assertTrue(math.isfinite(target_loss(model, batches)['target_nll']))
