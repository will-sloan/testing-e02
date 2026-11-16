"""Optional real-library CPU smoke test; never downloads model weights."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'peft', 'accelerate')),
                     'Optional ML dependencies are not installed')
class RuntimeSmokeTests(unittest.TestCase):
    def test_tiny_qwen_adapter_training_and_reload(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast, TrainingArguments
        from peft import PeftModel
        from clinician_training_v2.runtime import add_adapter, make_trainer
        from clinician_training_v2.settings import load_config

        torch.set_num_threads(1)
        config = load_config()
        config.update(epochs=1, gradient_accumulation_steps=2, eval_examples=2)
        tiny_config = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2,
                                  num_key_value_heads=1, head_dim=8, max_position_embeddings=64,
                                  eos_token_id=2, pad_token_id=0, tie_word_embeddings=False)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1, '[EOS]': 2, 'hello': 3}, unk_token='[UNK]')),
            pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]')
        model = add_adapter(Qwen3ForCausalLM(tiny_config), config)
        rows = [{'input_ids': [3, 4, 5, 2], 'attention_mask': [1, 1, 1, 1], 'labels': [-100, -100, 5, 2]},
                {'input_ids': [3, 6, 7, 8, 2], 'attention_mask': [1] * 5, 'labels': [-100, -100, 7, 8, 2]}] * 2
        before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        def cpu_arguments(**kwargs):
            # Exercise production args/API, substituting only the hardware settings.
            kwargs.update(bf16=False, fp16=False, use_cpu=True, dataloader_pin_memory=False, disable_tqdm=True)
            return TrainingArguments(**kwargs)
        with tempfile.TemporaryDirectory() as directory, patch('transformers.TrainingArguments', side_effect=cpu_arguments):
            trainer = make_trainer(model, tokenizer, config, rows, rows[:1], Path(directory) / 'checkpoints')
            trainer.train()
            self.assertGreater(trainer.state.global_step, 0)
            self.assertTrue(trainer.state.best_model_checkpoint)
            self.assertTrue(any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before))
            self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate()['eval_loss'])))
            model.save_pretrained(Path(directory) / 'adapter')
            restored = PeftModel.from_pretrained(Qwen3ForCausalLM(tiny_config), Path(directory) / 'adapter')
            self.assertTrue(restored.peft_config['default'].use_rslora)
