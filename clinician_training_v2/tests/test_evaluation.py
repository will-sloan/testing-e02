import types
import unittest

from clinician_training_v2.evaluation import (
    baseline_passes, blind_review, describe_stop, repetition_metrics,
    select_examples, stop_token_ids,
)


class EvaluationTests(unittest.TestCase):
    def test_stop_ids_include_all_model_eos_and_tokenizer_end_marker(self):
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[151643, 151645]))
        tokenizer = types.SimpleNamespace(eos_token_id=151645)
        self.assertEqual(stop_token_ids(model, tokenizer), [151643, 151645])
        model.generation_config.eos_token_id = 151643
        self.assertEqual(stop_token_ids(model, tokenizer), [151643, 151645])
        model.generation_config.eos_token_id = None
        self.assertEqual(stop_token_ids(model, tokenizer), [151645])

    def test_stop_reason_distinguishes_eos_at_limit_from_truncation(self):
        self.assertEqual(describe_stop([1, 151643], [151643, 151645], 256), 'eos')
        self.assertEqual(describe_stop([1, 151645], [151643, 151645], 2), 'eos')
        self.assertEqual(describe_stop([1, 2], [151643, 151645], 2), 'length')
        self.assertEqual(describe_stop([1], [151643, 151645], 256), 'other')
        self.assertEqual(describe_stop([], [151645], 256), 'other')

    def test_repetition_detection_identifies_loop_not_normal_repeated_word(self):
        self.assertTrue(repetition_metrics('you were like, ' * 8)['loop'])
        self.assertTrue(repetition_metrics('yes yes yes yes')['loop'])
        self.assertFalse(repetition_metrics('You can plan lunch, and you can plan dinner.')['loop'])
        self.assertFalse(repetition_metrics('')['loop'])
        self.assertEqual(repetition_metrics('Short reply.')['repeated_4gram_fraction'], 0)
        self.assertGreater(repetition_metrics('you were like, ' * 8)['repeated_4gram_fraction'], .5)

    def test_baseline_gate_catches_each_mechanical_failure(self):
        config = {'minimum_stop_rate': .75, 'maximum_loop_rate': .1}
        summary = {'stop_rate': 1.0, 'loop_rate': 0, 'empty_rate': 0}
        self.assertTrue(baseline_passes(summary, config))
        for changes in ({'stop_rate': .74}, {'loop_rate': .11}, {'empty_rate': .01}):
            with self.subTest(changes=changes):
                self.assertFalse(baseline_passes({**summary, **changes}, config))

    def test_selection_is_reproducible_without_altering_input(self):
        examples = [{'id': str(i)} for i in range(50)]
        config = {'seed': 17, 'eval_examples': 8}
        chosen = select_examples(examples, config)
        self.assertEqual(chosen, select_examples(examples, config))
        self.assertEqual(len({row['id'] for row in chosen}), 8)
        self.assertEqual(examples, [{'id': str(i)} for i in range(50)])
        self.assertEqual(len(select_examples(examples[:2], config)), 2)

    def test_blind_review_pairs_by_id_and_seed_not_list_order(self):
        def row(identity, seed, reply):
            return {'id': identity, 'seed': seed, 'reply': reply,
                    'messages': [{'role': 'user', 'content': 'question'}]}
        before = [row('a', 17, 'base a17'), row('a', 43, 'base a43'), row('b', 17, 'base b17')]
        after = [row('b', 17, 'adapt b17'), row('a', 43, 'adapt a43'), row('a', 17, 'adapt a17')]
        reviews, key = blind_review(before, after, seed=11)
        self.assertEqual((reviews, key), blind_review(before, after, seed=11))
        self.assertEqual(len(reviews), 3)
        after_lookup = {(r['id'], r['seed']): r for r in after}
        for original, review, mapping in zip(before, reviews, key):
            identity = (review['id'], review['seed'])
            self.assertEqual(identity, (original['id'], original['seed']))
            self.assertEqual({review['A'], review['B']}, {original['reply'], after_lookup[identity]['reply']})
            self.assertEqual(review['preferred'], '')
            for label in ('A', 'B'):
                expected = original['reply'] if mapping[label] == 'base' else after_lookup[identity]['reply']
                self.assertEqual(review[label], expected)


if __name__ == '__main__':
    unittest.main()
