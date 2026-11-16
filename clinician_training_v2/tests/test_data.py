import json
from pathlib import Path
import tempfile
import unittest

from clinician_training_v2.data import prepare_records, target_rejection


def conversation(index, participant=None):
    return {
        "participant_id": participant or f"participant-{index}",
        "conversation_id": f"session-{index}",
        "messages": [
            {"speaker": "patient", "text": f"I need lunch help for day {index}."},
            {"speaker": "clinician", "text": f"What lunch would work on day {index}?"},
            {"speaker": "patient", "text": f"I prefer soup for day {index}."},
            {"speaker": "clinician", "text": f"Could you prepare soup for day {index}?"},
        ],
    }


class TranscriptDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "transcripts.jsonl"
        self.config = {"system_prompt": "Write the next clinician response.", "seed": 42}

    def run_records(self, records, **settings):
        folder = Path(self.temp.name) / "records"
        folder.mkdir(exist_ok=True)
        for path in folder.glob("*.json"):
            path.unlink()
        for index, record in enumerate(records):
            (folder / f"record-{index}.json").write_text(json.dumps(record), encoding="utf-8")
        return prepare_records(folder, self.config | settings)

    def examples(self, result):
        return [example for split in ("train", "validation", "test") for example in result[split]]

    def test_fragments_merge_and_targets_exclude_future(self):
        special = conversation(0)
        special["messages"] = [
            {"Speaker": "Patient", "Text": " I need  lunch help. "},
            {"role": "participant", "content": "I work\nlate."},
            {"speaker": "dietitian", "text": "What options"},
            {"speaker": "provider", "text": "would work for you?"},
            {"speaker": "patient", "text": "FUTURE_SECRET"},
            {"speaker": "clinician", "text": "Tell me more about that."},
        ]
        result = self.run_records([special] + [conversation(i) for i in range(1, 5)])
        example = next(e for e in self.examples(result) if e["target"] == "What options would work for you?")
        self.assertEqual(example["messages"], [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": "I need lunch help. I work late."},
        ])
        self.assertEqual(example["metadata"]["source_message_indices"], [2, 3])
        self.assertNotIn("FUTURE_SECRET", json.dumps(example))

    def test_participant_sessions_never_cross_splits(self):
        records = [conversation(i, participant=f"person-{i // 2}") for i in range(12)]
        result = self.run_records(records)
        groups = [{e["group_id"] for e in result[split]} for split in ("train", "validation", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(result["report"]["independent_groups"], 6)
        self.assertEqual(result, self.run_records(records))

    def test_duplicate_transcripts_join_participant_groups(self):
        duplicate = conversation(0, participant="another-person")
        duplicate["conversation_id"] = "duplicate-session"
        extra_session = conversation(20, participant="another-person")
        result = self.run_records([conversation(i) for i in range(5)] + [duplicate, extra_session])
        self.assertEqual(result["report"]["independent_groups"], 5)
        self.assertEqual(result["report"]["rejection_counts"]["duplicate_conversation"], 1)
        zero = next(e for e in self.examples(result) if e["target"] == "What lunch would work on day 0?")
        twenty = next(e for e in self.examples(result) if e["target"] == "What lunch would work on day 20?")
        self.assertEqual(zero["group_id"], twenty["group_id"])

    def test_exact_examples_deduplicated_across_splits(self):
        records = [conversation(i) for i in range(10)]
        for record in records:
            record["messages"][:2] = [
                {"speaker": "patient", "text": "I need lunch help."},
                {"speaker": "clinician", "text": "What would help you?"},
            ]
        result = self.run_records(records)
        examples = self.examples(result)
        self.assertEqual(sum(e["target"] == "What would help you?" for e in examples), 1)
        self.assertEqual(result["report"]["rejection_counts"]["duplicate_example"], 9)

    def test_missing_ids_require_explicit_opt_out(self):
        records = [conversation(i)["messages"] for i in range(5)]
        with self.assertRaisesRegex(ValueError, "missing participant_id"):
            self.run_records(records, require_participant_id=True)
        result = self.run_records(records, require_participant_id=False)
        self.assertEqual(result["report"]["conversations_missing_participant_id"], 5)
        self.assertTrue(result["report"]["warnings"])

    def test_unknown_roles_and_empty_messages_fail(self):
        for message in ({"speaker": "Speaker 2", "text": "I need help."},
                        {"speaker": "patient", "text": " "},
                        {"speaker": "patient", "role": "clinician", "text": "I need help."},
                        {"speaker": "patient", "text": "I need help.", "content": "Different words."}):
            bad = conversation(0)
            bad["messages"][0] = message
            with self.assertRaises(ValueError):
                self.run_records([bad] + [conversation(i) for i in range(1, 5)])

    def test_filters_quarantine_without_printing_target_text(self):
        record = conversation(0)
        record["messages"][1]["text"] = "SECRET_NAME [REDACTED]"
        result = self.run_records([record] + [conversation(i) for i in range(1, 5)])
        self.assertEqual(result["report"]["rejection_counts"]["redacted_target"], 1)
        self.assertNotIn("SECRET_NAME", json.dumps(result["rejected"]))
        # Excluding a target does not silently erase it from later conversational history.
        later = next(e for e in self.examples(result) if e["target"] == "Could you prepare soup for day 0?")
        self.assertIn("SECRET_NAME", json.dumps(later["messages"]))

    def test_quality_flags_preserve_meaning(self):
        self.assertEqual(target_rejection("you were like, you were like, you were like"), "repeated_phrase_target")
        self.assertEqual(target_rejection("[laughter] (sighs)"), "nonverbal_target")
        self.assertIsNone(target_rejection("Okay."))
        self.assertEqual(target_rejection("Okay.", min_words=3), "short_target")
        self.assertEqual(target_rejection("word " * 501), "long_target")
        self.assertIsNone(target_rejection("Do not change your medication without discussing it with your prescriber."))
        self.assertIsNone(target_rejection("No, that does not mean you failed."))

    def test_cap_samples_beyond_first_turn_and_reports_exclusions(self):
        records = [conversation(i) for i in range(10)]
        result = self.run_records(records, max_examples_per_conversation=1)
        examples = self.examples(result)
        self.assertEqual(len(examples), 10)
        self.assertTrue(any(e["metadata"]["turn_index"] == 3 for e in examples))
        self.assertEqual(result["report"]["rejection_counts"]["conversation_cap"], 10)

    def test_three_independent_groups_required(self):
        with self.assertRaisesRegex(ValueError, "at least three independent"):
            self.run_records([conversation(0), conversation(1)])
        with self.assertRaisesRegex(ValueError, "at least three independent"):
            self.run_records([conversation(i, participant="same-person") for i in range(5)])

    def test_json_folder_and_bom(self):
        folder = Path(self.temp.name) / "json"
        folder.mkdir()
        for i in range(5):
            (folder / f"session-{i}.json").write_text(json.dumps(conversation(i)), encoding="utf-8-sig")
        result = prepare_records(folder, self.config)
        self.assertEqual(result["report"]["conversations_loaded"], 5)

    def test_invalid_config_and_schema(self):
        for settings in ({"seed": True}, {"validation_fraction": 0}, {"min_target_words": 10, "max_target_words": 3}, {"max_examples_per_conversation": 0}, {"require_participant_id": "false"}):
            with self.assertRaises(ValueError):
                self.run_records([conversation(i) for i in range(5)], **settings)
        bad = conversation(0)
        bad["messages"][0]["Speaker"] = "clinician"
        with self.assertRaisesRegex(ValueError, "duplicate case-insensitive"):
            self.run_records([bad] + [conversation(i) for i in range(1, 5)])

    def test_single_jsonl_without_ids_splits_each_conversation(self):
        for as_arrays in (False, True):
            records = [conversation(i) for i in range(6)]
            for record in records:
                del record['participant_id']
            if as_arrays:
                records = [record['messages'] for record in records]
            self.path.write_text('\n\n'.join(json.dumps(row) for row in records), encoding='utf-8-sig')
            result = prepare_records(self.path, self.config)
            self.assertEqual(result['report']['independent_groups'], 6)
            self.assertEqual(result['report']['conversations_loaded'], 6)
            self.assertEqual(len(self.examples(result)), 12)
            # Folder input discovers the same single JSONL file.
            self.assertEqual(result, prepare_records(self.path.parent, self.config))
            groups = [{e['group_id'] for e in result[s]} for s in ('train', 'validation', 'test')]
            self.assertTrue(all(groups))
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
            for example in self.examples(result):
                self.assertTrue(example['metadata']['source_record'].startswith('transcripts.jsonl:'))

    def test_single_jsonl_mixed_ids_preserves_known_patient_groups(self):
        records = [conversation(i, participant='shared-patient') for i in range(2)]
        for i in range(2, 6):
            record = conversation(i)
            del record['participant_id']
            records.append(record)
        self.path.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8')
        result = prepare_records(self.path, self.config)
        self.assertEqual(result['report']['independent_groups'], 5)
        examples = self.examples(result)
        zero = next(e for e in examples if e['target'] == 'What lunch would work on day 0?')
        one = next(e for e in examples if e['target'] == 'What lunch would work on day 1?')
        two = next(e for e in examples if e['target'] == 'What lunch would work on day 2?')
        self.assertEqual(zero['group_id'], one['group_id'])
        self.assertNotEqual(zero['group_id'], two['group_id'])

    def test_reserved_control_tokens_rejected_even_in_context(self):
        for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            record = conversation(0)
            record["messages"][0]["text"] = "A patient quoted " + marker
            with self.assertRaisesRegex(ValueError, "reserved model control token"):
                self.run_records([record] + [conversation(i) for i in range(1, 5)])


if __name__ == "__main__":
    unittest.main()
