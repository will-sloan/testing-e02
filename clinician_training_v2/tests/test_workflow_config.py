"""Single-file workflow configuration and backwards-compatible CLI resolution."""
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clinician_training_v2.__main__ import resolve_command_args
from clinician_training_v2.settings import (
    LEGACY_TRAINER_SETTINGS, PATH_KEYS, PROJECT_ROOT, WORKFLOW_KEYS,
    load_config, training_config,
)


def arguments(command, **overrides):
    return argparse.Namespace(command=command, input=None, data=None, output=None,
                              reviews=None, resume=None, adapter=None, split=None,
                              **overrides)


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class WorkflowConfigTests(unittest.TestCase):
    def test_all_commands_use_config_without_path_flags(self):
        for command in ('prepare', 'baseline', 'train', 'evaluate', 'doctor'):
            with self.subTest(command=command):
                config = load_config()
                original = copy.deepcopy(config)
                args = resolve_command_args(arguments(command), config)
                if command == 'prepare':
                    self.assertEqual(args.input, str((PROJECT_ROOT / original['input_path']).resolve()))
                    self.assertEqual(args.output, str((PROJECT_ROOT / original['prepared_path']).resolve()))
                    self.assertIsNone(args.reviews)
                elif command != 'doctor':
                    self.assertEqual(args.data, str((PROJECT_ROOT / original['prepared_path']).resolve()))
                    key = {'baseline': 'baseline_output', 'train': 'training_output',
                           'evaluate': 'evaluation_output'}[command]
                    self.assertEqual(args.output, str((PROJECT_ROOT / original[key]).resolve()))
                if command == 'train':
                    self.assertIsNone(args.resume)
                elif command == 'evaluate':
                    self.assertEqual(args.adapter, str((PROJECT_ROOT / original['training_output'] / 'adapter').resolve()))
                    self.assertEqual(args.split, 'test')
                self.assertEqual(config['learning_rate'], original['learning_rate'])

    def test_config_paths_are_repo_relative_cli_paths_are_cwd_relative(self):
        with tempfile.TemporaryDirectory() as folder, working_directory(folder):
            config = load_config()
            config.update(input_path='inputs/transcripts.jsonl', prepared_path='prepared/configured',
                          reviews_path='reviews/decisions.jsonl')
            args = arguments('prepare')
            args.input = 'override/input.jsonl'
            args.output = 'override/prepared'
            args.reviews = 'override/reviews.jsonl'
            resolve_command_args(args, config)
            self.assertEqual(args.input, str((Path(folder) / 'override/input.jsonl').resolve()))
            self.assertEqual(args.output, str((Path(folder) / 'override/prepared').resolve()))
            self.assertEqual(args.reviews, str((Path(folder) / 'override/reviews.jsonl').resolve()))
            self.assertEqual(config['input_path'], args.input)
            self.assertEqual(config['prepared_path'], args.output)
            self.assertEqual(config['reviews_path'], args.reviews)
            # An unused configured path is still rooted at the repository, not cwd.
            self.assertEqual(config['training_output'], str((PROJECT_ROOT / 'clinician_training_v2/runs/dataset1-training').resolve()))
            config = load_config()
            config['input_path'] = 'inputs/transcripts.jsonl'
            resolved = resolve_command_args(arguments('prepare'), config)
            self.assertEqual(resolved.input, str((PROJECT_ROOT / 'inputs/transcripts.jsonl').resolve()))

    def test_absolute_config_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            config = load_config()
            config['input_path'] = str(Path(folder) / 'source.jsonl')
            config['prepared_path'] = str(Path(folder) / 'prepared')
            args = resolve_command_args(arguments('prepare'), config)
            self.assertEqual(args.input, str((Path(folder) / 'source.jsonl').resolve()))
            self.assertEqual(args.output, str((Path(folder) / 'prepared').resolve()))

    def test_explicit_adapter_and_split_take_precedence(self):
        config = load_config()
        config.update(adapter_path='older-run/adapter', evaluation_split='validation')
        args = resolve_command_args(arguments('evaluate'), config)
        self.assertEqual(args.adapter, str((PROJECT_ROOT / 'older-run/adapter').resolve()))
        self.assertEqual(args.split, 'validation')
        with tempfile.TemporaryDirectory() as folder, working_directory(folder):
            args = arguments('evaluate')
            args.adapter = 'override/adapter'
            args.split = 'test'
            args.data = 'override/data'
            resolve_command_args(args, config)
            self.assertEqual(args.adapter, str((Path(folder) / 'override/adapter').resolve()))
            self.assertEqual(args.data, str((Path(folder) / 'override/data').resolve()))
            self.assertEqual(config['adapter_path'], args.adapter)
            self.assertEqual(config['prepared_path'], args.data)
            self.assertEqual(config['evaluation_split'], 'test')

    def test_configured_review_and_resume_are_used(self):
        config = load_config()
        config['reviews_path'] = 'prepared/reviews.jsonl'
        config['resume_checkpoint'] = 'runs/prior/checkpoints/checkpoint-10'
        args = resolve_command_args(arguments('prepare'), config)
        self.assertEqual(args.reviews, str((PROJECT_ROOT / 'prepared/reviews.jsonl').resolve()))
        args = resolve_command_args(arguments('train'), config)
        self.assertEqual(args.resume, str((PROJECT_ROOT / 'runs/prior/checkpoints/checkpoint-10').resolve()))
        with tempfile.TemporaryDirectory() as folder, working_directory(folder):
            args = arguments('train')
            args.resume = 'other/checkpoint-20'
            resolve_command_args(args, config)
            self.assertEqual(args.resume, str((Path(folder) / 'other/checkpoint-20').resolve()))
            self.assertEqual(config['resume_checkpoint'], args.resume)

    def test_resume_comparison_ignores_only_workflow_fields(self):
        config = load_config()
        different_paths = copy.deepcopy(config)
        for key in PATH_KEYS:
            different_paths[key] = f'changed/{key}'
        different_paths['evaluation_split'] = 'validation'
        self.assertEqual(training_config(config), training_config(different_paths))
        for key, value in (('learning_rate', 0.0001), ('train_batch_size', 2)):
            changed_training = {**config, key: value}
            self.assertNotEqual(training_config(config), training_config(changed_training))

    def test_legacy_defaults_are_historical_not_current_user_values(self):
        original = load_config()
        legacy = {key: value for key, value in original.items()
                  if key not in WORKFLOW_KEYS and key not in LEGACY_TRAINER_SETTINGS}
        self.assertEqual(training_config(legacy), training_config(original))
        changed = {**original, **{key: value + 3 for key, value in LEGACY_TRAINER_SETTINGS.items()}}
        with tempfile.TemporaryDirectory() as folder:
            current_path = Path(folder) / 'current.json'
            current_path.write_text(json.dumps(changed), encoding='utf-8')
            with patch('clinician_training_v2.settings.DEFAULT_CONFIG', current_path):
                current = load_config()
                self.assertNotEqual(training_config(legacy), training_config(current))
                for key, historical in LEGACY_TRAINER_SETTINGS.items():
                    self.assertEqual(training_config(legacy)[key], historical)

    def test_workflow_path_and_split_validation(self):
        invalid = [(key, value) for key in PATH_KEYS for value in ('', ' ', 12, False, [])]
        invalid += [(key, None) for key in PATH_KEYS
                    if key not in ('adapter_path', 'reviews_path', 'resume_checkpoint')]
        invalid += [('evaluation_split', value) for value in ('train', '', None)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'overrides.json'
            for key, value in invalid:
                with self.subTest(key=key, value=value):
                    path.write_text(json.dumps({key: value}), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        load_config(path)
            for key in ('adapter_path', 'reviews_path', 'resume_checkpoint'):
                path.write_text(json.dumps({key: None}), encoding='utf-8')
                self.assertIsNone(load_config(path)[key])

    def test_new_trainer_controls_validate_positive_integers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'overrides.json'
            for key in LEGACY_TRAINER_SETTINGS:
                for value in (0, -1, True, 1.5, '2'):
                    with self.subTest(key=key, value=value):
                        path.write_text(json.dumps({key: value}), encoding='utf-8')
                        with self.assertRaises(ValueError):
                            load_config(path)
                path.write_text(json.dumps({key: 3}), encoding='utf-8')
                self.assertEqual(load_config(path)[key], 3)


if __name__ == '__main__':
    unittest.main()
