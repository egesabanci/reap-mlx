"""Tests for the MLX calibration data loader."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from reap.data import (
    extract_text,
    load_calibration_sequences,
    tokenize_text,
)


def test_data_module_import_does_not_import_heavy_runtime_packages():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    code = textwrap.dedent(
        """
        import sys

        BLOCKED_ROOTS = ("torch", "vllm", "mlx", "mlx_lm", "datasets")

        def is_blocked(fullname):
            return any(
                fullname == root or fullname.startswith(root + ".")
                for root in BLOCKED_ROOTS
            )

        class ImportBlocker:
            def find_spec(self, fullname, path=None, target=None):
                if is_blocked(fullname):
                    raise AssertionError(
                        "forbidden import during MLX data import: "
                        f"{fullname}"
                    )
                return None

        sys.meta_path.insert(0, ImportBlocker())

        from reap.data import load_calibration_sequences

        assert load_calibration_sequences is not None

        forbidden_loaded = sorted(
            name for name in sys.modules if is_blocked(name)
        )
        if forbidden_loaded:
            raise AssertionError(
                "forbidden modules loaded during MLX data import: "
                + ", ".join(forbidden_loaded)
            )
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src_dir)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout


class WhitespaceTokenizer:
    chat_template = None

    def __init__(self):
        self.texts = []

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        self.texts.append(text)
        return [len(piece) for piece in text.split()]


class ChatTokenizer(WhitespaceTokenizer):
    chat_template = "template"

    def __init__(self):
        super().__init__()
        self.template_calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
    ):
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return " ".join(f"{message['role']}:{message['content']}" for message in messages)


class CallableTokenizer:
    def __call__(self, text):
        return {"input_ids": [[ord(char) for char in text]]}


def test_load_calibration_sequences_forwards_dataset_options_and_truncates():
    calls = []

    def fake_load_dataset(dataset_name, **kwargs):
        calls.append((dataset_name, kwargs))
        return [
            {"text": "alpha beta gamma"},
            {"text": "   "},
            {"text": "delta epsilon"},
        ]

    tokenizer = WhitespaceTokenizer()

    sequences = load_calibration_sequences(
        tokenizer,
        "example/dataset",
        split="validation",
        dataset_config_name="code",
        max_samples=2,
        max_seq_length=2,
        seed=123,
        load_dataset_fn=fake_load_dataset,
    )

    assert calls == [
        (
            "example/dataset",
            {"split": "validation", "name": "code"},
        )
    ]
    assert len(sequences) == 2
    assert sequences[0]["input_ids"].dtype == np.int32
    np.testing.assert_array_equal(sequences[0]["input_ids"], [5, 4])
    np.testing.assert_array_equal(sequences[1]["input_ids"], [5, 7])


def test_extract_text_supports_instruction_input_output_records():
    text = extract_text(
        {
            "instruction": "Write a function.",
            "input": "Use Python.",
            "output": "def f(): pass",
        }
    )

    assert text == "Write a function.\n\nUse Python.\n\ndef f(): pass"


def test_message_records_use_tokenizer_chat_template_when_available():
    tokenizer = ChatTokenizer()

    sequences = load_calibration_sequences(
        tokenizer,
        "chat/data",
        max_samples=1,
        max_seq_length=10,
        load_dataset_fn=lambda dataset_name, **kwargs: [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        ],
    )

    assert tokenizer.template_calls == [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "tokenize": False,
            "add_generation_prompt": False,
        }
    ]
    assert tokenizer.texts == ["user:Hello assistant:Hi"]
    np.testing.assert_array_equal(sequences[0]["input_ids"], [10, 12])


def test_conversation_records_fall_back_to_role_content_text():
    text = extract_text(
        {
            "conversations": [
                {"from": "human", "value": "Explain REAP."},
                {"from": "assistant", "value": "REAP prunes experts."},
            ]
        },
        tokenizer=WhitespaceTokenizer(),
    )

    assert text == "human: Explain REAP.\nassistant: REAP prunes experts."


class RecordingShuffleDataset(list):
    def __init__(self, records):
        super().__init__(records)
        self.seeds = []

    def shuffle(self, *, seed):
        self.seeds.append(seed)
        records = list(self)
        records.reverse()
        return records


def test_dataset_shuffle_is_used_with_seed_before_sampling():
    dataset = RecordingShuffleDataset(
        [
            {"text": "one"},
            {"text": "two"},
            {"text": "three"},
        ]
    )

    sequences = load_calibration_sequences(
        WhitespaceTokenizer(),
        "shuffle/data",
        max_samples=1,
        max_seq_length=10,
        seed=7,
        load_dataset_fn=lambda dataset_name, **kwargs: dataset,
    )

    assert dataset.seeds == [7]
    np.testing.assert_array_equal(sequences[0]["input_ids"], [5])


def test_callable_tokenizer_outputs_are_supported_and_flattened():
    token_ids = tokenize_text(
        CallableTokenizer(),
        "ab",
        max_seq_length=1,
    )

    assert token_ids == [97]
