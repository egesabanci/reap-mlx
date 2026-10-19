"""Tests for the MLX calibration data loader."""

from __future__ import annotations

import logging
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


def test_instruction_records_use_chat_template_when_available():
    tokenizer = ChatTokenizer()

    text = extract_text(
        {
            "instruction": "Write a function.",
            "input": "Use Python.",
            "output": "def f(): pass",
        },
        tokenizer=tokenizer,
    )

    assert tokenizer.template_calls == [
        {
            "messages": [
                {"role": "user", "content": "Write a function.\n\nUse Python."},
                {"role": "assistant", "content": "def f(): pass"},
            ],
            "tokenize": False,
            "add_generation_prompt": False,
        }
    ]
    assert text == "user:Write a function.\n\nUse Python. assistant:def f(): pass"


def test_instruction_records_without_a_prompt_stay_plain_text():
    tokenizer = ChatTokenizer()

    text = extract_text({"output": "def f(): pass"}, tokenizer=tokenizer)

    assert text == "def f(): pass"
    assert tokenizer.template_calls == []


class TemplatelessChatTokenizer(ChatTokenizer):
    """Has apply_chat_template but no template (Hugging Face base models)."""

    chat_template = None


def test_records_skip_apply_chat_template_when_tokenizer_has_no_template():
    tokenizer = TemplatelessChatTokenizer()

    message_text = extract_text(
        {"messages": [{"role": "user", "content": "Hello"}]},
        tokenizer=tokenizer,
    )
    instruction_text = extract_text(
        {"instruction": "Write a function.", "output": "def f(): pass"},
        tokenizer=tokenizer,
    )

    assert tokenizer.template_calls == []
    assert message_text == "user: Hello"
    assert instruction_text == "Write a function.\n\ndef f(): pass"


def test_mlx_lm_has_chat_template_flag_enables_templates():
    tokenizer = TemplatelessChatTokenizer()
    tokenizer.has_chat_template = True

    extract_text({"instruction": "Hi", "output": "Hello"}, tokenizer=tokenizer)

    assert len(tokenizer.template_calls) == 1


def test_pack_concatenates_samples_into_full_length_sequences():
    records = [
        {"text": "a bb ccc"},
        {"text": "dddd eeeee"},
        {"text": "f gg hhh iiii"},
        {"text": "a bb"},
    ]

    sequences = load_calibration_sequences(
        WhitespaceTokenizer(),
        "pack/data",
        max_samples=3,
        max_seq_length=4,
        pack=True,
        load_dataset_fn=lambda dataset_name, **kwargs: records,
    )

    # [1, 2, 3] + [4, 5] fills the first sequence (5 is dropped), [1, 2, 3, 4]
    # fills the second, and [1, 2] is kept as a shorter final sequence.
    assert [seq["input_ids"].tolist() for seq in sequences] == [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2],
    ]


def test_pack_stops_at_max_samples_packed_sequences():
    records = [{"text": "a bb ccc dddd"}] * 5

    sequences = load_calibration_sequences(
        WhitespaceTokenizer(),
        "pack/data",
        max_samples=2,
        max_seq_length=4,
        pack=True,
        load_dataset_fn=lambda dataset_name, **kwargs: records,
    )

    assert [seq["input_ids"].tolist() for seq in sequences] == [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]


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


def test_sharegpt_conversations_are_mapped_to_chat_template_roles():
    tokenizer = ChatTokenizer()

    text = extract_text(
        {
            "conversations": [
                {"from": "system", "value": "Be brief."},
                {"from": "human", "value": "Explain REAP."},
                {"from": "gpt", "value": "REAP prunes experts."},
            ]
        },
        tokenizer=tokenizer,
    )

    assert tokenizer.template_calls[0]["messages"] == [
        {"from": "system", "value": "Be brief.", "role": "system", "content": "Be brief."},
        {"from": "human", "value": "Explain REAP.", "role": "user", "content": "Explain REAP."},
        {
            "from": "gpt",
            "value": "REAP prunes experts.",
            "role": "assistant",
            "content": "REAP prunes experts.",
        },
    ]
    assert text == "system:Be brief. user:Explain REAP. assistant:REAP prunes experts."


def test_chat_template_receives_text_only_content_for_multimodal_messages():
    tokenizer = ChatTokenizer()

    extract_text(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image_url", "image_url": {"url": "x.png"}},
                    ],
                }
            ]
        },
        tokenizer=tokenizer,
    )

    assert tokenizer.template_calls[0]["messages"] == [
        {"role": "user", "content": "Describe this."}
    ]


class RejectingChatTokenizer(ChatTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        raise ValueError("Conversation roles must alternate user/assistant")


def test_chat_template_errors_fall_back_to_role_content_text(caplog):
    with caplog.at_level(logging.WARNING, logger="reap.data"):
        text = extract_text(
            {
                "messages": [
                    {"role": "user", "content": "One"},
                    {"role": "user", "content": "Two"},
                ]
            },
            tokenizer=RejectingChatTokenizer(),
        )

    assert text == "user: One\nuser: Two"
    assert any("Chat template failed" in rec.message for rec in caplog.records)


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


def test_extract_text_skips_multimodal_content_blocks():
    text = extract_text(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        },
                        {
                            "type": "audio",
                            "audio": {"url": "https://example.com/a.wav"},
                        },
                        {
                            "type": "video",
                            "video": {"url": "https://example.com/v.mp4"},
                        },
                        {"type": "text", "text": "Use one sentence."},
                    ],
                }
            ]
        },
        tokenizer=WhitespaceTokenizer(),
    )

    assert text == "user: Describe this.\nUse one sentence."


def test_load_calibration_sequences_tokenizes_only_text_content_blocks():
    tokenizer = WhitespaceTokenizer()

    sequences = load_calibration_sequences(
        tokenizer,
        "multimodal/data",
        max_samples=1,
        max_seq_length=10,
        load_dataset_fn=lambda dataset_name, **kwargs: [
            {
                "text": [
                    {"type": "text", "text": "Hello world"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                    {"type": "text", "text": "Goodbye"},
                ]
            }
        ],
    )

    assert tokenizer.texts == ["Hello world\nGoodbye"]
    np.testing.assert_array_equal(sequences[0]["input_ids"], [5, 5, 7])


def test_extract_text_preserves_untyped_mapping_content_fallback():
    text = extract_text(
        {
            "text": [
                "metadata",
                {"source": "manual", "value": 3},
            ]
        }
    )

    assert text == 'metadata\n{"source": "manual", "value": 3}'


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


def test_dataset_without_shuffle_uses_sequential_order(caplog):
    dataset = [{"text": "a"}, {"text": "b"}]  # plain list, no .shuffle
    with caplog.at_level(logging.INFO, logger="reap.data"):
        seqs = load_calibration_sequences(
            WhitespaceTokenizer(),
            "no-shuffle/data",
            max_samples=2,
            max_seq_length=10,
            seed=3,
            load_dataset_fn=lambda dataset_name, **kwargs: dataset,
        )
    assert len(seqs) == 2
    # Sequential (un-shuffled) order preserved + a log explaining why.
    assert any("does not support shuffle" in rec.message for rec in caplog.records)


def test_dataset_shuffle_failure_falls_back_to_sequential(caplog):
    class BrokenShuffleDataset(list):
        def shuffle(self, *, seed):
            raise RuntimeError("shuffle not supported for streaming")

    dataset = BrokenShuffleDataset([{"text": "x"}, {"text": "y"}])
    with caplog.at_level(logging.WARNING, logger="reap.data"):
        seqs = load_calibration_sequences(
            WhitespaceTokenizer(),
            "broken-shuffle/data",
            max_samples=2,
            max_seq_length=10,
            seed=3,
            load_dataset_fn=lambda dataset_name, **kwargs: dataset,
        )
    assert len(seqs) == 2
    assert any("shuffle failed" in rec.message for rec in caplog.records)


def test_callable_tokenizer_outputs_are_supported_and_flattened():
    token_ids = tokenize_text(
        CallableTokenizer(),
        "ab",
        max_seq_length=1,
    )

    assert token_ids == [97]
