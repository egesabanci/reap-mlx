# Calibration

Calibration data loading is implemented in `reap.data`. It converts a Hugging
Face dataset split into unpadded token sequences for batch-size-1 observer
replay.

## Loader Contract

```python
load_calibration_sequences(
    tokenizer,
    dataset_name,
    *,
    split="train",
    dataset_config_name=None,
    max_samples=128,
    max_seq_length=2048,
    seed=42,
    load_dataset_fn=None,
) -> list[dict[str, np.ndarray]]
```

Returns:

```python
[
    {"input_ids": np.ndarray(dtype=np.int32, shape=(seq_len,))},
    ...
]
```

Only non-empty tokenized samples are returned. Loading stops when
`max_samples` non-empty sequences have been collected or the dataset is
exhausted.

## Dataset Loading

When `load_dataset_fn` is not provided, the loader lazily imports:

```python
from datasets import load_dataset
```

It calls:

```python
load_dataset(dataset_name, split=split)
```

When `dataset_config_name` is provided, it passes:

```python
load_dataset(dataset_name, name=dataset_config_name, split=split)
```

If the returned dataset object exposes `.shuffle(seed=...)`, calibration uses
that method before sampling.

## Text Extraction Priority

`extract_text(record, tokenizer=tokenizer)` supports common dataset shapes.

For mapping records, it checks in this order:

1. `messages`
2. `conversations`
3. `text`
4. instruction-style fields:
   - instruction side: `instruction`, `prompt`, `question`
   - optional input side: `input`, `context`
   - output side: `output`, `completion`, `response`, `answer`

Non-mapping records are converted to text through normalization.

Instruction-style records are rendered as a conversation through the chat
template when the tokenizer has one, matching the REAP reference (which formats
`theblackcat102/evol-codealpaca-v1` the same way):

```python
[
    {"role": "user", "content": "<instruction>\n\n<input>"},
    {"role": "assistant", "content": "<output>"},
]
```

Without a chat template, or when there is no instruction-side text, the parts
are joined with blank lines instead.

## Chat And Conversation Records

For `messages` or `conversations`, JSON strings are parsed when possible.

When the tokenizer has a chat template (mlx-lm's `has_chat_template`, or a
non-empty Hugging Face `chat_template`), the loader calls `apply_chat_template`
with:

```python
tokenize=False
add_generation_prompt=False
```

If the tokenizer's chat template signature does not accept
`add_generation_prompt`, the loader retries with `tokenize=False` only.

Before rendering, turns are converted to the `role`/`content` shape chat
templates expect: ShareGPT turns such as `{"from": "human", "value": ...}`
become `{"role": "user", "content": ...}` (`gpt`/`bot`/`model` map to
`assistant`), and multimodal content lists are reduced to their text parts.
If the template still rejects a record (for example strict role alternation),
the loader logs a warning and uses the plain text rendering below for that
record.

Without a chat template, each message is rendered as:

```txt
role: content
```

The role keys are checked as `role`, `from`, then `speaker`. Content keys are
checked as `content`, `value`, then `text`.

## Content Normalization

Content normalization handles:

- strings as-is;
- lists by joining normalized items with newlines;
- multimodal-style text items with `{"type": "text", "text": ...}`;
- other mappings as sorted JSON;
- all other values through `str(value)`.

Empty or whitespace-only text is skipped after tokenization.

## Tokenization

`tokenize_text` first strips the text. Empty text returns an empty list.

Tokenizer priority:

1. Use `tokenizer.encode(text, add_special_tokens=True)`.
2. If that signature fails, retry `tokenizer.encode(text)`.
3. If `encode` is unavailable and the tokenizer is callable, call
   `tokenizer(text)` and read `input_ids`.

Callable tokenizer output can be a mapping with `input_ids` or an object with an
`input_ids` attribute.

Token IDs are flattened when they are returned as a batch-like nested sequence,
then truncated to `max_seq_length` and cast to Python `int`.

## Packing

By default every non-empty sample becomes one calibration sequence, so short
samples produce short sequences. With `pack=True` (CLI: `--pack-calibration`)
consecutive samples are concatenated into sequences of exactly
`max_seq_length` tokens, as the REAP reference implementation does. A sample
that does not fit the remaining room is truncated, `max_samples` counts packed
sequences, and the last sequence may be shorter when the dataset runs out.
Packing raises the number of observed tokens per sequence, which gives more
stable saliency estimates at a higher observation cost.

## Observer Shape Rules

The observer accepts:

- `[seq]`
- `[1, seq]`

It rejects:

- scalar input;
- empty sequences;
- batches where the first dimension is not 1;
- tensors with more than two dimensions.

The calibration loader returns `[seq]`; observer replay adds the batch dimension.

## Practical Guidance

For smoke runs, small values are enough:

```bash
--max-samples 8 --max-seq-length 1024
```

For more stable saliency estimates, increase both sample count and sequence
length, then compare `validation-metrics.json` across runs. Observation time
scales with token count and number of layers replayed.

