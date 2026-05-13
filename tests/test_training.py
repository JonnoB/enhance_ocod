"""
Tests for NERDataProcessor.align_labels_with_tokens.

Uses a lightweight mock encoding so no model download is required.
Each test case verifies the one-label-per-word rule: only the first subword of
each word receives a real label; continuation tokens and special tokens get -100.
"""

import pytest
from unittest.mock import MagicMock
from enhance_ocod.training import NERDataProcessor, create_label_list

ENTITY_TYPES = ["street_name", "street_number", "city"]
LABEL_LIST = create_label_list(ENTITY_TYPES)


def make_processor():
    processor = NERDataProcessor.__new__(NERDataProcessor)
    processor.label_list = LABEL_LIST
    processor.label2id = {label: i for i, label in enumerate(LABEL_LIST)}
    processor.id2label = {i: label for label, i in processor.label2id.items()}
    return processor


def make_encoding(word_ids, token_to_chars_map, n_tokens=None):
    """
    Build a mock tokenizer encoding.

    word_ids: list of word IDs per token position (None for special tokens)
    token_to_chars_map: dict of token_idx → (char_start, char_end)
    """
    if n_tokens is None:
        n_tokens = len(word_ids)

    enc = MagicMock()
    enc.__getitem__ = lambda self, key: [0] * n_tokens if key == "input_ids" else None
    enc.word_ids = MagicMock(return_value=word_ids)

    def token_to_chars(idx):
        if idx in token_to_chars_map:
            start, end = token_to_chars_map[idx]
            result = MagicMock()
            result.start = start
            result.end = end
            return result
        return None

    enc.token_to_chars = token_to_chars
    return enc


class TestAlignLabelsWithTokens:
    """Verify one-label-per-word rule after the subword alignment fix."""

    def test_special_tokens_always_minus100(self):
        # [CLS]=None, word0=0, [SEP]=None  — no entity
        enc = make_encoding(
            word_ids=[None, 0, None],
            token_to_chars_map={1: (0, 4)},
        )
        processor = make_processor()
        labels = processor.align_labels_with_tokens("road", [], enc)

        assert labels[0] == -100, "[CLS] must be -100"
        assert labels[2] == -100, "[SEP] must be -100"

    def test_non_entity_word_gets_O(self):
        # Single word, no spans
        enc = make_encoding(
            word_ids=[None, 0, None],
            token_to_chars_map={1: (0, 4)},
        )
        processor = make_processor()
        labels = processor.align_labels_with_tokens("road", [], enc)

        assert labels[1] == processor.label2id["O"]

    def test_single_token_entity_gets_B(self):
        # "Road" is one token and is a street_name entity
        enc = make_encoding(
            word_ids=[None, 0, None],
            token_to_chars_map={1: (0, 4)},
        )
        processor = make_processor()
        spans = [{"start": 0, "end": 4, "label": "street_name"}]
        labels = processor.align_labels_with_tokens("Road", spans, enc)

        assert labels[1] == processor.label2id["B-street_name"]

    def test_multi_subword_entity_continuation_is_minus100(self):
        # "ranelagh" splits into tokens 1 ("ran") and 2 ("##elagh"), word_id=0 for both
        # Token 1 should get B-, token 2 must get -100 (not I-)
        enc = make_encoding(
            word_ids=[None, 0, 0, None],
            token_to_chars_map={1: (0, 3), 2: (3, 8)},
        )
        processor = make_processor()
        spans = [{"start": 0, "end": 8, "label": "street_name"}]
        labels = processor.align_labels_with_tokens("ranelagh", spans, enc)

        assert labels[1] == processor.label2id["B-street_name"], "first subword → B-"
        assert labels[2] == -100, "continuation subword → -100, not I-"

    def test_multi_word_entity_uses_B_then_I(self):
        # "33 high street" — word 0 = "33" (street_number), words 1-2 = "high street" (street_name)
        # Tokens: [CLS]=None, "33"=w0, "high"=w1, "street"=w2, [SEP]=None
        enc = make_encoding(
            word_ids=[None, 0, 1, 2, None],
            token_to_chars_map={1: (0, 2), 2: (3, 7), 3: (8, 14)},
        )
        processor = make_processor()
        spans = [
            {"start": 0, "end": 2, "label": "street_number"},
            {"start": 3, "end": 14, "label": "street_name"},
        ]
        labels = processor.align_labels_with_tokens("33 high street", spans, enc)

        assert labels[1] == processor.label2id["B-street_number"]
        assert labels[2] == processor.label2id["B-street_name"]
        assert labels[3] == processor.label2id["I-street_name"]

    def test_multi_word_entity_with_subword_split_continuation_is_minus100(self):
        # "high street" where "street" → ["str", "##eet"] (word_id=1 for both subwords)
        # Tokens: [CLS], "high"(w0), "str"(w1), "##eet"(w1), [SEP]
        enc = make_encoding(
            word_ids=[None, 0, 1, 1, None],
            token_to_chars_map={1: (0, 4), 2: (5, 8), 3: (8, 11)},
        )
        processor = make_processor()
        spans = [{"start": 0, "end": 11, "label": "street_name"}]
        labels = processor.align_labels_with_tokens("high street", spans, enc)

        assert labels[1] == processor.label2id["B-street_name"]
        assert labels[2] == processor.label2id["I-street_name"]
        assert labels[3] == -100, "continuation of 'street' must be -100"

    def test_adjacent_entities_both_start_with_B(self):
        # "London Road" — "London"=city, "Road"=street_name, back to back
        enc = make_encoding(
            word_ids=[None, 0, 1, None],
            token_to_chars_map={1: (0, 6), 2: (7, 11)},
        )
        processor = make_processor()
        spans = [
            {"start": 0, "end": 6, "label": "city"},
            {"start": 7, "end": 11, "label": "street_name"},
        ]
        labels = processor.align_labels_with_tokens("London Road", spans, enc)

        assert labels[1] == processor.label2id["B-city"]
        assert labels[2] == processor.label2id["B-street_name"], \
            "new entity after adjacent one must start with B-, not I-"

    def test_label_count_equals_token_count(self):
        # Sanity: output length always matches number of tokens
        enc = make_encoding(
            word_ids=[None, 0, 1, 1, 2, None],
            token_to_chars_map={1: (0, 2), 2: (3, 6), 3: (6, 9), 4: (10, 16)},
        )
        processor = make_processor()
        spans = [{"start": 0, "end": 9, "label": "street_name"}]
        labels = processor.align_labels_with_tokens("33 ran##elagh London", spans, enc)

        assert len(labels) == 6
