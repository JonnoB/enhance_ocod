"""
Dump model predictions to JSON for analysis in absurd_data.

Outputs one record per address with:
  - gold_spans: char-level spans from ground truth (with extracted text)
  - pred_spans: char-level spans reconstructed from model predictions
  - gold_entities / pred_entities: token-level entity dicts for nervaluate

Usage:
  python dump_predictions.py --output preds.json
  python dump_predictions.py --model path/to/model --data path/to/test.json --output preds.json
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics.sequence_labeling import get_entities

from enhance_ocod.training import NERDataProcessor

SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_MODEL = str(SCRIPT_DIR / ".." / "models" / "address_parser_dev" / "final_model")
DEFAULT_DATA = str(
    SCRIPT_DIR
    / ".."
    / "data"
    / "training_data"
    / "ner_ready"
    / "ground_truth_test_set_labels.json"
)
MAX_LENGTH = 128


def bio_to_char_spans(text, bio_seq, tokenizer):
    """Convert a word-level BIO sequence to char-level entity spans.

    bio_seq contains one label per word (continuation subword tokens are masked
    with -100 during label alignment and excluded from y_pred/y_true).
    get_entities therefore returns word-level indices, which we map to character
    positions by aggregating all subword token offsets for each word.
    """
    entities = get_entities(bio_seq)
    if not entities:
        return []

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]
    word_ids = encoding.word_ids()

    # Build word_index → [first_subword_char_start, last_subword_char_end]
    word_to_char = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        char_start, char_end = offsets[token_idx]
        if word_id not in word_to_char:
            word_to_char[word_id] = [char_start, char_end]
        else:
            word_to_char[word_id][1] = char_end  # extend through all subwords of word

    spans = []
    for label, word_start, word_end in entities:
        if word_start in word_to_char and word_end in word_to_char:
            char_start = word_to_char[word_start][0]
            char_end = word_to_char[word_end][1]
            spans.append(
                {
                    "label": label,
                    "start": char_start,
                    "end": char_end,
                    "text": text[char_start:char_end],
                }
            )
    return spans


def main():
    parser = argparse.ArgumentParser(description="Dump NER model predictions to JSON.")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to trained model")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to test data JSON")
    args = parser.parse_args()

    print(f"Loading model from {args.model}")
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    id2label = model.config.id2label

    label_list = list(id2label.values())
    processor = NERDataProcessor(label_list, tokenizer.name_or_path)
    processor.tokenizer = tokenizer

    print(f"Loading data from {args.data}")
    val_data = processor.load_json_data(args.data)
    val_dataset = processor.create_dataset(val_data, MAX_LENGTH)
    print(f"Loaded {len(val_dataset)} examples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    records = []
    with torch.no_grad():
        for i, example in enumerate(val_dataset):
            input_ids = torch.tensor([example["input_ids"]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"]]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

            y_true, y_pred = [], []
            for pred_id, true_id in zip(preds, example["labels"]):
                if true_id != -100:
                    y_true.append(id2label[true_id])
                    y_pred.append(id2label[pred_id])

            text = val_data[i]["text"]

            gold_spans = [
                {
                    "label": s["label"],
                    "start": s["start"],
                    "end": s["end"],
                    "text": text[s["start"] : s["end"]],
                }
                for s in val_data[i]["spans"]
            ]
            pred_spans = bio_to_char_spans(text, y_pred, tokenizer)

            gold_entities = [
                {"label": label, "start": start, "end": end}
                for label, start, end in get_entities(y_true)
            ]
            pred_entities = [
                {"label": label, "start": start, "end": end}
                for label, start, end in get_entities(y_pred)
            ]

            records.append(
                {
                    "text": text,
                    "gold_spans": gold_spans,
                    "pred_spans": pred_spans,
                    "gold_entities": gold_entities,
                    "pred_entities": pred_entities,
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
