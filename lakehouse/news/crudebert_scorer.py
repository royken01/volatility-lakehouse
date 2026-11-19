"""CrudeBERT loader and scorer, shared by the silver GDELT scoring step and ``score_headlines.py``.

CrudeBERT is a FinBERT-TRC2 checkpoint fine-tuned on crude-oil text; it ships a config and a state
dict but no tokenizer, so the ``bert-base-uncased`` tokenizer is used.  Two bugs of the original
scripts stay fixed:

  1. The weights were never loaded (``from_config`` allocates RANDOM weights); the state dict is
     loaded here.
  2. The label order was assumed; the index of each label is read from the config's ``label2id``.

``sentiment_score = P(positive) - P(negative)``; ``sentiment_ternary`` is +1 / 0 / -1 around
+-``ternary_threshold``::

    scorer = CrudeBertScorer()
    score, ternary, probabilities = scorer.score("OPEC+ agrees to deepen output cuts")
    results = scorer.score_batch(list_of_headlines)

Needs ``torch`` and ``transformers``.  ``python -m lakehouse.news.crudebert_scorer`` is a sanity
check: probabilities near [0.33, 0.33, 0.33] for every headline mean the weights are not loaded.
"""
import os

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from lakehouse.paths import CRUDEBERT_MODEL_DIR

# Older checkpoints saved ``position_ids`` as a persistent buffer, newer transformers do not: it shows up as
# "unexpected" on load although no trained weight is missing.  Safe to ignore.
BENIGN_UNEXPECTED_KEYS = {"bert.embeddings.position_ids"}


class CrudeBertScorer:
    def __init__(self, model_dir=CRUDEBERT_MODEL_DIR, device: str | None = None, ternary_threshold: float = 0.25):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ternary_threshold = ternary_threshold

        config_path = os.path.join(model_dir, "crude_bert_config.json")
        weights_path = os.path.join(model_dir, "crude_bert_model.bin")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError(f"Expected crude_bert_config.json and crude_bert_model.bin in {model_dir}")

        self.config = AutoConfig.from_pretrained(config_path)
        self.model = AutoModelForSequenceClassification.from_config(self.config)

        state_dict = torch.load(weights_path, map_location=self.device)
        is_wrapped_checkpoint = (isinstance(state_dict, dict) and "state_dict" in state_dict
                                 and not any(key.startswith(("bert.", "classifier.")) for key in state_dict))
        if is_wrapped_checkpoint:   # checkpoint saved as {"state_dict": ..., "optimizer": ...}
            state_dict = state_dict["state_dict"]

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        unexpected_keys = [key for key in unexpected_keys if key not in BENIGN_UNEXPECTED_KEYS]
        if missing_keys or unexpected_keys:
            print(f"[crudebert_scorer] WARNING: weights only partly loaded - {len(missing_keys)} missing, "
                  f"{len(unexpected_keys)} unexpected key(s). Investigate before trusting scores.")
            print("  missing (first 10):", missing_keys[:10], " unexpected (first 10):", unexpected_keys[:10])
        else:
            print("[crudebert_scorer] crude_bert_model.bin loaded cleanly.")

        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        label_to_index = {str(label).lower(): index for label, index in self.config.label2id.items()}
        if not {"positive", "negative", "neutral"}.issubset(label_to_index):
            raise ValueError(f"config.label2id is missing expected labels: {label_to_index}")
        self.positive_index = label_to_index["positive"]
        self.negative_index = label_to_index["negative"]
        self.neutral_index = label_to_index["neutral"]

    def _to_score(self, probabilities):
        """Class probabilities -> (score, ternary label, {"positive", "negative", "neutral"} probabilities)."""
        positive = probabilities[self.positive_index]
        negative = probabilities[self.negative_index]
        neutral = probabilities[self.neutral_index]
        sentiment_score = positive - negative
        if sentiment_score > self.ternary_threshold:
            ternary = 1
        elif sentiment_score < -self.ternary_threshold:
            ternary = -1
        else:
            ternary = 0
        return sentiment_score, ternary, {"positive": positive, "negative": negative, "neutral": neutral}

    @torch.no_grad()
    def score(self, text: str):
        """Score one headline."""
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
        probabilities = torch.softmax(self.model(**tokens).logits, dim=-1)[0].tolist()
        return self._to_score(probabilities)

    @torch.no_grad()
    def score_batch(self, texts, batch_size: int = 32):
        """Score many headlines in padded batches; returns one ``(score, ternary, probabilities)`` per text."""
        results = []
        for first in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[first:first + batch_size], return_tensors="pt", truncation=True,
                                    padding=True, max_length=512)
            tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
            probabilities_batch = torch.softmax(self.model(**tokens).logits, dim=-1).tolist()
            results.extend(self._to_score(probabilities) for probabilities in probabilities_batch)
        return results


if __name__ == "__main__":
    scorer = CrudeBertScorer()
    sample_headlines = [
        "OPEC+ agrees to deepen production cuts amid oversupply concerns",
        "Crude oil prices surge as Middle East conflict threatens supply",
        "US crude inventories fall sharply, tightening market",
        "Oil prices tumble on weak demand and record US stockpiles",
        "OPEC monthly report shows no material change in market balance",
    ]
    print("\nternary  score   probabilities                                    | headline")
    for headline in sample_headlines:
        score, ternary, probabilities = scorer.score(headline)
        print(f"{ternary:+d}       {score:+.3f}  pos={probabilities['positive']:.3f} "
              f"neg={probabilities['negative']:.3f} neu={probabilities['neutral']:.3f}  | {headline}")
