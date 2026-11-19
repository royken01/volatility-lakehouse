"""Headline sentiment scorers behind one interface.

``[news] sentiment_model`` (or ``--sentiment-model``) picks the scorer:

    crudebert             the local CrudeBERT weights (crude-oil domain, see ``crudebert_scorer.py``)
    <hugging face id>     any sequence-classification model with positive / negative / neutral labels,
                          e.g. ``ProsusAI/finbert`` (general finance) or a domain model of your own

Every scorer returns, per headline, ``(score, ternary, {"positive", "negative", "neutral"})`` with
``score = P(positive) - P(negative)``.  Needs ``torch`` and ``transformers`` (the ``[news]`` extra).
"""
from __future__ import annotations

from lakehouse.settings import get_settings


def ternary_label(score: float, threshold: float) -> int:
    return 1 if score > threshold else -1 if score < -threshold else 0


class HuggingFaceSentimentScorer:
    """A Hugging Face sentiment classifier; labels are matched by name from the model config."""

    LABEL_ALIASES = {"positive": ("positive", "pos", "bullish"), "negative": ("negative", "neg", "bearish"),
                     "neutral": ("neutral", "neu")}

    def __init__(self, model_id: str, device: str | None = None, ternary_threshold: float = 0.25):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device).eval()
        self.ternary_threshold = ternary_threshold
        labels = {str(label).lower(): index for index, label in self.model.config.id2label.items()}
        self.index = {}
        for name, aliases in self.LABEL_ALIASES.items():
            found = next((labels[a] for a in aliases if a in labels), None)
            if found is None and name != "neutral":
                raise ValueError(f"{model_id}: no {name!r} label in {sorted(labels)}")
            self.index[name] = found

    def _to_score(self, probabilities):
        values = {name: (probabilities[i] if i is not None else 0.0) for name, i in self.index.items()}
        score = values["positive"] - values["negative"]
        return score, ternary_label(score, self.ternary_threshold), values

    def score_batch(self, texts, batch_size: int = 32):
        results = []
        with self.torch.no_grad():
            for first in range(0, len(texts), batch_size):
                tokens = self.tokenizer(texts[first:first + batch_size], return_tensors="pt", truncation=True,
                                        padding=True, max_length=512)
                tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
                batch = self.torch.softmax(self.model(**tokens).logits, dim=-1).tolist()
                results.extend(self._to_score(probabilities) for probabilities in batch)
        return results

    def score(self, text: str):
        return self.score_batch([text])[0]


def get_scorer(model: str | None = None):
    """The scorer named by ``model`` (default: the settings' ``sentiment_model``)."""
    model = model or get_settings().sentiment_model
    if model.lower() == "crudebert":
        from lakehouse.news.crudebert_scorer import CrudeBertScorer

        return CrudeBertScorer()
    return HuggingFaceSentimentScorer(model)
