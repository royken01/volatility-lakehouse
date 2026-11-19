"""LSTM regressor for the event-volatility task.

A specific LSTM architecture is used to predict the volatility of a market event due to stacked news releases. 
The model takes as input a sequence of lagged volatilities and 
static features, in our case event type and features etc. and outputs a single predicted volatility value.

"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

from lakehouse.feature_columns import SEQUENCE_PREFIX


class TensorBatches:
    """Mini-batches cut straight from tensors.

    Much cheaper than a ``DataLoader`` for this tiny network.
    """

    def __init__(self, sequences, static_features, target=None, batch_size=16,
                 shuffle=False, generator=None):
        self.sequences = torch.tensor(np.asarray(sequences), dtype=torch.float32)
        self.static_features = torch.tensor(np.asarray(static_features), dtype=torch.float32)
        self.target = torch.tensor(
            np.zeros(len(self.sequences)) if target is None else np.asarray(target),
            dtype=torch.float32,
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return -(-len(self.target) // self.batch_size)

    def __iter__(self):
        row_count = len(self.target)
        order = (
            torch.randperm(row_count, generator=self.generator)
            if self.shuffle
            else torch.arange(row_count)
        )
        for first in range(0, row_count, self.batch_size):
            rows = order[first:first + self.batch_size]
            yield self.sequences[rows], self.static_features[rows], self.target[rows]


class PreReleaseVolatilityNetwork(nn.Module):
    """LSTM over the lagged volatilities (mean-pooled) + dense branch for static features -> one number."""

    def __init__(self, static_input_size=4, hidden_size=64, static_hidden=32,
                 head_hidden=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.has_static_branch = static_input_size > 0
        if self.has_static_branch:
            self.static_branch = nn.Sequential(
                nn.Linear(static_input_size, static_hidden), nn.ReLU(),
            )
        head_input = hidden_size + (static_hidden if self.has_static_branch else 0)
        self.head = nn.Sequential(
            nn.Linear(head_input, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, sequence, static_features):
        lstm_output, _ = self.lstm(sequence)
        parts = [lstm_output.mean(dim=1)]  # mean-pool over the time steps
        if self.has_static_branch:
            parts.append(self.static_branch(static_features))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


def predict_batches(network, batches):
    """Predictions and targets for every batch (any batch size)."""
    network.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for sequence, static_features, target in batches:
            predictions.append(
                network(sequence, static_features).cpu().numpy().ravel()
            )
            targets.append(target.cpu().numpy().ravel())
    return np.concatenate(predictions), np.concatenate(targets)


def train_network(network, train_batches, epochs=30, learning_rate=1e-3,
                  weight_decay=0.0, validation_batches=None, patience=None):
    """Adam + MSE training loop.
    """
    try:
        optimizer = torch.optim.Adam(
            network.parameters(), lr=learning_rate,
            weight_decay=weight_decay, fused=True,
        )
    except (RuntimeError, TypeError, ValueError):
        optimizer = torch.optim.Adam(
            network.parameters(), lr=learning_rate, weight_decay=weight_decay,
        )
    criterion = nn.MSELoss()
    history = []
    best_validation_loss = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        network.train()
        loss_sum, row_count = 0.0, 0
        for sequence, static_features, target in train_batches:
            optimizer.zero_grad()
            loss = criterion(network(sequence, static_features), target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(target)
            row_count += len(target)
        record = {"epoch": epoch + 1, "train_loss": loss_sum / max(row_count, 1)}

        if validation_batches is not None:
            predictions, targets = predict_batches(network, validation_batches)
            record["val_loss"] = float(np.mean((predictions - targets) ** 2))
            if record["val_loss"] < best_validation_loss - 1e-6:
                best_validation_loss = record["val_loss"]
                best_weights = copy.deepcopy(network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        history.append(record)
        if (
            patience is not None
            and validation_batches is not None
            and epochs_without_improvement >= patience
        ):
            break

    if best_weights is not None:
        network.load_state_dict(best_weights)
    return history


class LSTMRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn wrapper around PreReleaseVolatilityNetwork
    """

    def __init__(self, sequence_prefix=SEQUENCE_PREFIX, hidden_size=64,
                 static_hidden=32, head_hidden=64, dropout=0.0, epochs=30,
                 learning_rate=1e-3, weight_decay=0.0, batch_size=16,
                 val_fraction=0.1, patience=5, seed=123, num_threads=1):
        self.sequence_prefix = sequence_prefix
        self.hidden_size = hidden_size
        self.static_hidden = static_hidden
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.patience = patience
        self.seed = seed
        self.num_threads = num_threads

    def _split_columns(self, columns):
        pattern = re.compile(rf"^{re.escape(self.sequence_prefix)}(\d+)$")
        sequence_columns = sorted(
            (c for c in columns if pattern.match(c)),
            key=lambda c: -int(pattern.match(c).group(1)),
        )
        static_columns = [c for c in columns if c not in sequence_columns]
        return sequence_columns, static_columns

    def _standardised_arrays(self, features: pd.DataFrame):
        """(sequence array [events, steps, 1], static array) after imputing and standardising."""
        features = pd.DataFrame(features, columns=self.feature_names_in_)[self.feature_names_in_]
        features = (
            features
            .apply(pd.to_numeric, errors="coerce")
            .fillna(self.medians_)
            .fillna(0.0)
        )
        sequence = (
            (features[self.sequence_columns_].to_numpy(float) - self.sequence_mean_)
            / self.sequence_scale_
        )
        static = (
            (features[self.static_columns_].to_numpy(float) - self.static_means_)
            / self.static_scales_
        )
        return sequence[:, :, None], static

    def fit(self, features, target):
        previous_thread_count = torch.get_num_threads()
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        try:
            return self._fit(features, target)
        finally:
            torch.set_num_threads(previous_thread_count)

    def _fit(self, features, target):
        if not isinstance(features, pd.DataFrame):
            raise TypeError("LSTMRegressor needs a DataFrame with named columns")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        target = np.asarray(target, dtype=float)
        has_target = np.isfinite(target)  # events without a finite target are not trained on
        if not has_target.all():
            features = features.iloc[np.flatnonzero(has_target)]
            target = target[has_target]

        self.feature_names_in_ = list(features.columns)
        self.sequence_columns_, self.static_columns_ = self._split_columns(
            self.feature_names_in_
        )
        if not self.sequence_columns_:
            raise ValueError(
                f"No sequence columns '{self.sequence_prefix}<k>' in the features"
            )

        numeric = features.apply(pd.to_numeric, errors="coerce")
        self.medians_ = numeric.median()
        numeric = numeric.fillna(self.medians_).fillna(0.0)

        sequence_values = numeric[self.sequence_columns_].to_numpy(float)
        self.sequence_mean_ = float(sequence_values.mean())
        self.sequence_scale_ = float(sequence_values.std()) or 1.0

        static_values = numeric[self.static_columns_].to_numpy(float)
        self.static_means_ = (
            static_values.mean(axis=0) if static_values.size else np.zeros(0)
        )
        static_deviations = (
            static_values.std(axis=0) if static_values.size else np.zeros(0)
        )
        self.static_scales_ = np.where(static_deviations > 0, static_deviations, 1.0)

        self.target_mean_ = float(target.mean())
        self.target_scale_ = float(target.std()) or 1.0

        sequence, static = self._standardised_arrays(features)
        standardised_target = (target - self.target_mean_) / self.target_scale_

        validation_count = (
            int(len(standardised_target) * self.val_fraction) if self.patience else 0
        )
        validation_count = validation_count if validation_count >= 20 else 0
        train_count = len(standardised_target) - validation_count

        generator = torch.Generator().manual_seed(self.seed)
        train_batches = TensorBatches(
            sequence[:train_count], static[:train_count],
            standardised_target[:train_count],
            batch_size=self.batch_size, shuffle=True, generator=generator,
        )
        validation_batches = (
            TensorBatches(
                sequence[train_count:], static[train_count:],
                standardised_target[train_count:], batch_size=256,
            )
            if validation_count
            else None
        )

        self.network_ = PreReleaseVolatilityNetwork(
            static_input_size=len(self.static_columns_),
            hidden_size=self.hidden_size,
            static_hidden=self.static_hidden,
            head_hidden=self.head_hidden,
            dropout=self.dropout,
        )
        self.history_ = train_network(
            self.network_, train_batches,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            validation_batches=validation_batches,
            patience=self.patience if validation_count else None,
        )
        self.n_epochs_trained_ = len(self.history_)
        return self

    def predict(self, features):
        sequence, static = self._standardised_arrays(features)
        previous_thread_count = torch.get_num_threads()
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        try:
            predictions, _ = predict_batches(
                self.network_, TensorBatches(sequence, static, batch_size=256),
            )
        finally:
            torch.set_num_threads(previous_thread_count)
        return predictions * self.target_scale_ + self.target_mean_