from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_lstm_sequences(features: pd.DataFrame, target: pd.Series, sequence_length: int):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.to_numpy().reshape(-1, 1))

    xs = []
    ys = []
    for index in range(sequence_length, len(features_scaled)):
        xs.append(features_scaled[index - sequence_length:index])
        ys.append(target_scaled[index])
    return np.array(xs), np.array(ys), feature_scaler, target_scaler


def transform_lstm_sequences(
    features: pd.DataFrame,
    target: pd.Series,
    sequence_length: int,
    feature_scaler: MinMaxScaler,
    target_scaler: MinMaxScaler,
):
    features_scaled = feature_scaler.transform(features)
    target_scaled = target_scaler.transform(target.to_numpy().reshape(-1, 1))

    xs = []
    ys = []
    for index in range(sequence_length, len(features_scaled)):
        xs.append(features_scaled[index - sequence_length:index])
        ys.append(target_scaled[index])
    return np.array(xs), np.array(ys)


def build_inference_sequence(features: pd.DataFrame, feature_scaler: MinMaxScaler, sequence_length: int) -> np.ndarray:
    scaled = feature_scaler.transform(features.tail(sequence_length))
    return np.expand_dims(scaled, axis=0)
