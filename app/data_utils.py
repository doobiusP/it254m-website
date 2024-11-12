import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import typing

def add_moving_average_features(df : pd.DataFrame, window_sizes=[5, 10, 20]) -> pd.DataFrame:
    for window in window_sizes:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    return df

def normalize_data(data : pd.DataFrame) -> typing.Tuple[MinMaxScaler, np.ndarray]:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaler, scaled_data

def prepare_sequences(data : pd.DataFrame, sequence_length : int) -> typing.Tuple[np.ndarray, np.ndarray]:
    x_data, y_data = [], []
    for i in range(sequence_length, len(data)):
        x_data.append(data[i - sequence_length:i, :-1])
        y_data.append(data[i, -1])
    return np.array(x_data), np.array(y_data)
