import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime

def load_data(data_name):
    path = os.path.join('data', data_name)
    dataset = pd.read_csv(path)
    dataset = dataset.replace('?', np.nan)
    dataset.iloc[:, 2:] = dataset.iloc[:, 2:].astype(float)
    dataset = dataset[(dataset.iloc[:, 2:] != 0).all(axis=1) & dataset.notna().all(axis=1)]
    data = dataset.values
    # encoder = OneHotEncoder()
    # one_hot = encoder.fit_transform(np.expand_dims(data[:, -4], axis=1)).toarray()
    # data = np.delete(data, -4, axis=1)
    # data = np.concatenate((data, one_hot), axis=1)
    # data = data.astype('float32')
    date_column = data[:, 0]
    time_column = data[:, 1]
    float_times = []
    float_dates = []
    for date_str, time_str in zip(date_column, time_column):
        time = datetime.datetime.strptime(time_str, "%H:%M:%S")
        date = datetime.datetime.strptime(date_str, "%d/%m/%Y")
        float_time = (time.hour * 3600 + time.minute * 60 + time.second) / 86400.0
        float_date = date.timestamp()
        float_times.append(float_time)
        float_dates.append(float_date)
    data[:, 0] = np.array(float_dates)
    data[:, 1] = np.array(float_times)
    data = data.astype('float32')
    real_y = data[:, 2]
    data = np.delete(data, 2, axis=1)
    real_y = real_y.reshape(-1, 1)
    data = np.hstack((real_y, data))
    scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(data[:, 1:])
    scaled_y = y_scaler.fit_transform(np.expand_dims(data[:, 0], axis=1))
    scaled_data = np.concatenate((scaled_y, scaled_X), axis=1)
    return scaled_data, y_scaler


def data_to_series_features(data, time_steps):
    data_size = len(data) - time_steps
    series_X = []
    series_y = []
    for i in range(data_size):
        series_X.append(data[i:i + time_steps])
        series_y.append(data[i + time_steps, 0])
    series_X = np.array(series_X)
    series_y = np.array(series_y)
    return series_X, series_y


def is_minimum(value, indiv_to_rmse):
    if len(indiv_to_rmse) == 0:
        return True
    temp = list(indiv_to_rmse.values())
    return True if value < min(temp) else False


def apply_weight(series_X, weight):
    weight = np.array(weight)
    weighted_series_X = series_X * np.expand_dims(weight, axis=1)
    return weighted_series_X
