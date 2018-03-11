# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from typing import Tuple


def time_step_data_convert(original_data: np.array,
                           num_time_steps: int,
                           batch_size: int,
                           train_size: float=0.9
                           ) -> Tuple[object, np.array, np.array, np.array, np.array]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_size = original_data.shape[0]
    X = np.zeros((original_data.shape[0], num_time_steps))
    Y = np.zeros((original_data.shape[0], 1))
    length_data_size = data_size - (data_size % batch_size)
    for i in range(length_data_size - num_time_steps - 1):
        X[i] = original_data[i:i + num_time_steps].T
        Y[i] = original_data[i + num_time_steps + 1]
    X = scaler.fit_transform(X)
    Y = scaler.fit_transform(Y)
    sp = int(train_size * data_size)
    Xtrain, Xtest, Ytrain, Ytest = X[0:sp], X[sp:], Y[0:sp], Y[sp:]
    train_size = (Xtrain.shape[0] // batch_size) * batch_size
    test_size = (Xtrain.shape[0] // batch_size) * batch_size
    Xtrain, Ytrain = Xtrain[0: train_size], Ytrain[0: train_size]
    Xtest, Ytest = Xtest[0: test_size], Ytest[0: test_size]
    return scaler, Xtrain, Ytrain, Xtest, Ytest


def multi_variate_time_series_convert(original_data: np.array,
                                      num_time_steps: int,
                                      batch_size: int,
                                      train_size: float=0.9
                                      ):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(original_data)
    variable_number = original_data.shape[0]
    data_size = original_data.shape[1]
    X = np.zeros((variable_number, data_size, num_time_steps))
    Y = np.zeros((variable_number, data_size))
    length_data_size = data_size - (data_size % batch_size)
    for i in range(variable_number):
        for j in range(length_data_size - num_time_steps - 1):
            X[i][j] = scaled_data[i, j:j + num_time_steps].T
            Y[i][j] = scaled_data[i, j + num_time_steps + 1]
    sp = int(train_size * data_size)
    Xtrain, Xtest, Ytrain, Ytest = X[:, 0:sp, :], X[:, sp:, :], \
                                   Y[:, 0:sp], Y[:, sp:]
    train_size = (Xtrain.shape[1] // batch_size) * batch_size
    test_size = (Xtest.shape[1] // batch_size) * batch_size
    Xtrain, Ytrain = Xtrain[:, 0:train_size], Ytrain[:, 0:train_size]
    Xtest, Ytest = Xtest[:, 0:test_size], Ytest[:, 0:test_size]
    Xtrain = np.swapaxes(Xtrain, 0, 1)
    Ytrain = np.swapaxes(Ytrain, 0, 1)
    Xtest = np.swapaxes(Xtest, 0, 1)
    Ytest = np.swapaxes(Ytest, 0, 1)
    return scaler, Xtrain, Xtest, Ytrain, Ytest


