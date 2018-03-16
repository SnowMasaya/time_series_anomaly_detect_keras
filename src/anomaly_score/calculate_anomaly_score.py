# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np
from keras.models import load_model
import re
from os import path


def calculate_mse(predict_values: np.array,
                  answer_value :np.array):
    mse_check_values = np.zeros(predict_values.shape[0])
    for index, (value, predict_value) in enumerate(zip(predict_values, answer_value)):
         mse_check_values[index] = (value - predict_value) ** 2
    return mse_check_values


def mse_to_anomaly_score(mse_check_values: np.array, var: np.array) -> np.array:
    anomaly_score = np.zeros(mse_check_values.shape)
    for index, each_mse in enumerate(mse_check_values):
        anomaly_score[index] = (1.0 / var) * each_mse  # noqa
    return anomaly_score


class AnomalyScore(object):

    def __init__(self, model_name :str):
        self.load_model = load_model(model_name)
        m = re.match(
            r"(?P<model_name>[a-z]+)_(?P<batch_size>[a-z]+_[a-z]+_[0-9]+)_(?P<window_size>[a-z]+_[a-z]+_[a-z]+_[0-9]+)",  # noqa
            path.basename(model_name))
        batch_size = m.group('batch_size')
        extract_batch_size = \
            re.match(r"(?P<batch_name>[a-z]+_[a-z]+)_(?P<batch_size>[0-9]+)",
                     batch_size)
        self.model_batch_size = int(extract_batch_size.group('batch_size'))

    def calculate_anomaly_score(self,
                                preprocess_data: np.array,
                                answer_value :np.array):
        print('batch_size {}'.format(self.model_batch_size))
        predicted = self.load_model.predict(preprocess_data,
                                            batch_size=self.model_batch_size,)
        print('predict size {} original size {}'.format(predicted.shape,
                                                        answer_value.shape))
        np.save('test_predict.npy', predicted)
        np.save('test_answer.npy', answer_value)
        mse_check_value = \
            calculate_mse(predict_values=predicted, answer_value=answer_value)
        return mse_check_value