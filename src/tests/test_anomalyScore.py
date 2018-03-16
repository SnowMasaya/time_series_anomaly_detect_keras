# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from unittest import TestCase
import pandas as pd
from pandas_datareader.data import DataReader
from datetime import datetime
import numpy as np
from data.time_series import time_step_data_convert
from anomaly_score.calculate_anomaly_score import AnomalyScore


class TestAnomalyScore(TestCase):
    def test_calculate_anomaly_score(self):
        a = DataReader('F', 'google',
                       datetime(2006, 6, 1),
                       datetime(2016, 6, 1))
        a_returns = pd.DataFrame(np.diff(np.log(a['Close'].values)))
        a_returns.index = a.index.values[1:a.index.values.shape[0]]
        train_value = a_returns.values
        scaler, _, _, prepare_value, check_value = time_step_data_convert(
            original_data=train_value,
            num_time_steps=5,
            batch_size=10,
        )
        model_name = '../models/lstm_batch_size_10_num_time_step_5_num_epochs_3_hidden_64_optimizer_adam.h5'  # noqa
        anomaly_score_instance = AnomalyScore(model_name=model_name)
        anomaly_score = anomaly_score_instance.calculate_anomaly_score(
            preprocess_data=prepare_value, answer_value=check_value)
        np.save('test_anomaly_score.npy', anomaly_score)
