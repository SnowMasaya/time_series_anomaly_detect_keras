# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.time_series import time_step_data_convert
import pandas as pd
from pandas_datareader.data import DataReader
import numpy as np
from datetime import datetime
import filecmp
from models.train_model import TrainModel
from models.lstm_model import LSTMModel
from visualization.visualize import visualize_wave


class TestTrainModel(TestCase):
    def test_train(self):
        a = DataReader('F', 'google',
                       datetime(2006, 6, 1),
                       datetime(2016, 6, 1))
        a_returns = pd.DataFrame(np.diff(np.log(a['Close'].values)))
        a_returns.index = a.index.values[1:a.index.values.shape[0]]
        train_value = a_returns.values
        scaler, x_train, y_train, x_test, y_test = time_step_data_convert(
            original_data=train_value,
            num_time_steps=5,
            batch_size=10,
        )
        lstm_instance = LSTMModel(num_time_steps=5)
        train_model_instance = TrainModel(model_object=lstm_instance,
                                          model_name='lstm')
        predicted, score, rmse = train_model_instance.train(
            Xtrain=x_train, Ytrain=y_train,
            Xtest=x_test, Ytest=y_test)
        score_check = False
        rmse_check = False
        if score < 0.001:
            score_check = True
        self.assertEqual(True, score_check)
        if rmse < 0.031:
            rmse_check = True
        self.assertEqual(True, rmse_check)
        color_list = ['r--', 'b:']
        label_list = ['predict_x', 'real_y']
        data_list = [predicted, y_train]
        visualize_wave(color_list=color_list, label_list=label_list,
                       data_list=data_list)