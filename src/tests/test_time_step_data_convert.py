# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.time_series import time_step_data_convert
from data.time_series import multi_variate_time_series_convert
import pandas as pd
from pandas_datareader.data import DataReader
import numpy as np
from pandas import read_csv
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import filecmp
import os


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


class TestTime_step_data_convert(TestCase):
    def test_time_step_data_convert(self):
        a = DataReader('F', 'google',
                       datetime(2006, 6, 1),
                       datetime(2016, 6, 1))
        a_returns = pd.DataFrame(np.diff(np.log(a['Close'].values)))
        a_returns.index = a.index.values[1:a.index.values.shape[0]]
        scaler, x_train, y_train, x_test, y_test = time_step_data_convert(
            original_data=a_returns.values,
            num_time_steps=5,
            batch_size=10,
        )

        x_normalize_file = 'test_X_train_normalize.npy'
        y_normalize_file = 'test_Y_train_normalize.npy'
        x_test_normalize_file = 'test_X_test_normalize.npy'
        y_test_normalize_file = 'test_Y_test_normalize.npy'
        x_file = 'test_X.npy'
        y_file = 'test_Y.npy'

        np.save(x_normalize_file, x_train)
        np.save(y_normalize_file, y_train)
        np.save(x_test_normalize_file, x_test)
        np.save(y_test_normalize_file, y_test)
        self.assertEqual(True,
                         filecmp.cmp(x_normalize_file,
                                     'tests/test_data/' + x_normalize_file))
        self.assertEqual(True,
                         filecmp.cmp(y_normalize_file,
                                     'tests/test_data/' + y_normalize_file))
        self.assertEqual(True,
                         filecmp.cmp(x_test_normalize_file,
                                     'tests/test_data/' +
                                     x_test_normalize_file))
        self.assertEqual(True,
                         filecmp.cmp(y_test_normalize_file,
                                     'tests/test_data/' +
                                     y_test_normalize_file))
        x = scaler.inverse_transform(x_train)
        y = scaler.inverse_transform(y_train)
        np.save(x_file, x)
        np.save(y_file, y)
        self.assertEqual(True,
                         filecmp.cmp(x_file,
                                     'tests/test_data/' + x_file))
        self.assertEqual(True,
                         filecmp.cmp(y_file,
                                     'tests/test_data/' + y_file))
        os.remove(x_file)
        os.remove(y_file)
        os.remove(x_normalize_file)
        os.remove(y_normalize_file)
        os.remove(x_test_normalize_file)
        os.remove(y_test_normalize_file)

    def test_multi_variate_time_series_convert(self):
        dataset = read_csv('tests/test_data/PRSA_data_2010.1.1-2014.12.31.csv',
                           parse_dates=[['year', 'month', 'day', 'hour']],
                           index_col=0, date_parser=parse
                           )
        dataset.drop('No', axis=1, inplace=True)
        dataset.columns = ['pollution', 'dew', 'temp',
                           'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        dataset.index.name = 'date'
        dataset['pollution'].fillna(0, inplace=True)
        dataset = dataset[24:]
        dataset.to_csv('tests/test_data/pollution.csv')
        dataset = read_csv('tests/test_data/pollution.csv',
                           header=0, index_col=0)
        values = dataset.values
        encoder = LabelEncoder()
        values[:, 4] = encoder.fit_transform(values[:, 4])
        values = np.swapaxes(values, 0, 1)
        scaler, x_train, y_train, x_test, y_test = \
            multi_variate_time_series_convert(values,
                                              num_time_steps=5,
                                              batch_size=10,
                                              )
        x_normalize_file = 'test_X_train_multivariate__normalize.npy'
        y_normalize_file = 'test_Y_train_multivariate__normalize.npy'
        x_test_normalize_file = 'test_X_test_multivariate_normalize.npy'
        y_test_normalize_file = 'test_Y_test_multivariate_normalize.npy'

        np.save(x_normalize_file, x_train)
        np.save(y_normalize_file, y_train)
        np.save(x_test_normalize_file, x_test)
        np.save(y_test_normalize_file, y_test)

        self.assertEqual(True,
                         filecmp.cmp(x_normalize_file,
                                     'tests/test_data/' + x_normalize_file))
        self.assertEqual(True,
                         filecmp.cmp(y_normalize_file,
                                     'tests/test_data/' + y_normalize_file))
        self.assertEqual(True,
                         filecmp.cmp(x_test_normalize_file,
                                     'tests/test_data/' +
                                     x_test_normalize_file))
        self.assertEqual(True,
                         filecmp.cmp(y_test_normalize_file,
                                     'tests/test_data/' +
                                     y_test_normalize_file))
        os.remove(x_normalize_file)
        os.remove(y_normalize_file)
        os.remove(x_test_normalize_file)
        os.remove(y_test_normalize_file)
