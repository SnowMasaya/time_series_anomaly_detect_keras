# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals

from unittest import TestCase
from data.smooth_time_series import smooth
import pandas as pd
from pandas_datareader.data import DataReader
import numpy as np
from datetime import datetime
import os
import filecmp


class TestSmooth(TestCase):
    def test_smooth(self):
        a = DataReader('F', 'google',
                       datetime(2006, 6, 1),
                       datetime(2016, 6, 1))
        a_returns = pd.DataFrame(np.diff(np.log(a['Close'].values)))
        reshape_value = a_returns.values.reshape(a_returns.values.shape[0])
        smooth_value = smooth(reshape_value)
        smooth_value_file = 'test_value_smooth.npy'
        np.save(smooth_value_file, smooth_value)
        self.assertEqual(True,
                         filecmp.cmp(smooth_value_file,
                                     'tests/test_data/' + smooth_value_file))
        os.remove(smooth_value_file)
