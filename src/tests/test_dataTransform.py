# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from unittest import TestCase
from data.data_transform import DataTransform
import pandas as pd
from pandas_datareader.data import DataReader
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import filecmp
import os


class TestDataTransform(TestCase):
    def test_data_transform(self):
        a = DataReader('F', 'google',
                       datetime(2006, 6, 1),
                       datetime(2016, 6, 1))
        a_returns = pd.DataFrame(np.diff(np.log(a['Close'].values)))
        reshape_value = a_returns.values.reshape(a_returns.values.shape[0])
        data_transoform_instance = DataTransform(data=reshape_value)
        f, t, Sxx = \
            data_transoform_instance.data_transform(
                transform_option='spectrogram')
        plt.pcolormesh(t, f, Sxx)

        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        spectrogram_file = 'test_spectrogram.png'
        plt.savefig(spectrogram_file)
        self.assertEqual(True,
                         filecmp.cmp(spectrogram_file,
                                     'tests/test_data/' + spectrogram_file))
        os.remove(spectrogram_file)

    def test_data_transform_wave_let(self):
        a = DataReader('F', 'google',
                       datetime(2006, 6, 1),
                       datetime(2016, 6, 1))
        a_returns = pd.DataFrame(np.diff(np.log(a['Close'].values)))
        reshape_value = a_returns.values.reshape(a_returns.values.shape[0])
        data_transoform_instance = DataTransform(data=reshape_value)
        cwtmatr, max_frequency = \
            data_transoform_instance.data_transform(
                transform_option='wavelet')
        print(cwtmatr.shape)
        plt.imshow(cwtmatr, extent=[0, cwtmatr.shape[1], 1, cwtmatr.shape[0]],
                   cmap='PRGn', aspect='auto',
                   vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        wavelet_file = 'test_wavelet.png'
        plt.savefig(wavelet_file)
        self.assertEqual(True,
                         filecmp.cmp(wavelet_file,
                                     'tests/test_data/' + wavelet_file))
        os.remove(wavelet_file)
