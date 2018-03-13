# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from scipy import signal
import numpy as np
from scipy import signal


class DataTransform(object):

    def __init__(self, data: np.array, sampling_frequency: int=10e3):
        self.data = data
        self.sampling_frequency = sampling_frequency

    def data_transform(self, transform_option: str='spectrogram') -> np.array:
        if transform_option == 'spectrogram':
            return self.__spectrogram_transform()
        if transform_option == 'wavelet':
            return self.__wavelet_transform()

    def __wavelet_transform(self):
        f, t, Sxx = signal.spectrogram(self.data, self.sampling_frequency)
        f = np.arange(1, max(f))
        cwtmatr = signal.cwt(self.data, signal.ricker, f)
        return cwtmatr, max(f)

    def __spectrogram_transform(self):
        f, t, Sxx = signal.spectrogram(self.data, self.sampling_frequency)
        return f, t, Sxx
