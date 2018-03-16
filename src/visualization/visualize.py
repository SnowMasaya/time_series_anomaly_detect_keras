# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import pylab


def visualize_wave(color_list: list, label_list: list, data_list):
    pylab.figure(1)
    plot_value = int('111')
    pylab.figure(figsize=(10, 10))
    pylab.subplot(plot_value)
    pylab.xlabel("Time")
    pylab.ylabel("wave")
    for index, data in enumerate(data_list):
        pylab.plot(data, color_list[index], label=label_list[index])
    pylab.legend(loc='upper right')
    pylab.show()
    pylab.close()
