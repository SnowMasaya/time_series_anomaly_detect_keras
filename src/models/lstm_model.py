# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.models import Input, Model
import tensorflow as tf


class LSTMModel(object):

    def __init__(self,
                 input_size: tuple = (12, 45003, 1),
                 batch_size: int = 100,
                 num_time_steps: int = 5,
                 optimizer: str = 'adam',
                 loss: str = 'mean_squared_error',
                 metrics: list = ['mean_squared_error'],
                 hidden: int = 64,
                 variable_number: int = 1,
                 wave_type_number: int = 1,
                 stateful: bool = False,
                 ):
        self.hidden = hidden
        self.batch_size = batch_size
        self.num_time_steps = num_time_steps
        self.input_size = input_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.variable_number = variable_number
        self.wave_type_number = wave_type_number
        self.stateful = stateful

    def model_define(self):
        wave_output = self.wave_type_number
        if self.wave_type_number == 1:
            batch_input_shape = (self.batch_size, self.num_time_steps,
                                self.wave_type_number)
        if self.wave_type_number != 1:
            batch_input_shape = (self.batch_size,
                                 self.wave_type_number,
                                 self.num_time_steps)

        with tf.name_scope('Model'):
            with tf.name_scope('Input'):
                input = Input(shape=(self.num_time_steps,
                                     self.wave_type_number))
            with tf.name_scope('LSTM'):
                lstm_out = LSTM(self.hidden, stateful=self.stateful,
                                batch_input_shape=batch_input_shape,
                                return_sequences=False)(input)
            with tf.name_scope('Out'):
                output = Dense(wave_output)(lstm_out)
        model = Model(input=[input], output=output)
        with tf.name_scope('ModelCompile'):
            model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)
        return model
