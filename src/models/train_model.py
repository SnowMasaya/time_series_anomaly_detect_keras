from keras.callbacks import TensorBoard
from time import gmtime, strftime
import os
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import math


class TrainModel(object):

    def __init__(self,
                 model_object: object,
                 model_name: str,
                 batch_size: int=10,
                 num_time_step: int=5,
                 num_epoches: int = 3
                 ):
        self.model_object = model_object
        self.batch_size = batch_size
        self.num_time_step = num_time_step
        self.num_epoches = num_epoches
        self.define_model = self.__model_setting()
        self.model_setting_parameter_save_log = self.__model_setting_parameter(model_name)

    def train(self, Xtrain: np.array, Ytrain: np.array,
              Xtest: np.array, Ytest: np.array):
        with tf.name_scope('Train'):
            self.define_model.fit(
                Xtrain, Ytrain, batch_size=self.model_object.batch_size,
                callbacks=[self.__make_tensor_board(set_dir_name="./log/" + self.model_setting_parameter_save_log),],  # noqa
                epochs=self.num_epoches,
                validation_data=(Xtest, Ytest),
                shuffle=False)
            self.define_model.reset_states()
        predicted, score, rmse  = self.__evaluate(Xtest=Xtest, Ytest=Ytest)
        return predicted, score, rmse

    def __evaluate(self,  Xtest: np.array, Ytest: np.array):
        score, _ = self.define_model.evaluate(Xtest, Ytest,
                                  batch_size=self.model_object.batch_size)
        rmse = math.sqrt(score)
        print("\nMSE: {:.3f}, RMSE: {:.3f}".format(score, rmse))
        predicted = self.define_model.predict(Xtest,
                                  batch_size=self.model_object.batch_size,)
        predicted = np.reshape(predicted, (predicted.size,))
        self.define_model.save('../models/' +
                               self.model_setting_parameter_save_log + '.h5')
        return predicted, score, rmse

    def __model_setting(self):
        input_size = (self.batch_size, self.num_time_step, 1)
        self.model_object.input_size = input_size
        self.model_object.batch_size = self.batch_size
        Model = self.model_object.model_define()
        Model.summary()
        return Model

    def __make_tensor_board(self, set_dir_name: str) -> object:
        tictoc = strftime("%a_%d_%b_%Y_%H_%M_%S", gmtime())
        directory_name = tictoc
        log_dir = set_dir_name + '_' + directory_name
        os.mkdir(log_dir)
        tensorboard = TensorBoard(log_dir=log_dir,
                                  write_graph=True,
                                  )
        return tensorboard

    def __model_setting_parameter(self, model_name: str):
        model_setting_parameter = model_name + \
                                  '_batch_size_' + \
                                  str(self.batch_size) + \
                                  '_num_time_step_' + \
                                  str(self.num_time_step) + \
                                  '_num_epochs_' + \
                                  str(self.num_epoches) + \
                                  '_hidden_' + \
                                  str(self.model_object.hidden) + \
                                  '_optimizer_' + \
                                  str(self.model_object.optimizer)
        return model_setting_parameter
