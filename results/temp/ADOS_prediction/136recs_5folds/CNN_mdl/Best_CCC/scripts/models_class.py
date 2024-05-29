# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:04:53 2020

@author: marinamu
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D 
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import RandomUniform 


class Models:
    def __init__(self, learn_rate, input_shape, model_name, n_out=1,
                 out_act='linear', loss=['mse'], metric=['mse']):
        self.learn_rate = learn_rate
        self.input_shape = input_shape
        self.model_name = model_name
        self.n_out = n_out
        self.out_act = out_act
        self.loss = loss
        self.metric = metric
        self.model = None

    # --------------------------------------------------------------------------------------------#
    def define_mdl(self):
        if self.model_name == 'CNN_mdl':
            self.CNN_mdl()

    # --------------------------------------------------------------------------------------------#
    def CNN_mdl(self):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=3, input_shape=self.input_shape,  # w = kx48xf, b = fx1
                         activation='relu', kernel_initializer=RandomUniform(seed=1)))
        model.add(MaxPool1D(pool_size=3))
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu',
                         kernel_initializer=RandomUniform(seed=1)))  # , bias_initializer=Zeros()
        model.add(Dense(1024, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        model.add(Dense(128, activation='relu',
                        kernel_initializer=RandomUniform(seed=1)))
        #
        model.add(Flatten())
        model.add(Dense(self.n_out, activation=self.out_act,
                        kernel_initializer=RandomUniform(seed=1)))
        #  
        model.compile(loss=self.loss,
                      optimizer=RMSprop(learning_rate=self.learn_rate),  # RMS default params: lr=1e-3.rho(beta) = 0.9
                      metrics=[self.correlation, self.metric])

        self.model = model

    # --------------------------------------------------------------------------------------------#
    @staticmethod
    def correlation(x, y):  
        # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
        mx = tf.math.reduce_mean(x)
        my = tf.math.reduce_mean(y)
        xm, ym = x - mx, y - my
        r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
        r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
        r = r_num / r_den
        return 1.0 - r

