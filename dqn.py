###########################################################
'''
    Copyright 2017, MIT

    This file is part of PremieRL.
    
'''
############################
# Written by Tim Ioannidis #
############################

# ************************************************** #
# ********* Class of deep Neural Network  ********** #
# ************************************************** #

import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DQN:
    def __init__(self, no_in_vars, no_actions):
        self.model = Sequential()
        self.build_network(no_in_vars,no_actions)

    ###################################
    #### Build deep neural network ####
    ###################################
    def build_network(self,no_in_vars,no_actions):
        ##################
        ### Parameters ###
        ##################
        hidden_units_1 = 32
        hidden_units_2 = 32
        #################
        ### Build DNN ###
        #################
        ### add first dense layer
        self.model.add(Dense(units=hidden_units_1, input_dim=no_in_vars,
                                    kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', 
                                    alpha_regularizer=None, alpha_constraint=None, shared_axes=None)))
        self.model.add(Dropout(0.25))
        ### add second dense layer
        self.model.add(Dense(units=hidden_units_2, kernel_initializer='random_uniform', bias_initializer='zeros'))
        self.model.add(BatchNormalization())
        self.model.add(Activation(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', 
                                    alpha_regularizer=None, alpha_constraint=None, shared_axes=None)))
        self.model.add(Dropout(0.25))
        ### add final layers
        self.model.add(Dense(units = no_actions, activation = 'linear'))
        ##################### 
        ### compile model ###
        ##################### 
        self.model.compile(keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), "mse")

    ###################################
    ### copy weights to other model ###
    ###################################
    def update_model(self,dqn_origin):
        self.model.set_weights(dqn_origin.model.get_weights())
 
