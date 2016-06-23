# -*- coding: utf-8 -*-
"""
Testing the genetic improvement of datasets in a toy model (Cartpole balance)
from the OpenAI Gym

Created on Wed Jun 22 19:46:14 2016

@author: Fernando Cucchietti
"""

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Masking
from keras.preprocessing import sequence

#### MAIN PARAMETERS
max_timesteps = 200
lstm_layers = [512,512]

# Setup
env=gym.make('CartPole-v0')
input_dim = env.action_space.n + env.observation_space.shape[0]
output_dim = env.action_space.n

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# Create model
dropout_U = 0.3
dropout_W = 0.3
def generator_model(layer_sizes,time_steps):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(time_steps, input_dim)))
    for lsize in layer_sizes:
        model.add(LSTM(lsize, return_sequences=True, stateful=False, dropout_W=dropout_W, dropout_U=dropout_U))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


mixer = model(lstm_layers,max_timesteps])

'''
1) Create model

2) For each generation:
 #compute new specimens 
     reset environment
     predict sequence from initial state (model )
     store initial value+action sequence, stopping when done          

3) rank specimens (lenght of each sequence)
4) print or record max ranking of specimens
5) keep best N specimens and pad them to zero (truncate?)
6) retrain the model using only the N best ones
'''