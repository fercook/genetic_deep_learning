# -*- coding: utf-8 -*-
"""

acousticbrains.org
freesound.org
MTG/essentia github

Testing the genetic improvement of datasets in a toy model (Cartpole balance)
from the OpenAI Gym

Created on Wed Jun 22 19:46:14 2016

@author: Fernando Cucchietti

Strategy:

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

"""

import gym
import numpy as np
#import scipy
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Masking
from keras.preprocessing import sequence
from collections import defaultdict


#### MAIN PARAMETERS
# genetic
generations = 4
num_population = 100
num_survivors = 10
# model
lstm_layers = [50,50,50]
epochs_by_training = 20
batch_size=1
# io and various
output_path = "output/"
max_timesteps = 200

# Setup
env=gym.make('CartPole-v0')
input_dim = env.action_space.n + env.observation_space.shape[0]
output_dim = input_dim #env.action_space.n

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# Create model
dropout_U = 0.3
dropout_W = 0.3
def generator_model(layer_sizes):
    model = Sequential()
    #model.add(Masking(mask_value=0., input_shape=(max_timesteps, input_dim)))
    model.add(LSTM(layer_sizes[0], return_sequences=True, stateful=False, dropout_W=dropout_W, dropout_U=dropout_U,input_dim=input_dim))
    for lsize in layer_sizes[1:-1]:
        model.add(LSTM(lsize, return_sequences=True, stateful=False, dropout_W=dropout_W, dropout_U=dropout_U))
    model.add(LSTM(layer_sizes[-1], return_sequences=True, stateful=False, dropout_W=dropout_W, dropout_U=dropout_U))
    model.add(TimeDistributed(Dense(output_dim, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop') #or adamax ?
    return model

def generate_sequence(model,environment):
    # The seed is the initial state of the environment, random every time
    # Vectors will have action, observation where observation is different from zero only at beginning
    seq = np.empty(0)
    zero_observation = np.zeros( environment.observation_space.shape[0] )
    zero_action = np.zeros( environment.action_space.n )
    # Get initial state as seed
    seed = np.concatenate(( zero_action, environment.reset() ))
    seq=seed.reshape(((1,)+seed.shape))
    # Compute first action
    # from predictions of the model I need to sample
    #only one "batch", last action, only action space
    action = sample(model.predict(seq.reshape((1,)+seq.shape))[0,-1,0:2])
    observation, reward, done, info = environment.step(action)
    one_hot_action = np.eye(environment.action_space.n)[action]
    seq=np.vstack((seq,np.concatenate((one_hot_action,zero_observation))))
    while(not done and seq.shape[0] < max_timesteps):
        action = sample(model.predict(seq.reshape((1,)+seq.shape))[0,-1,0:2]) # np.random.randint(2) #
        observation, reward, done, info = environment.step(action)
        one_hot_action = np.eye(environment.action_space.n)[action]
        seq=np.vstack((seq,np.concatenate((one_hot_action,zero_observation))))
    # Do I need to pad short sequences????
    return seq

def breed_generation(population,model, environment):
    pop = []
    for _ in range(population):
        pop.append(generate_sequence(model,environment))
    return pop

def decimate_generation(population, survivors):
    population.sort(key=len)
    return population[-survivors:]

fixeddataset = """
def prepare_dataset(population):
    X=np.zeros((len(population),200,6))
    Y=np.zeros((len(population),200,6))
    for i,x in enumerate(population):
        sublen = len(x)-1
        X[i,:sublen,:]=x[:-1]
        Y[i,:sublen,:]=x[1:]
    #X = [ x[:-1] for x in population]
    #Y = [ x[1:] for x in population]
    return X, Y


def prepare_dataset(population):
    X = [ x[:-1] for x in population]
    Y = [ x[1:] for x in population]
    return X, Y
    """

def prepare_dataset(population):
    X = defaultdict(list)
    Y = defaultdict(list)
    for x in population:
        X[len(x)].append(x[:-1])
        Y[len(x)].append(x[1:])
    return X, Y


def log_stats(population, output_file):
    lens = [len(x) for x in population]
    mi = np.min(lens)
    ma = np.max(lens)
    mean = np.mean(lens)
    std = np.std(lens)
    with open(output_file+".stats",'w') as fo:
        fo.write(str(mi)+",")
        fo.write(str(ma)+",")
        fo.write(str(mean)+",")
        fo.write(str(std))
    with open(output_file+".hist",'w') as fo:
        for n in range(mi,ma+1):
            fo.write(str(n)+","+str(lens.count(n))+"\n")
    return mean, std

### And we start
print ("Creating model")
mixer = generator_model(lstm_layers)

print ("Breeding initial generation...")
full_pop = breed_generation(num_population, mixer, env)
m,s = log_stats(full_pop, output_path+"gen.0")
print ("Start stats: "+str(m))
for gen in range(generations):
    best_breed = decimate_generation(full_pop, num_survivors)
    X_buckets,Y_buckets = prepare_dataset(best_breed)
    total_buckets = len(X_buckets.keys())
    # We must train batches by hand because sequences have different lengths
    for epoch in range(epochs_by_training):
        for n,seqlen in enumerate(X_buckets.keys()):
            if seqlen>0: # some strange bug
                X=np.asarray(X_buckets[seqlen])
                Y=np.asarray(Y_buckets[seqlen])
                loss = mixer.train_on_batch(X,Y)
                print ("Generation: "+str(gen)+", epoch: "+str(epoch)+", bucket: "+str(n)+"/"+str(total_buckets)+", LOSS: "+str(loss))
    print ("Breeding generation "+str(gen)+"...")
    full_pop = breed_generation(num_population, mixer, env)
    m,s = log_stats(full_pop, output_path+"gen."+str(gen))
    print ("Generation "+str(gen)+": "+str(m))
