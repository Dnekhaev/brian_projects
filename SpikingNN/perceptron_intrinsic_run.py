from Stupid_digits import *
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import animation
from copy import copy

X,y = stupid_digits_dataset(100)
#classes = int(len(set(y)))
#n_input = int(X.shape[1])

from brian2 import *
import numpy as np
from time import clock

import json
import codecs

from equations_perceptron_intrinsic import *
from Perceptron_intrinsic import Perceptron as Perceptron

start_scope()

accuracies = []
''' -------==|| Fixed parameters ||==------- '''
time_per_image = 100*ms
time_step = 0.1*ms
#tau = 10*ms # tau for neuron's voltage 'v'
#tau_I = 15*ms # tau for neuron's current 'I'
#tau_h = 50*ms # tau for neuron's treshold

wmax = 1.
decay = 0.001

''' -------==|| Initialization parameters ||==------- '''
# Deafult example: inits = [['equal_[0,1]', None], ['equal_[-1,0]', 'i!=j']]
inits = [['equal_[0,1]', None], ['equal_[-1,0]', 'i!=j']]

''' -------==|| Monitor parameters ||==------- '''
monitor = {} # dict with names of objects to record and their parameters
#monitor['H'] = {'variables':['a'], 'dt':1*ms, 'record':True}
monitor['H'] = {'variables':['I_inp','I_intr','I','a'], 'dt':1*ms, 'record':True}
#monitor['Intrinsic_output_weights'] ={'variables':['w'], 'dt':50*ms, 'record':True}
#monitor['Input_weights'] ={'variables':['w'], 'dt':50*ms, 'record':True}

''' -------==|| Simulation parameters ||==------- '''
params_set = {'lr_intr': 0.1, 
              'c_intr_out': 0.9, 
              'beta': 150. * msecond, 
              'alpha': 27. * msecond, 
              'Teacher_amplitude': 45.0, 
              'lr': 0.1, 
              'c_inp': 0.3, 
              'c_diff': 1.25}

''' -------==|| Run the simulation ||==------- '''
#for j in np.arange(1):
j = 0
for c_intr_out in np.arange(10):
    for c_diff in np.arange(1,1.35,0.05):
        '''alpha = np.random.choice([10, 15, 20, 25, 30])*ms
        beta = np.random.choice([80, 90, 100, 110, 120, 130, 140, 150])*ms
        lr = 0.1
        lr_intr = 0.1
        c_inp = np.random.choice([0.1, 0.3, 0.5, 0.75, 1.0])
    
        c_intr_out = c_inp*np.random.choice([0,1,2,3,4,5,6,7,8,9,10])
        c_diff = np.random.choice([1.0, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
        Teacher_amplitude = c_inp*np.random.choice([25,50,100,150,200])
        
        params_to_optimize = {}
        params_to_optimize['alpha'] = alpha
        params_to_optimize['beta'] = beta
        params_to_optimize['lr'] = lr
        params_to_optimize['lr_intr'] = lr_intr
        params_to_optimize['c_inp'] = c_inp
        params_to_optimize['c_intr_out'] = c_intr_out
        params_to_optimize['c_diff'] = c_diff
        params_to_optimize['Teacher_amplitude'] = Teacher_amplitude
        '''
        params_set['c_diff'] = c_diff
        params_set['c_intr_out'] = c_intr_out
        params_to_optimize = params_set
        web = ['imshow_forward_weights_H','imshow_intrinsic_weights_H', 'plot_H']
        #web = ['plot_H']
        #web = False
                
        #params_to_optimize['c_intr_out'] = 0.0
        NN = Perceptron(X, y, params_to_optimize, 
                          time_step=time_step, 
                          time_per_image=time_per_image,
                          inits=inits, 
                          monitor=monitor, 
                          mod=True, 
                          high_verbosity=False,
                          web=web)
    
        print ('SIMULATION', j, params_to_optimize)
        #NN.load_weights('./NN.net')
        NN.mod = True
        print('train')
        NN.run(80000*ms) #400
        NN.mod = False
        print('test')
        NN.run(20000*ms) #100
        tries = [accuracy_score(NN.shown_labels[-100:], NN.predictions[-100:]), params_to_optimize]
        accuracies.append(tries)

        #save final figures for multiple simulations
        if 0:
            NN.web = ['imshow_forward_weights_H','imshow_intrinsic_weights_H']
       	    NN.imshow_forward_weights('H',5,2,5,5,j)
            NN.imshow_intrinsic_weights('H',5,2,2,5,j)
            #NN.web = False
            print (accuracies[-1][0])
            j += 1
            #NN.save_weights('./NN.net')
accs = []
for acc in accuracies:
    accs.append(acc[0])


print ('best', accuracies[np.argmax(accs)])









