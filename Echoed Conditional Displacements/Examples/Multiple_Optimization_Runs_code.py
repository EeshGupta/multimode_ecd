# Objective: n=5 >15 layers

#System Info
from tensorflow.python.client import device_lib
import tensorflow as tf

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpus= get_available_gpus()
print(gpus)

# Imports
import time
import sys 
sys.path.append('/home/eag190/ECD_control/')

import numpy as np
from qutip import *
sys.path.append(r'/home/eag190/mcd/Echoed Conditional Displacements/Two Mode/class_description')
from MECD_paramV1 import BatchOptimizer as BatchOptimizer
#from Simulation_Classes_Two_ModeV8 import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd

# Data Saving
fname = 'fock_swap_5.csv'
columns = ['task', 'layer', 'pulse_time', 'BO_fid', 'qutip_fid', 'filename']
df = pd.DataFrame([[None for i in range(len(columns))]], columns=columns)
df.to_csv(fname, index=False)

#2 files here are redundant
#angles_filename_prefix = 'Data/angles'
opt_filename_prefix = 'Data/opt_'

# ----------------------------------------------------------------------------------------------
# Relevant Code
# ----------------------------------------------------------------------------------------------

#Modes Truncation
N = 15

def two_mode_state(fock1, fock2, qubit_g = True): 
    '''
    Returns g x fock1 x fock 2
    
    #takes N1, N2 from global
    '''
    psi_1 = basis(N,fock1) #initial state
    psi_2 = basis(N,fock2)
    return tensor(basis(2,0), psi_1, psi_2)

#Optimization of ECD Circuit parameters (betas, phis, and thetas)
#the optimization options
opt_params = {
    'N_modes': 2,
    'N_blocks' : 4, #circuit depth
    'N_multistart' : 50, #Batch size (number of circuit optimizations to run in parallel)
    'BCH_approx': False,
    'epochs' : 100, #number of epochs before termination
    'epoch_size' : 100, #number of adam steps per epoch
    'learning_rate' : 0.001, #adam learning rate
    'term_fid' : 0.999, #terminal fidelitiy
    'dfid_stop' : 1e-6, #stop if dfid between two epochs is smaller than this number
    'beta_scale' : 3.0, #maximum |beta| for random initialization
    'N_cav': N, #number of levels in mode 1
    #'N_cav2': N2, #number of levels in mode 2
    'initial_states' : [], #qubit tensor oscillator, start in |g> |0>
    'target_states' : [], #end in |e> |target>.
    #"initial_params": init_params,
    'name' : '',#'Fock1 %d' % Fock1, #name for printing and saving
    'filename' : None, #if no filename specified, results will be saved in this folder under 'name.h5'
    }
def main( n, start, df, reruns=1):
    '''
    Vary depth
    0n->n0 state transfer
    start time
    df is data frame to store
    '''
    first_iter = True #this is for dataframe
    filenum = 800
    
    for n__ in range(5, n+1): #for |0n> -> |n0> transfer   # in general
        #n_ = 5 -n__+1#for this particular file
        n_ = n__
        initial = two_mode_state(n_, 0)
        target = two_mode_state(0, n_)
        opt_params['initial_states'] = [initial]
        opt_params['target_states'] = [target]

        best_fid = 0
        layer =15

        while best_fid<(1-1e-3) and layer <20 : 
            for r in range(reruns):
                clear_output(wait = True)            
                print('prev layer' + str(layer))
                #print('prev fid' + str(fid))
                print(time.time()-start)
                print('--------')

                #optimizer
                opt_params['N_blocks'] = layer
                opt_params['name'] = opt_filename_prefix + str(filenum) + '.h5'
                opt = BatchOptimizer(**opt_params)
                #print(opt.filename)
                opt.optimize()
                BO_fid = opt.best_fidelity()
                BO_fid = BO_fid.real
                #angles_filename = angles_filename_prefix + '_' + str(filenum) + '.txt'
                #opt.save_angles(filename = angles_filename)

                print('finished optimization')
                print(time.time()-start)
                print('-------')

                #pulses
    #             pulse_sim = ecd_pulse_two_mode(param_file = opt_params['name']) 
    #             pulse_sim.get_pulses()
                pulse_time =0#len(pulse_sim.cavity1_dac_pulse_GHz)

    #             #qutip simulation
    #             qutip_sim  = qutip_sim_two_mode(n_q = 2, n_c1 = N1,
    #                                             n_c2 = N2, alpha1 = pulse_sim.alpha1,
    #                                             alpha2 = pulse_sim.alpha2,
    #                                             qubit_pulse = pulse_sim.qubit_dac_pulse_GHz)
    #             qutip_sim.me_solve(initial = initial)
                qutip_fid = 0#qutip_sim.get_fidelity(target)

        #             print('finished qutip_sim')
        #             print(time.time()-start)
        #             print('-------')

                new_row = [[n_, 
                            layer, 
                            pulse_time,
                            BO_fid, 
                            qutip_fid,
                            opt_params['name']]]
                df_new = pd.DataFrame(new_row, columns=columns)
                df = pd.concat([df,df_new], ignore_index = True)
                df.to_csv(fname, index=False)
                filenum+=1

                fid = BO_fid
                if fid>best_fid:
                    best_fid = fid
            layer +=1
            
    return df

#----------------------------------------------------------------------------------------
# Run Code
#----------------------------------------------------------------------------------------
with tf.device(gpus[0]):
    start = time.time()
    # reruns = 
    df = main(5, start, df)
