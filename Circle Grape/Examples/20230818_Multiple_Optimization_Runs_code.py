#--------------------------------------------------------------------------------Imports 
#%matplotlib inline
import os
import sys
import inspect
import numpy as np
from scipy.special import factorial
import h5py

#data_path = '/data'     ... data path specified later
#data_path
#initial_pulse = '../pulses/example_pulses/transmon_cat_initial_pulse.h5'
from h5py import File
import matplotlib.pyplot as plt
from pylab import*
from qutip import*

from scipy import interpolate
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)


sys.path.append(r'/home/eag190/quantum-optimal-control')
from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape
from quantum_optimal_control import*

from IPython.display import clear_output

sys.path.append(r'/home/eag190/mcd/Circle Grape/class description')
from circle_grape_v5 import *
#--------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------Data Storage
import pandas as pd
fname = '/08072023_state_prep'
parent_path = r'/home/eag190/Multimode-Conditional-Displacements/hpc_runs/multimode_circle_grape/Grape on multiple modes/State Transfer/20230720'
columns=['task', 'time', 'steps','alpha', 'detuning', 'qubit_drive_amp','err', 'filename']#total_time, steps, alpha, detuning,  qubit_drive_amp, err, filenum
#columns = ['task', 'layer', 'pulse_time', 'BO_fid', 'qutip_fid', 'filename']
df = pd.DataFrame([], columns=columns)
df.to_csv(parent_path + fname, index=False)
opt_filename_prefix  = r'Data/opt_data'


#Add old df
old_fnames = [parent_path + '/MASTER_Dataframe']
for f in old_fnames:
    old_df = pd.read_csv(f)
    df = pd.concat([df, old_df])

#drop unnecssary columns
df_ = df.copy()
for col in df_.columns:
    if col not in columns:
        df = df.drop(columns = [col])


data_path = parent_path + '/Data2'
#----------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------Pre Code + Parameter Setting
chis = array([-33, -35]) 
kappas  = array([0,0]) # kHz
transmon_levels = 3

def pad(listy, n): 
    '''
    Adds 0s to end of listy so that it containts n amount of terms
    '''
    listy = list(listy)
    while len(listy)<n:
        listy.append(0.0)
    return listy

states_forbidden_list = []

convergence = {'rate': 0.01, 'update_step': 1000, 
               'max_iterations': 10000,
               'conv_target': 1e-3, 
               'learning_rate_decay': 500.0}

initial_guess = None

def check_max_amp(listy, max_amp):
    '''
    check whether any element is greater than max amp, if so, make it equal to max amp
    '''
    new_listy = np.copy(listy)
    for i in range(len(listy)): 
        if (listy[i] / (2 * np.pi * max_amp)) >1: 
            new_listy[i] = 2 * np.pi * max_amp
    return new_listy

def initial_guess(guess_fock, steps, time, max_amp, df): 
    '''
    FOr g0 -> gn transfer, retrieves the opt param for g0->n-i as initial guess 
    
    n = current fock (not req as argument)
    n-i = guess fock
    '''
    #print(df)
    df_ = df.loc[(df['time']<=time) & (df['task'] == guess_fock)] # want parameters from optimized pulse satisfying these conditions
    df_ = df_.reset_index(drop = True)
    if len(df_) ==0:
        return None
    idx = np.argmin(list(df_['err']))
    #return df_['err'][idx]
    filename = df_['filename'][idx]
    file = h5py.File(filename,'r')
    
    #take data corresponding to best fidelity for 0 to 1 pulse
    uks = list(file['uks'][np.argmin(file['error'])])
    #extend these lists to current step size
    initial_guess = np.array([pad(uk, steps) for uk in uks])
    
    #check whether any element is greater than max amp, if so, make it equal to max amp
    for i in range(len(initial_guess)): 
        initial_guess[i] = check_max_amp( initial_guess[i], max_amp) 
    return initial_guess


#Parameter Setting 
## f state stuff
deltas = [-2.0e+9 *(1e-9), -2.1e+9 *(1e-9)]   # Mode 1,2 freq w.r.t qubit (omega_q - omega_r)
alpha = 150e+6 *(1e-9)
g = 30e+6 *(1e-9)

def Lambda(g, delta, alpha, j): 
    num = j*(g**2)
    denom = delta - ((j+1)*alpha)
    return num/denom

def chi(g, delta, alpha, j): 
    if j == 0: 
        return -1*Lambda(g, delta, alpha, 1)
    return Lambda(g, delta, alpha, j) - Lambda(g, delta, alpha, j+1)

def chi_wrt_g(g, delta, alpha, j): 
    '''
    with respect to chi_g = chi_0
    '''
    return chi(g, delta, alpha, j) - chi(g, delta, alpha, 0)

chi_e = array([chi_wrt_g(g, deltas[i], alpha, 1) for i in range(2)]) 
chi_f = array([chi_wrt_g(g, deltas[i], alpha, 2) for i in range(2)])
kappa  = array([0,0]) # kHz

##More parameter setting

alpha = 30
detuning = 0.005 # GHz
qubit_drive_amp = 0.025
#param

mode = 2
mode_levels = 10
#chi,kappa = chis[:mode],kappas[:mode]
kappa = kappas[:mode]
#chi = [i*1e-6 for i in chi]
kappa = [i*1e-6 for i in kappa]
# circle_grape_params = {"chis_e":chi,"chis_f":chi, "kappas":kappa,"alpha":alpha,"delta_c":detuning}
circle_grape_params = {"chis_e":chi_e,
                       "chis_f": chi_f,
                       "kappas":kappa,
                       "alpha":alpha,
                       "delta_c":detuning}
#-----------------------------------------------------------------------------

# ------------------------------------------------------------------------------ Data COmpression 

def remove_inter_vecs(filename):
    '''
    Removes intervecs from saved simulation file, massively cutting down on memory cost.
    '''
    filename = filename

    if not os.path.exists(filename):
        print('not exist')
        return filename

    new_filename = filename.split('.')[0] + '_r'+'.h5'

    f_old = h5py.File(filename, 'r')
    f_new = h5py.File(new_filename, 'w')

    #data to be thrown out (reg_coeffs and convg. included because could not copy them into new file; numpy unicode error)
    ignore_keys = ['reg_coeffs', 'convergence',  'inter_vecs_imag', 'inter_vecs_mag_squared', 'inter_vecs_raw_imag', 'inter_vecs_raw_real', 'inter_vecs_real']

    #copying data into new file
    for key in f_old.keys():
        if key not in ignore_keys:
            data = np.array(f_old.get(key))
            #print(key)
            f_new.create_dataset(key, data = data)

    #deleting old file
    f_old.close()
    os.remove(filename)
    #new file close
    f_new.close()
    return new_filename

#------------------------------------------------------------------Looping
fock_states = [2]

def main(fname = fname, df = df):
    '''
    Vary time length, get fidelity
    '''
    filenum = 1000
    last_err = 1
    last_time = 100
    initial_guess_ = None

    for fock in fock_states:
        print(fock)
        last_err = 1
        # if fock == 1: 
        #     last_time = 2250
        #     last_err = 3e-3
        
        
        while last_err>1e-3 and last_time<2000:
            time = last_time
            print(time)
            clear_output(wait = True)
            total_time = time
            
            steps = int(time/2) # 2ns timestep
            
            reg_coeffs = {'dwdt': 0.1, 'd2wdt2': 1.0e-3, 'forbid_dressed': False,
              'states_forbidden_list':states_forbidden_list,
              'forbidden_coeff_list': [1.0*steps] * len(states_forbidden_list)}
            
            op = multimode_circle_grape_optimal_control(mode_state_num = mode_levels,
                                                        transmon_levels = transmon_levels, 
                                                        f_state = False,
                                                        number_of_modes = mode,hparams = circle_grape_params,
                                                        add_disp_kerr=False)


            #choosing initial guess
            
            for t in range(0, 5, 1): # 5 initial guesses
                initial_guess_ = None
                if fock-t >0: 
                    initial_guess_ = initial_guess(fock -t, steps, time, qubit_drive_amp,  df)


                #qubit_drive_amp = drive_amp # Ghz
                filename = 'opt_data'  + str(filenum )
                filenum+=1
                ss = op.run_optimal_control(state_transfer = True, initial_states = [fock], target_states = [mode_levels*fock], 
                                total_time = total_time, steps = steps,max_amp = qubit_drive_amp, 
                                taylor_terms = None,is_dressed=False, 
                                use_gpu= True,
                                convergence = convergence, reg_coeffs =  reg_coeffs,
                                plot_only_g = True,
                                #f_state = True,
                                states_forbidden_list = states_forbidden_list,initial_guess = initial_guess_, 
                                file_name=filename, data_path=data_path, save = True)
                
                #saving data
                hf = op.openfile()
                err  = min(hf['error'])
                if np.isnan(err):
                    err = -1
                #err = 1
                last_err = err
                new_filename = remove_inter_vecs(op.filename)
                new_row = [[fock, total_time, steps, alpha, detuning,  qubit_drive_amp, err, new_filename]]
                df_new = pd.DataFrame(new_row, columns=columns)
                df = pd.concat([df_new,df], ignore_index = True)
                df.to_csv(parent_path + fname, index=False)
                
            last_time +=100
        last_time -=100 # so that next fock state transfer starts at last_time, not last_time + 200ns
        
    clear_output(wait = True)
    return df

df = main()