# The time dependence of the blockade drive is rotated out.

import os
import sys
import inspect
import numpy as np
from scipy.special import factorial
import h5py
from quantum_optimal_control.helper_functions.grape_functions import *
from quantum_optimal_control.main_grape.grape import Grape
from itertools import product
import itertools

#data_path = '/home/eag190/Multimode-Conditional-Displacements/hpc_runs/multimode_circle_grape/sample workflow/data'
#data_path
#initial_pulse = '../pulses/example_pulses/transmon_cat_initial_pulse.h5'
from h5py import File
import matplotlib.pyplot as plt
from pylab import*
from qutip import*

from scipy import interpolate

#V1 : Given By Vatsan
#V3: Make sure that Grape now returns filename and that the following class has a local variable called filename; this avoids manual input
#V4: error with inserting nonlinearity in modes; for now ignore
#V5: Added f state
# The time dependence of the blockade drive is rotated out.


class multimode_circle_grape_optimal_control:
    
    def __init__(self,mode_state_num,number_of_modes,hparams,transmon_levels, f_state, t1params = None,add_disp_kerr=False, ROTATING=True,SAMPLE_RATE = 1):       
        self.mnum = mode_state_num
        self.mmnum = number_of_modes
        self.ROTATING = ROTATING
        self.hparams = hparams
        self.f_state = f_state # boolean 
        self.t1params = t1params
        self.SAMPLE_RATE = SAMPLE_RATE
        #redundant variables
        self.transmon_levels = transmon_levels
        self.qnum = transmon_levels
        self.mode_levels = self.mnum
        self.add_disp_kerr =add_disp_kerr

        self.initialize_operators()
        
####
#To Do : Clean up this matrix math by modularizing (either create general N level pauli x,y,z or use position and momentum
# Representations that ALec uses)
###
    def initialize_operators(self):
        '''
        Create qubit and mode versions of pauli/creation/annhilation operators
        '''
        self.Q_x = np.diag(np.sqrt(np.arange(1, self.transmon_levels)), 1)+np.diag(np.sqrt(np.arange(1, self.transmon_levels)), -1)
        self.Q_y = (0-1j) * (np.diag(np.sqrt(np.arange(1, self.transmon_levels)), 1)-np.diag(np.sqrt(np.arange(1, self.transmon_levels)), -1))
        self.Q_z = np.diag(np.arange(0, self.transmon_levels))
        self.I_q = np.identity(self.transmon_levels)

        #Qubit projection operators 
        self.Q_projs = []
        q_zeroes = [0 for i in range(self.transmon_levels)] #|g><g|, |e><e|, ...
        for k in range(0, self.transmon_levels): 
            diag = q_zeroes.copy()
            diag[k] = 1
            proj = np.diag(diag)
            self.Q_projs.append(proj)
        
        #Sigma_x matrices 
        self.Q_sigmaXs = [] # sigma_x^ge, sigma_x^fe, ...
        zeroes = np.zeros((self.transmon_levels, self.transmon_levels))
        for num in range(1, self.transmon_levels): 
            sigma_x = zeroes.copy()
            sigma_x[num, num-1] = 1
            sigma_x[num-1, num] = 1
            self.Q_sigmaXs.append(sigma_x)
        
        #Sigma_y matrices 
        self.Q_sigmaYs = [] # sigma_y^ge, sigma_y^fe, ...
        for num in range(1, self.transmon_levels): 
            sigma_y = zeroes.copy()
            sigma_y[num, num-1] = 1
            sigma_y[num-1, num] = -1
            self.Q_sigmaYs.append((0+1j)*sigma_y)

        
        # Mode Pauli Operatirs
        self.M_x = np.diag(np.sqrt(np.arange(1, self.mode_levels)), 1)+np.diag(np.sqrt(np.arange(1, self.mode_levels)), -1)
        self.M_y = (0-1j) * (np.diag(np.sqrt(np.arange(1, self.mode_levels)), 1)-np.diag(np.sqrt(np.arange(1, self.mode_levels)), -1))
        self.M_z = np.diag(np.arange(0, self.mode_levels))
        self.I_m = np.identity(self.mode_levels)
        self.am =  np.diag(np.sqrt(np.arange(1, self.mode_levels)), 1)
        self.amdag =  np.diag(np.sqrt(np.arange(1, self.mode_levels)), -1)
        self.aq =  np.diag(np.sqrt(np.arange(1, self.transmon_levels)), 1)

        self.M_zs,self.M_xs,self.M_ys,self.ams=  [],[],[],[]
        self.a_s,self.adag_s = [],[]
        
        for k in np.arange(self.mmnum):
            mmz = self.M_z*(k==0) + self.I_m*(1-(k==0))
            mmx = self.M_x*(k==0) + self.I_m*(1-(k==0))
            mmy = self.M_y*(k==0) + self.I_m*(1-(k==0))
            mma = self.am*(k==0) + self.I_m*(1-(k==0))
            mmadag = self.amdag*(k==0) + self.I_m*(1-(k==0))
            for m in np.arange(1,self.mmnum):
                mmz = np.kron(mmz,self.M_z*(k==m) + self.I_m*(1-(k==m)))
                mmx = np.kron(mmx,self.M_x*(k==m) + self.I_m*(1-(k==m)))
                mmy = np.kron(mmy,self.M_y*(k==m) + self.I_m*(1-(k==m)))
                mma = np.kron(mma,self.am*(k==m) + self.I_m*(1-(k==m)))
                mmadag = np.kron(mmadag,self.amdag*(k==m) + self.I_m*(1-(k==m)))
            self.M_zs.append(mmz)
            self.M_xs.append(mmx)
            self.M_ys.append(mmy)
            self.a_s.append(mma)
            self.adag_s.append(mmadag)
            self.ams.append(Qobj(np.kron(self.I_q, mma)))
        self.I_mm = self.I_m
        for m in np.arange(1,self.mmnum):self.I_mm = np.kron(self.I_mm,self.I_m)
        self.aqmm = Qobj(np.kron(self.aq,self.I_mm))

    def openfile(self,filename = None):
        if filename is None: 
            filename = self.filename
        return h5py.File(filename,'r')
    
    def H_rot(self):
        chis_e, chis_f,kappas,alpha,delta_c = self.hparams["chis_e"], self.hparams["chis_f"],self.hparams["kappas"],self.hparams["alpha"],self.hparams["delta_c"]
        freq, mode_freq = 0, delta_c # GHz, in lab frame
        #dekta c : detunig of cavity
     

        H0 = 0
        for ii in range(self.mmnum): # for each mode 

            chi_e_mat = chis_e[ii]*self.Q_projs[1] #chi_e |e><e|
            if self.transmon_levels>2 : chi_f_mat = chis_f[ii]*self.Q_projs[2] # ''f'''f''f'
            
            #now making Delta*a^\dagger a but have to account for all other modes being identity
            #mode_ens = np.array([2*np.pi*mm*(mode_freq - 0.5*(mm-1)*kappas[ii]) for mm in np.arange(self.mnum)]) #each level has a diff frequency (if anharmonic i guess)
            mode_ens = np.array([2*np.pi*mm*(mode_freq - 0.5*(mm-1)*0) for mm in np.arange(self.mnum)])
            H_m = np.diag(mode_ens)
            ret = H_m*(ii==0) + self.I_m*(1-(ii==0))
            for m in np.arange(1,self.mmnum):
                ret = np.kron(ret,H_m*(ii==m) + self.I_m*(1-(ii==m)))
            H0 += np.kron(self.I_q, ret)                                    #
            
            H0 += 2* np.pi*(np.kron(chi_e_mat, (self.adag_s[ii] * self.a_s[ii])))          # chi a^dag a sigma_z term
            H0 += 2* np.pi*alpha*(np.kron(chi_e_mat, (self.adag_s[ii] + self.a_s[ii])))    # constant real displacement
            H0 += 2* np.pi*(np.abs(alpha)**2)*(np.kron(chi_e_mat, (self.I_mm)))    # constant real displacement

            if self.f_state: 
                H0 += 2* np.pi*(np.kron(chi_f_mat, (self.adag_s[ii] * self.a_s[ii])))          # chi a^dag a sigma_z term
                H0 += 2* np.pi*alpha*(np.kron(chi_f_mat, (self.adag_s[ii] + self.a_s[ii])))    # constant real displacement
                H0 += 2* np.pi*(np.abs(alpha)**2)*(np.kron(chi_f_mat, (self.I_mm)))    # constant real displacement


#         if not self.add_disp_kerr:pass
#         else:
#             Hnl = 0
#             for ii,chi in enumerate(chis):
#                 Hnl+= 2*np.pi/2.0*kappas[ii]*2*alpha**3*(np.kron(self.I_q, self.M_xs[ii]))
#                 Hnl+= 2*np.pi/2.0*kappas[ii]*2*alpha**2*(np.kron(self.I_q, self.a_s[ii] @ self.a_s[ii] \
#                                                                  +self.adag_s[ii] @ self.adag_s[ii]))         
#                 Hnl+= 2*np.pi/2.0*kappas[ii]*2*alpha*(np.kron(self.I_q, self.adag_s[ii] @ self.adag_s[ii] @ self.a_s[ii] \
#                                        + self.adag_s[ii] @ self.a_s[ii] @ self.a_s[ii]))
#                 Hnl+= 2*np.pi/2.0*kappas[ii]*4*alpha**2*(np.kron(self.I_q, self.M_zs[ii])) 
#             H0 += Hnl
           
        return (H0)

    def controlHs(self):
        '''
        Returns a bunch of qubit operators tensor producted with mode indetity for each of the modes
        '''
        controlHs = []  
        
       # for each mode in cavity   #doesn't make sense, same qubit drive for each mode
#         for m in np.arange(self.mmnum):
        
        X_geI = np.kron(self.Q_sigmaXs[0], self.I_mm)
        Y_geI = np.kron(self.Q_sigmaYs[0], self.I_mm)

        controlHs.append(X_geI)
        controlHs.append(Y_geI)
       
        if self.f_state: 
            X_efI = np.kron(self.Q_sigmaXs[1], self.I_mm)
            Y_efI = np.kron(self.Q_sigmaYs[1], self.I_mm)
            controlHs.append(X_efI)
            controlHs.append(Y_efI)

        return controlHs
    
    def return_pulses(self,filename):

        fine_pulses = []
        num_ops = len(self.controlHs())
        self.f = self.openfile(filename)
        self.total_time = self.f['total_time'][()]
        self.steps = self.f['steps'][()]
        self.dt = float(self.total_time) /self.steps
        self.fine_steps = self.total_time * self.SAMPLE_RATE
        self.base_times = np.arange(self.steps + 1) * self.total_time / self.steps
        self.tlist = np.arange(self.fine_steps + 1) * self.total_time / self.fine_steps
        for i in range(num_ops):

            base_pulse = self.f['uks'][-1][i]  # final control pulse

            base_pulse = np.append(base_pulse, 0.0)  # add extra 0 on end of pulses for interpolation
            interpFun = interpolate.interp1d(self.base_times, base_pulse)
            pulse_interp = interpFun(self.tlist)
            fine_pulses.append(pulse_interp)

        return fine_pulses

    def plot_pulses(self,filename = None,plot_cavity=False):
        if filename == None: 
            filename = self.filename
        pulses = self.return_pulses(filename)
        fig, ax = plt.subplots(nrows=1, figsize=(14,4))
        if plot_cavity:end=4
        else:end = len(self.controlHs())
        labels = ["Qubit_ge_x","Qubit_ge_y","Qubit_ef_x","Qubit_ef_y", "Cavity_x","Cavity_y"] 
        for ii,x in enumerate(pulses[:end]):
            ax.plot(self.tlist/1e3,x,label = labels[ii] )
        ax.set_xlabel("Time ($\\mu$s)")
        ax.legend()
        ax.set_ylabel("Pulse amplitude (GHz)")
        return fig
    
    def total_H(self,filename):
        if self.ROTATING:

            H = [Qobj(self.H_rot())]
            controlHs = self.controlHs()
            fine_pulses = self.return_pulses(filename)

            for index in range(len(fine_pulses)):
                H.append([Qobj(controlHs[index]), fine_pulses[index]]) 
                
        else: H = 0
        
        return (H)


# Main function
    def run_optimal_control(self,state_transfer = True, initial_states = [0], target_states = [2], 
                            target_unitary = None, 
                            #initial_unitary= None,
                            total_time = 25000.0, steps = 800,max_amp = 15e-6, 
                            taylor_terms = None,is_dressed=True, 
                            convergence = {}, reg_coeffs = {},
                            plot_only_g = True,
                            states_forbidden_list = [],initial_guess = None, 
                            file_name = "test",data_path="test",specify_state_amplitudes = False, save = True):
   
        Hops = self.controlHs()
        H0 = self.H_rot()
        ops_max_amp = []
        Hnames = []
        #for ii in np.arange(self.mmnum):
        ops_max_amp.extend([max_amp*2*np.pi, max_amp*2*np.pi])
        Hnames.extend(['qubit'+str('')+'__ge_x', 'qubit'+str('')+'_ge_y'])
        if self.f_state: 
            Hnames.extend(['qubit'+str('')+'__ef_x', 'qubit'+str('')+'_ef_y'])
            ops_max_amp.extend([max_amp*2*np.pi, max_amp*2*np.pi])

        print([len(Hops), len(ops_max_amp), len(Hnames)])


        U = []
        #U0= initial_unitary
        psi0 = [] # concerned states
     
         
        blank = np.zeros(self.qnum*self.mnum**self.mmnum)
        
        # if is_dressed:
            # dressed_info = None
            # print("Computing dressed states.")
            # dressed_val, dressed_vec, dressed_id = get_dressed_info(self.H_rot())
            # print("Eigenvectors:",dressed_vec)
            # print("Eigenvalues:",dressed_val)
            # print("Map to bare states:",dressed_id)

            # dressed_info = {}
            # dressed_info['eigenvectors'] = dressed_vec
            # dressed_info['dressed_id'] = dressed_id
            # dressed_info['eigenvalues'] = dressed_val
            # dressed_info['is_dressed'] = True
            
            # if specify_state_amplitudes:
            #     for ii,initial_amps in enumerate(initial_states):
            #         initial_amps = array(initial_amps)
            #         final_amps = array(target_states[ii])
            #         g = np.zeros_like(blank)
            #         target = np.zeros_like(blank)
            #         for i_a,a in enumerate(initial_amps):
            #             g+= a*dressed_vec[dressed_id[i_a]]
            #             target += final_amps[i_a]*dressed_vec[dressed_id[i_a]] 
            #         psi0.append(g)
            #         U.append(target)
                
            # else:
                # for ii,state in enumerate(initial_states):
                #     g = dressed_vec[dressed_id[state]]
                #     target = dressed_vec[dressed_id[target_states[ii]]] 
                #     psi0.append(g)
                #     U.append(target)
            
        # else:
        dressed_info = None
            # if specify_state_amplitudes:
            #     for ii,initial_amps in enumerate(initial_states):
            #         initial_amps = array(initial_amps)
            #         final_amps = array(target_states[ii])
            #         g = np.zeros_like(blank)
            #         target = np.zeros_like(blank)
            #         for i_a,a in enumerate(initial_amps):
            #             g[i_a]= a
            #             target[i_a] = final_amps[i_a]
            #         psi0.append(g)
            #         U.append(target)
            # else:
        for ii,state in enumerate(initial_states):
            g = np.zeros_like(blank)
            g[state] = 1
            psi0.append(g)
            if state_transfer:
                target = np.zeros_like(blank)
                target[target_states[ii]] = 1
                U.append(target)
            else: 
                U = target_unitary
                psi0 = initial_states #problem in qoc code
    
        #Defining Concerned states (starting states)
        print("starting states:")
        print(psi0)
        print("target states:")
        print(U)

        # #Defining states to include in the drawing of occupation
        if plot_only_g:
            states_draw_list = np.arange(self.mnum**self.mmnum) 
            state_indices  = [arg for arg in itertools.product(np.arange(self.mnum),repeat = self.mmnum)]
  
            states_draw_names = []
            for ii in range(len(states_draw_list)):
                s = ''
                for a in state_indices[ii]:s+=str(a)
                states_draw_names.append('g_' + s)
        # states_draw_list = None
        # states_draw_names = None


        states_forbidden_list = states_forbidden_list
        #print(states_forbidden_list)

        ss = Grape(H0, Hops, Hnames, U, total_time, steps, psi0, convergence=convergence,
                            # U0 = U0, 
                             draw=[states_draw_list, states_draw_names], state_transfer=state_transfer, use_gpu=False,
                             sparse_H=False, show_plots=True, Taylor_terms=taylor_terms, method='Adam', initial_guess=initial_guess,
                             maxA=ops_max_amp, reg_coeffs=reg_coeffs, dressed_info=dressed_info, 
                             file_name=file_name, data_path=data_path, save = save)
        self.filename = ss[-1]
        return ss

    def plot_optimal_control(self,scales = [4367,4367,81.1684054679128, 81.1684054679128],pad_FFT = 3,filename = None,lim_scale=1.0):
        
        if filename is None: 
            filename = self.filename
        
        a = self.openfile(filename)


        update_step = a['convergence']['update_step'][()]
        steps = a['steps'][()]
        print ("steps = ",steps)
        dt = a['total_time'][()] / steps

        fig = plt.figure(figsize=(14,4*6))
        ax = fig.add_subplot(611,title = 'Control fields in GHz')
        circle_grape_labels = ['qubit_x','qubit_y']
        
        for i in range(len(a['Hops'][()])):
            ax.plot(np.arange(0, steps) * dt/1e3, a['uks'][-1][i])
        plt.legend(labels=[circle_grape_labels[ii] for ii in range(len(a['Hnames']))], prop={'size': 12})
        ax.plot(np.arange(0, steps) * dt/1e3, np.zeros(steps), '--k')  # plot a line y=0 for reference
        ax.set_xlabel("Time ($\mu$s)")
        ax.set_ylabel("$\\xi$ ($2\pi$ x GHz)")
        

        ax = fig.add_subplot(612,title = 'Control fields in AWG amplitudes')
    
        # plot control pulses, last element is the final control pulse output
        for i in range(len(a['Hops'][()])):
            ax.plot(np.arange(0, steps) * dt/1e3, a['uks'][-1][i] * scales[i])

        ax.plot(np.arange(0, steps) * dt/1e3, sqrt(a['uks'][-1][0]**2 + a['uks'][-1][1]**2) * scales[0],'k-')
        plt.legend(labels=[a['Hnames'][ii] for ii in range(len(a['Hnames']))], prop={'size': 12})
        ax.plot(np.arange(0, steps) * dt/1e3, np.zeros(steps), '--k')  # plot a line y=0 for reference
        
        ax.set_xlabel("Time ($\mu$s)")
        ax.set_ylabel("Amplitude")

        # plot Fourier transform of control pulses
        ax2 = fig.add_subplot(613)
        for i in range(len(a['Hops'][()])):
            ax2.plot(fftfreq((2*pad_FFT+1)*steps, d=dt), np.abs(fft(pad(a['uks'][-1][i],(pad_FFT*steps,pad_FFT*steps),'constant'))),'.-', label=a['Hnames'][i])
        
        plt.legend(labels=[a['Hnames'][ii] for ii in range(len(a['Hnames']))], prop={'size': 12})
        ax2.set_xlabel("Freq (GHz)")
        ax2.set_ylabel("FFT")
        limits = abs(self.hparams["chi"]*2*lim_scale)
        ax2.axvline(abs(self.hparams['omega']),color='r',linestyle='dashed')
        for ii in range(5):
            ax2.axvline(abs(2*self.hparams['chi'])*ii,color='b',linestyle='dashed')
        ax2.set_xlim(0,limits)


        ax3 = fig.add_subplot(614)
        ax3.semilogy(np.arange(0, len(a['error'])) * 5, a['error'])
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Error")
        
        print("Minimum error", a['error'][-1])
        print("Number of taylor terms",a['taylor_terms'][()])
        
        ax4 = fig.add_subplot(615,title = "Cavity level populations")
        for i in range(len(a['inter_vecs_mag_squared'][0][0])):
            # second index picks which starting state to look at evolution of
            ax4.plot(np.arange(0, steps + 1) * dt/1e3, a['inter_vecs_mag_squared'][-1][0][i])
        ax4.set_xlabel("Time ($\mu$s)")
        ax4.set_ylabel("Populations")       
       
        plt.tight_layout()
        
        
    def qutip_mesolve_new(self,start_state,filename = None):
        if filename is None: 
            filename = self.filename
        ss = np.zeros(self.qnum*(self.mnum)**self.mmnum,dtype=complex)
        ss[:len(start_state)] = start_state  # g0
        psi0 = Qobj(ss)
        rho0 = psi0*psi0.dag()
        
        
        H = self.total_H(filename)
        tlist = self.tlist
#         if self.t1params is None:
#             c_ops = []
#         else:
#             gamma = 1/self.t1params['T1_q']*1e-3
#             gamma_phi = 1/self.t1params['T2_q']*1e-3- 1/2/self.t1params['T1_q']*1e-3
#             n_thq =  self.t1params['nth_q']
#             c_ops = gamma*(1+ n_thq)*lindblad_dissipator(self.aqmm) + gamma*(n_thq)*lindblad_dissipator(self.aqmm.dag())
#             c_ops += gamma_phi*lindblad_dissipator(self.aqmm.dag()*self.aqmm) 
#             kappa_ms = 1/array(self.t1params['T1_ms'])*1e-3
#             n_thms =  array(self.t1params['nth_ms'])
#             for ii,a in enumerate(self.ams):
#                 c_ops += kappa_ms[ii]*(1+n_thms[ii])*lindblad_dissipator(a) +  kappa_ms[ii]*(n_thms[ii])*lindblad_dissipator(a.dag()) 

           
#         H0 = Qobj(self.H_rot())
#         e_vecs = H0.eigenstates()[1]
#         self.e_ops = [e_vec*e_vec.dag() for e_vec in e_vecs]
#         self.n_mms = array([[expect(Qobj(np.kron(self.I_q,self.M_zs[ii])),e_vec) for e_vec in e_vecs] for ii in np.arange(self.mmnum)])
#         self.n_qs = array([expect(Qobj(np.kron(self.Q_z,self.I_mm)), e_vec) for e_vec in e_vecs])
#         self.nqinit = expect(Qobj(np.kron(self.Q_z,self.I_mm)), psi0)
#         self.nmminit = array([expect(Qobj(np.kron(self.I_q,self.M_zs[ii])),psi0) for ii in np.arange(self.mmnum)])
        
        nsteps = 1e+4
        opts = Options(store_states=True, store_final_state=True, nsteps = nsteps)
        out = mesolve(H, rho0, tlist, options =opts)
        return tlist, out    
        
    def qutip_mesolve(self,start_state,filename = None):
        if filename is None: 
            filename = self.filename
        ss = np.zeros(self.qnum*(self.mnum)**self.mmnum,dtype=complex)
        ss[:len(start_state)] = start_state  # g0
        print(ss[:len(start_state)])
        psi0 = Qobj(ss)
        rho0 = psi0*psi0.dag()
        
        
        H = self.total_H(filename)
        tlist = self.tlist
        if self.t1params is None:
            c_ops = []
        else:
            gamma = 1/self.t1params['T1_q']*1e-3
            gamma_phi = 1/self.t1params['T2_q']*1e-3- 1/2/self.t1params['T1_q']*1e-3
            n_thq =  self.t1params['nth_q']
            c_ops = gamma*(1+ n_thq)*lindblad_dissipator(self.aqmm) + gamma*(n_thq)*lindblad_dissipator(self.aqmm.dag())
            c_ops += gamma_phi*lindblad_dissipator(self.aqmm.dag()*self.aqmm) 
            kappa_ms = 1/array(self.t1params['T1_ms'])*1e-3
            n_thms =  array(self.t1params['nth_ms'])
            for ii,a in enumerate(self.ams):
                c_ops += kappa_ms[ii]*(1+n_thms[ii])*lindblad_dissipator(a) +  kappa_ms[ii]*(n_thms[ii])*lindblad_dissipator(a.dag()) 

           
        H0 = Qobj(self.H_rot())
        e_vecs = H0.eigenstates()[1]
        self.e_ops = [e_vec*e_vec.dag() for e_vec in e_vecs]
        self.n_mms = array([[expect(Qobj(np.kron(self.I_q,self.M_zs[ii])),e_vec) for e_vec in e_vecs] for ii in np.arange(self.mmnum)])
        self.n_qs = array([expect(Qobj(np.kron(self.Q_z,self.I_mm)), e_vec) for e_vec in e_vecs])
        self.nqinit = expect(Qobj(np.kron(self.Q_z,self.I_mm)), psi0)
        self.nmminit = array([expect(Qobj(np.kron(self.I_q,self.M_zs[ii])),psi0) for ii in np.arange(self.mmnum)])
        
 
        out = mesolve(H, rho0, tlist, c_ops=c_ops,e_ops = self.e_ops)
        return tlist, out
    
    def plot_mesolve(self,filename = None,show_low_only=True,MAX = 2,start_state = [1,0,0,0,0], title = ''):
        if filename is None: 
            filename = self.filename
        
        print("running mesolve for rotating frame")
        
        tlist_rot, out = self.qutip_mesolve(start_state,filename)
        pops= [out.expect[ii] for ii in arange(len(self.e_ops))]
        cutoff = self.qnum*(self.mnum)**self.mmnum
        
        fig, ax = plt.subplots(nrows=1, figsize=(14,6))
        if int(around(self.nqinit,0)) == 0:
            title = "$\\psi_0$ = |g,"
        else:label = "|e,"
        for mm in arange(self.mmnum):
            title+=str(int(self.nmminit[mm]))
            if mm == self.mmnum-1:title+='>'
            else:title+=','
        ax.set_title(label=title)
        for num in range(len(pops)):
            if around(self.n_qs[num],0) == 0:label = '|g,'
            elif around(self.n_qs[num],0) == 1:label = '|e,'
            else:label = '|g/e,'  

            
            for mm in arange(self.mmnum):
                label+=str(int(around(self.n_mms[mm][num],0)))
                if mm == self.mmnum-1:label+='>'
                else:label+=','
                    
            if show_low_only:
                if self.n_qs[num] + sum(array(self.n_mms),axis=0)[num]>MAX:
                    label = None
            

            ax.plot(tlist_rot,pops[num], label=label)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Populations")
            ax.legend(prop={'size': 12}, loc=2)    
    
print ('done')