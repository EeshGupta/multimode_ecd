from DECD_pulseV2 import *
from qutip import *
import numpy as np

#V3 : changed how angles are loaded 
#V4: Adding time dependence to collapse operator decay rates

class ecd_pulse_two_mode: 
    
    def __init__(self, 
                 param_file = None, 
                 storage1_params = None,
                 storage2_params = None, 
                 betas = None, 
                 gammas = None,
                 phis = None, 
                 thetas = None,
                 qubit_params = None, 
                 alpha_CD =30, 
                 kappa1 = 0.5e-6, # T1 for both modes is 2ms
                 kappa2 = 0.5e-6,
                 buffer_time = 4): 
        '''
        betas, thetas, phis : ecd parameters
        n_q : # of levels in the qubit
        n_c1 : # of levels in mode 1 of the cavity
        n_c2 : # of levels in mode 2 of the cavity 
        
        storage_params : {
                            "chi_kHz": -33, #dispersive shift
                            "chi_prime_Hz": 0, #second order dispersive shift
                            "Ks_Hz": 0, #Kerr correction not yet implemented.
                            "epsilon_m_MHz": 400, #largest oscillator drive amplitude in MHz (max|epsilon|)
                            "unit_amp": 0.01, #DAC unit amp of gaussian displacement to alpha=1.
                            "sigma": 11, #oscillator displacement sigma
                            "chop": 4, #oscillator displacement chop (number of stds. to include in gaussian pulse)
                        }
                        
        qubit_params : {'unit_amp': 0.5, 'sigma': 6, 'chop': 4} #parameters for qubit pi pulse.
        '''
        #self.param_file = param_file # for loading parameters
        self.betas = betas
        self.gammas = gammas
        self.phis = phis
        self.thetas = thetas
        #self.load_params()
        
        self.kappa1 = kappa1
        self.kappa2 = kappa2
        
        #Pulse parameters

        ## modes
        self.storage1_params = storage1_params
        if storage1_params == None: 
            self.storage1_params = {
                            "chi_kHz": -33, 
                            "chi_prime_Hz": 0, 
                            "Ks_Hz": 0,
                            "epsilon_m_MHz": 400, 
                            "unit_amp": 0.01, 
                            "sigma": 11, 
                            "chop": 4, 
                        }
        self.storage2_params = storage2_params
        if storage2_params == None: 
            self.storage2_params = {
                            "chi_kHz": -33, 
                            "chi_prime_Hz": 0, 
                            "Ks_Hz": 0,
                            "epsilon_m_MHz": 400, 
                            "unit_amp": 0.01, 
                            "sigma": 11, 
                            "chop": 4, 
                        }
            
        self.qubit_params = qubit_params
        if self.qubit_params == None: 
            self.qubit_params = {'unit_amp': 0.5,
                                 'sigma': 6, 
                                 'chop': 4} 
        self.storage1 = None
        self.storage2 = None
        self.qubit = None
        
        
        #parameters obtained after get_pulse() is called
        self.cavity1_dac_pulse_GHz = None
        self.cavity2_dac_pulse_Ghz = None
        self.qubit_dac_pulse_GHz = None
        self.alpha = None 
        
        #other keyword params
        self.alpha_CD = alpha_CD
        self.buffer_time = buffer_time
        
        
    
    ####
    import cmath
    ####
    def load_params(self): 
        '''
        Loads betas, thetas, phis
        '''
        params = np.loadtxt(self.param_file)
        self.betas = np.asarray([complex(params[0][i], params[1][i]) for i in range(len(params[0]))])
        self.gammas =  np.asarray([complex(params[2][i], params[3][i]) for i in range(len(params[0]))])
        self.phis = params[4]
        self.thetas = params[5]
        return None
    
    def get_pulses(self): 
        '''
        Evaluates cavity and qubit pulses for the desired ECD simulation
        '''
        #Creates objects
        self.storage1 = FakeStorage(**self.storage1_params)
        self.storage2 = FakeStorage(**self.storage2_params)
        self.qubit = FakeQubit(**self.qubit_params)
        
        #Qubit pi pulse stuff ... calculating conversion between qubit DAC units and MHz (Omega)
        pi = rotate(np.pi, phi=0, sigma=self.qubit_params['sigma'], chop=self.qubit_params['chop'], dt=1)
        Omega_m = np.real(np.max(pi))/self.qubit_params['unit_amp']
        
        #get pulses
        pulse_dict = conditional_displacement_circuit(self.betas,
                                                     self.gammas,
                                                     self.phis,
                                                     self.thetas,
                                                     self.storage1, 
                                                     self.storage2, 
                                                     self.qubit,
                                                     self.alpha_CD,
                                                     self.alpha_CD ,
                                                     buffer_time=self.buffer_time, 
                                                     kerr_correction = False, 
                                                     kappa1 = self.kappa1,
                                                     kappa2 = self.kappa2,
                                                     chi_prime_correction=True, 
                                                     final_disp=True, 
                                                     pad=True)
        cavity1_dac_pulse, cavity2_dac_pulse, qubit_dac_pulse, = pulse_dict['cavity1_dac_pulse'], pulse_dict['cavity2_dac_pulse'], pulse_dict['qubit_dac_pulse']
    
        #Dac units to Ghz conversion

        self.cavity1_dac_pulse_GHz = (2*np.pi)*(10**(-3))*self.storage1.epsilon_m_MHz*cavity1_dac_pulse #convert from DAC to Mhz to Ghz
        self.cavity2_dac_pulse_GHz = (2*np.pi)*(10**(-3))*self.storage2.epsilon_m_MHz*cavity2_dac_pulse #convert from DAC to Mhz to Ghz
        self.qubit_dac_pulse_GHz = (2*np.pi)*10**(-3)*self.qubit.Omega_m_MHz*qubit_dac_pulse #convert from DAC to Mhz to Ghz

        #alpha:  Solve equation of motion and get corresponding displacements for displaced frame simulations
        self.alpha1 = self.alpha_from_epsilon_nonlinear_finite_difference(
            epsilon_array =self.cavity1_dac_pulse_GHz , delta=0, kappa = self.kappa1, alpha_init=0 + 0j)
        
        self.alpha2 = self.alpha_from_epsilon_nonlinear_finite_difference(
            epsilon_array =self.cavity2_dac_pulse_GHz , delta=0, kappa = self.kappa2, alpha_init=0 + 0j)
        
        return None
    
    def alpha_from_epsilon_nonlinear_finite_difference(
        self, epsilon_array, delta=0, kappa = 0, alpha_init=0 + 0j):
        dt = 1
        alpha = np.zeros_like(epsilon_array)
        alpha[0] = alpha_init
        alpha[1] = alpha_init
        for j in range(1, len(epsilon_array) - 1):
            alpha[j + 1] = (
                    2*
                     dt
                    * (
                        -1j * delta# * alpha[j]
                        #- 2j * Ks * np.abs(alpha[j]) ** 2 * alpha[j]
                        - (kappa / 2.0) * alpha[j]
                        - 1j * epsilon_array[j]
                    )
                    + alpha[j - 1])
        return alpha
    

    #####
    import matplotlib.pyplot as plt
    #####
    def plot_pulses(self): 
        '''
        Plots cavity dac pulse, the resultant displacement of cavity and the qubit pulse
        '''
        fig, axs = plt.subplots(5,1)
        axs[0].plot(np.real(self.cavity1_dac_pulse_GHz))
        axs[0].plot(np.imag(self.cavity1_dac_pulse_GHz))
        axs[0].set_ylabel('cavity 1 dac pulse (Ghz)', fontsize = 10)
        axs[1].plot(np.real(self.alpha1))
        axs[1].plot(np.imag(self.alpha1))
        axs[1].set_ylabel('alpha 1', fontsize = 10)

        axs[2].plot(np.real(self.cavity2_dac_pulse_GHz))
        axs[2].plot(np.imag(self.cavity2_dac_pulse_GHz))
        axs[2].set_ylabel('cavity 2 dac pulse (Ghz)', fontsize = 10)
        axs[3].plot(np.real(self.alpha2))
        axs[3].plot(np.imag(self.alpha2))
        axs[3].set_ylabel('alpha 2', fontsize = 10)
            
        axs[4].plot(np.real(self.qubit_dac_pulse_GHz))
        axs[4].plot(np.imag(self.qubit_dac_pulse_GHz))
        axs[4].set_ylabel('qubit dac pulse (GHz)', fontsize = 10)
        plt.xlabel('ns')
        
        return None
    
    
####################################################################################
#############     Qutip Now   ######################################################
class qutip_sim_two_mode:
    
    def __init__(self, n_q, n_c1, n_c2, chi = None,
                 alpha1 = [], alpha2 = [], qubit_pulse = [],
                 sim_params = None, save_states = False, states_filename = 'states store'):
        '''
        n_q, n_c = # of levels in qubit, cavity
        Assumes n_c1 = n_c2
        
        sim_params = sim_params = {'bare_qubit_mode_coupling':0,
        'Stark Shift': 0,  'transmon_relax': 0,'transmon_dephasing': 0,
        'cavity_relax': 0,'cavity_dephasing': 0}  # 0 means false, 1 means true or activates    
        '''
        self.n_q = n_q
        self.n_c1 = n_c1
        self.n_c2 = n_c2
        
        self.chi1 = -33*2*np.pi*(10**(-6)) # Alec's params
        self.chi2 = self.chi1
        
        ##get pulses
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.qubit_pulse = qubit_pulse
        
        
        self.get_basic_ops()
        
            
        #get operators
        self.H0 = tensor(self.identity_q, self.identity_c, self.identity_c) # time independent part
        ## drive terms (ecd pulses)
        self.Hd = [ #qubit terms
                 [tensor(self.a_q, self.identity_c, self.identity_c), np.conjugate(self.qubit_pulse)], 
                 [tensor(self.adag_q, self.identity_c, self.identity_c), self.qubit_pulse],
                 # ecd pulses
                 ## mode 1
                 [(self.chi1/2)*tensor(sigmaz(), self.a_c, self.identity_c), np.conjugate(self.alpha1)],
                 [(self.chi1/2)*tensor(sigmaz(), self.adag_c, self.identity_c), self.alpha1],
                 ## mode2
                 [(self.chi2/2)*tensor(sigmaz(),  self.identity_c, self.a_c), np.conjugate(self.alpha2)],
                 [(self.chi2/2)*tensor(sigmaz(),  self.identity_c, self.adag_c), self.alpha2],
                 ] 
        self.c_ops = []
        
        self.states_filename = states_filename
        self.save_states= save_states
        
    
    def get_basic_ops(self): 
        '''
        Creates identity, creation/annihilation for qubit/cavity

        Assumption n_c1 = n_c2
        '''
        self.identity_q = qeye(self.n_q)
        self.identity_c = qeye(self.n_c1)

        self.a_q = destroy(self.n_q)
        self.a_c = destroy(self.n_c1)

        self.adag_q = create(self.n_q)
        self.adag_c = create(self.n_c1)

        self.num_q = num(self.n_q)
        self.num_c =  num(self.n_c1)
        
        return None
    
    def square_list(self, listy):
        '''
        Input: List of numbers [a,b,c,...]
        Output: [a^2, b^2, ...] (norm squared for complex numbers)
        '''
        return np.real( [np.real(i)**2 + np.imag(i)**2 for i in listy])
    
    def add_bare_qubit_mode_coupling(self):
        '''
        Add the basic dispersive shift term to Ham
        '''
        self.H0 +=  (self.chi1/2)*tensor(sigmaz(), self.num_c, self.identity_c)
        self.H0 +=  (self.chi2/2)*tensor(sigmaz(), self.identity_c, self.num_c,)
        return None

    def add_mode_mode_coupling(self, eta = None):
        '''
        Add mode mode coupling term or cross kerr interaction

        cross kerr = sqrt(kerr_mode1  * kerr_mode2) = chi1 * chi2 /anharmonicity of qubit
        '''
        eta = self.chi1*self.chi2/150
        self.H0+= eta*tensor(self.identity_q, self.num_c, self.num_c)
    
    def add_stark_shift(self):
        '''
        Add the basic dispersive shift term to Ham
        '''
        term1 = [(self.chi1/2)*tensor(sigmaz(), self.identity_c, self.identity_c), self.square_list(self.alpha1)]
        term2 = [(self.chi2/2)*tensor(sigmaz(), self.identity_c, self.identity_c), self.square_list(self.alpha2)]
        self.Hd.append(term1)
        self.Hd.append(term2)
        
        return None
    
    def add_qubit_relaxation(self, T1 = 30e+3):
        '''
        qubit relaxation (T1 in nanoseconds)
        '''
        gamma_relax = 1/T1
        term = np.sqrt(gamma_relax/2)*tensor(self.a_q, self.identity_c, self.identity_c)
        self.c_ops.append(term)
        return None
    
    def add_qubit_dephasing(self, T1 = 30e+3, Techo = 50e+3):
        '''
        qubit relaxation (T1, T2 in nanoseconds)
        '''
        gamma_relax = 1/T1
        gamma_echo = 1/Techo
        gamma_phi = gamma_echo - (gamma_relax/2)
        #print(gamma_phi)
        
        term = np.sqrt(gamma_phi)*tensor(self.num_q, self.identity_c, self.identity_c)
        self.c_ops.append(term)
        return None
    
    def add_cavity_relaxation(self, T1_mode1 = 10e+6, T1_mode2 = 10e+6):
        '''
        qubit relaxation (T1 in nanoseconds)
        
        in displaced frame, the a_c -> a_c + alpha but the alpha part
        can be taken care of by using classical EOM when designing pulses
        '''
        gamma_relax_mode1 = 1/T1_mode1
        term1 = np.sqrt(gamma_relax_mode1/2)*tensor(self.identity_q, self.a_c, self.identity_c)

        gamma_relax_mode2 = 1/T1_mode2
        term2 = np.sqrt(gamma_relax_mode2/2)*tensor(self.identity_q, self.identity_c, self.a_c)


        self.c_ops.append(term1)
        self.c_ops.append(term2)
        return None
    
    def add_cavity_dephasing_for_given_mode(self, T1, Techo, alpha, mode_index = 1):
        '''
        Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
        '''
        gamma_relax= 1/T1
        gamma_echo = 1/Techo
        gamma_phi = gamma_echo - (gamma_relax/2)
        
        
        #In transforming a -> a + alpha, the term a^dag a can be broken down as 
        # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
        # the latter term can be ignored cuz equal to identity
        if mode_index == 1: 
            term1 = gamma_phi*tensor(self.identity_q, self.num_c, self.identity_c)
            term2 = gamma_phi*tensor(self.identity_q, self.a_c  , self.identity_c)
            term3 = gamma_phi*tensor(self.identity_q, self.adag_c  , self.identity_c)
        else: #mode index = 2
            term1 = gamma_phi*tensor(self.identity_q, self.identity_c, self.num_c)
            term2 = gamma_phi*tensor(self.identity_q, self.identity_c, self.a_c )
            term3 = gamma_phi*tensor(self.identity_q, self.identity_c, self.adag_c)
        
        #add to collapse operator list
        self.c_ops.append(term1)
        self.c_ops.append([term2, np.conjugate(alpha)]) # adding the time dependent coefficients
        self.c_ops.append([term3, alpha])
        
        return None
        
    
    def add_cavity_dephasing(self, T1_mode1 = 10e+6, Techo_mode1 = 10e+6, T1_mode2 = 10e+6, Techo_mode2 = 10e+6 ):
        '''
        qubit dephasing (T1, Techo in nanoseconds)
        '''
       
        self.add_cavity_dephasing_for_given_mode( T1_mode1, Techo_mode1, self.alpha1, mode_index = 1)
        self.add_cavity_dephasing_for_given_mode( T1_mode2, Techo_mode2, self.alpha2, mode_index = 2)        
        return None
    
    
    def me_solve(self, nsteps = 10000, initial = None): 
        '''
        Solve the master equation 
        '''
        T = len(self.alpha1) # total time length in nanoseconds
        t_list = np.linspace(0, T, T)

        if initial == None: 
            initial = tensor(basis(self.n_q,0), basis(self.n_c1,0), basis(self.n_c2,0))
            
        opts = Options(store_states=True, store_final_state=True, nsteps = nsteps)#, max_step = 10)
        ## since smallest pulse is a pi pulse which is 40 ns long
        
        #Hamiltonian
        self.H = [self.H0]
        for i in self.Hd:
             self.H.append(i)
            
        self.output = mesolve(self.H, initial , t_list, self.c_ops, [], options =opts)        
        if self.save_states: 
            qsave(self.output.states, self.states_filename)
        
        return None
    
    def dot(self, state1, state2):
        '''
        dotting both states
        '''
        fid = state1.overlap(state2)
        return fid*np.conjugate(fid)

    
    def get_fidelity(self, target): 
        '''
        dot final state after evolution with target
        '''
        return self.dot(self.output.states[-1], target)
    
    
    def plot_populations(self, figname = 'figure'):
        '''
        Given output of mesolve, outputs populations with qubit as ground
        '''
#         if self.save_states:
#             output_states = qload(self.states_filename)
        output_states = self.output.states
        
        
        fig, axs = plt.subplots(2,1, figsize=(10,8))
        probs = []
        times = [k/1000 for k in range(len(output_states))]
        max_num_levels = 3 # to be shown on the plot
        
        #qubit grounded
        for i in range(max_num_levels):
            for j in range(max_num_levels):
                target = tensor(basis(self.n_q,0), basis(self.n_c1, i), basis(self.n_c2, j))
                pops = []
                for k in range(len(output_states)): 
                    z = self.dot(target ,output_states[k])
                    pops.append(z)
                axs[0].plot(times, pops, label = '|g,'+str(i)+',' + str(j)+'>')
        
        #qubit excited
        for i in range(max_num_levels):
            for j in range(max_num_levels):
                target = tensor(basis(self.n_q,1), basis(self.n_c1, i), basis(self.n_c2, j))
                pops = []
                for k in range(len(output_states)): 
                    z = self.dot(target ,output_states[k])
                    pops.append(z)
                axs[1].plot(times, pops, linestyle = '--',  label = '|e,'+str(i)+',' + str(j)+'>')
                
        axs[1].set_xlabel(r"Time ($\mu$s)", fontsize = 18)
        axs[1].set_ylabel("Populations", fontsize = 18)
        axs[0].set_ylabel("Populations", fontsize = 18)
        axs[0].tick_params(axis = 'both', which = 'major', labelsize = '15')
        axs[0].tick_params(axis = 'both', which = 'minor', labelsize = '15')
        axs[1].tick_params(axis = 'both', which = 'major', labelsize = '15')
        axs[1].tick_params(axis = 'both', which = 'minor', labelsize = '15')
    #     axs[0].set_xticks(fontsize= 10)
    #     axs[1].set_yticks(fontsize= 10)
    #     axs[0].set_yticks(fontsize= 10)
    #     plt.legend(prop={'size': 20},  fontsize = 8, loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=3, fancybox=True, shadow=True)   
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        #plt.legend(fontsize = '15')
        #fig.suptitle(title, fontsize = 15)
        plt.tight_layout()
        fig.savefig(figname, dpi = 1000)
        return None
