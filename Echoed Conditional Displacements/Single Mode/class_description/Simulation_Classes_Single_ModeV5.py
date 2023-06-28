from Simplified_ECD_pulse_constructionV2 import *
import numpy as np
import h5py as hf
# V3: corrected dephaing term to have time dependent amplitude.
# V4: Add thermal noise
#print('hi')
class ecd_pulse_single_mode: 
    
    def __init__(self, 
                 param_file = None, 
                 storage_params = None, 
                 qubit_params = None, 
                 kappa = 0,#0.5e-6, #T1 = 2 ms
                 alpha_CD =30, 
                 buffer_time = 4): 
        '''
        betas, thetas, phis : ecd parameters
        n_q : # of levels in the qubit
        n_c : # of levels in the cavity 
        
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
        self.param_file = param_file # for loading parameters
        self.betas = None
        self.phis = None
        self.thetas = None
        
        self.kappa = kappa
        self.load_params()
        
        #Pulse parameters
        self.storage_params = storage_params
        if storage_params == None: 
            self.storage_params = {
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
        self.storage = None
        self.qubit = None
        
        
        #parameters obtained after get_pulse() is called
        self.cavity_dac_pulse_GHz = None
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
        print(self.param_file)
        if type(self.param_file) is not str: 
            self.betas, self.phis, self.thetas = None, None, None
            return None
        
        ## Text file code
        params = np.loadtxt(self.param_file)
        self.betas = np.asarray([complex(params[0][i], params[1][i]) for i in range(len(params[0]))])
        self.phis = params[2]
        self.thetas = params[3]

        ## H5 file format
#         filename = self.param_file
#         file = hf.File(filename, "r")
#         timestamp = list(file.keys())[-1]
#         fids = file[timestamp]['fidelities'][-1]
#         print('fidelity for h5 param is ' + str(max(fids)))
#         best_fid_idx = np.argmax(fids)
#         print('index of fidelity for h5 param is ' + str(best_fid_idx))
#         self.betas = file[timestamp]['betas'][-1][best_fid_idx][0]
#         #self.gammas = file[timestamp]['gammas'][-1][best_fid_idx]
#         self.phis = file[timestamp]['phis'][-1][best_fid_idx]
#         #bug in MECD code
#         (m,n) = self.phis.shape
#         for m_ in range(m):
#             for n_ in range(n): 
#                 self.phis[m_,n_] = self.phis[m_,n_] - (np.pi/2)
#         self.phis = self.phis[0]
#         self.thetas = file[timestamp]['thetas'][-1][best_fid_idx][0]
        return None
    
    def get_pulses(self): 
        '''
        Evaluates cavity and qubit pulses for the desired ECD simulation
        '''
        #Creates objects
        self.storage = FakeStorage(**self.storage_params)
        self.qubit = FakeQubit(**self.qubit_params)
        
        #Qubit pi pulse stuff ... calculating conversion between qubit DAC units and MHz (Omega)
        pi = rotate(np.pi, phi=0, sigma=self.qubit_params['sigma'], chop=self.qubit_params['chop'], dt=1)
        Omega_m = np.real(np.max(pi))/self.qubit_params['unit_amp']
        
        #get pulses
        pulse_dict = conditional_displacement_circuit(self.betas, self.phis, 
                                                      self.thetas, 
                                                      self.storage, 
                                                      self.qubit,
                                                      self.alpha_CD,
                                                      buffer_time = self.buffer_time, 
                                                      kerr_correction = False, 
                                                      chi_prime_correction=True,
                                                      kappa = self.kappa,
                                                      final_disp=True, 
                                                      pad=True)
        
        
        cavity_dac_pulse, qubit_dac_pulse, = pulse_dict['cavity_dac_pulse'], pulse_dict['qubit_dac_pulse']

        # conversions
        self.cavity_dac_pulse_GHz = (2*np.pi)*(10**(-3))*self.storage.epsilon_m_MHz*cavity_dac_pulse #convert from DAC to Mhz to Ghz
        self.qubit_dac_pulse_GHz = (2*np.pi)*10**(-3)*self.qubit.Omega_m_MHz*qubit_dac_pulse #convert from DAC to Mhz to Ghz
        #alpha
        self.get_alpha()
        
        return None
    
    def alpha_from_epsilon_nonlinear_finite_difference(
        self, epsilon_array, delta=0, alpha_init=0 + 0j):
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
                        - (self.kappa / 2.0) * alpha[j]
                        - 1j * epsilon_array[j]
                    )
                    + alpha[j - 1])
        return alpha
    
    def get_alpha(self):
        '''
        Solve equation of motion and get corresponding displacements for displaced frame simulations
        '''
        self.alpha = self.alpha_from_epsilon_nonlinear_finite_difference(
            epsilon_array =self.cavity_dac_pulse_GHz , delta=0, alpha_init=0 + 0j)
        return None
    
    #####
    import matplotlib.pyplot as plt
    #####
    def plot_pulses(self): 
        '''
        Plots cavity dac pulse, the resultant displacement of cavity and the qubit pulse
        '''
        fig, axs = plt.subplots(3,1)
        axs[0].plot(np.real(self.cavity_dac_pulse_GHz))
        axs[0].plot(np.imag(self.cavity_dac_pulse_GHz))
        axs[0].set_ylabel('cavity dac pulse (Ghz)', fontsize = 10)
        axs[1].plot(np.real(self.alpha))
        axs[1].plot(np.imag(self.alpha))
        axs[1].set_ylabel('alpha', fontsize = 10)
            
        axs[2].plot(np.real(self.qubit_dac_pulse_GHz))
        axs[2].plot(np.imag(self.qubit_dac_pulse_GHz))
        axs[2].set_ylabel('qubit dac pulse (GHz)', fontsize = 10)
        plt.xlabel('ns')
        
        return None
    
    
####################################################################################
#############     Qutip Now   ######################################################
from qutip import *

class qutip_sim_single_mode:
    
    def __init__(self, n_q, n_c, chi = None,
                 alpha = [], qubit_pulse = [],
                 T1qubit = 30e+3, # in mu s                
                 sim_params = None, save_states = False, states_filename = 'states store'):
        '''
        n_q, n_c = # of levels in qubit, cavity
        
        sim_params = sim_params = {'bare_qubit_mode_coupling':0,
        'Stark Shift': 0,  'transmon_relax': 0,'transmon_dephasing': 0,
        'cavity_relax': 0,'cavity_dephasing': 0}  # 0 means false, 1 means true or activates    
        '''
        self.n_q = n_q
        self.n_c = n_c
        
        self.chi = -33*2*np.pi*(10**(-6)) # Alec's params
        self.T1qubit =T1qubit
        
        ##get pulses
        self.alpha = alpha
        self.qubit_pulse = qubit_pulse
        
        self.sim_params = sim_params
        if sim_params == None: 
            sim_params =  {'bare_qubit_mode_coupling':0,
        'Stark Shift': 0,  'transmon_relax': 0,'transmon_dephasing': 0,
        'cavity_relax': 0,'cavity_dephasing': 0}
        self.get_basic_ops()
        
            
        #get operators
        self.H0 = tensor(self.identity_q, self.identity_c) # time independent part
        ## drive terms (ecd pulses)
        self.Hd = [ #qubit terms
                 [tensor(self.a_q, self.identity_c), np.conjugate(self.qubit_pulse)], 
                 [tensor(self.adag_q, self.identity_c), self.qubit_pulse],
                 # ecd pulses
                 [(self.chi/2)*tensor(sigmaz(), self.a_c), np.conjugate(self.alpha)],
                 [(self.chi/2)*tensor(sigmaz(), self.adag_c), self.alpha],
                 ] 
        self.add_bare_qubit_mode_coupling()
        self.add_stark_shift()
        self.c_ops = []
        
        self.save_states = save_states
        self.states_filename = states_filename
        
    
    def get_basic_ops(self): 
        '''
        Creates identity, creation/annihilation for qubit/cavity
        '''
        self.identity_q = qeye(self.n_q)
        self.identity_c = qeye(self.n_c)

        self.a_q = destroy(self.n_q)
        self.a_c = destroy(self.n_c)

        self.adag_q = create(self.n_q)
        self.adag_c = create(self.n_c)

        self.num_q = num(self.n_q)
        self.num_c =  num(self.n_c)
        
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
        self.H0 +=  (self.chi/2)*tensor(sigmaz(), self.num_c)
        return None
    
    def add_stark_shift(self):
        '''
        Add the basic dispersive shift term to Ham
        '''
        term = [(self.chi/2)*tensor(sigmaz(), self.identity_c), self.square_list(self.alpha)]
        self.Hd.append(term)
        return None
    
    def add_qubit_relaxation(self, T1 = 30e+3):
        '''
        qubit relaxation (T1 in nanoseconds)
        '''
        gamma_relax = 2*np.pi*(1/T1)
        term = np.sqrt(gamma_relax/2)*tensor(self.a_q, self.identity_c)
        self.c_ops.append(term)
        return None
    
    def add_qubit_dephasing(self, T1 = 30e+3, Techo = 50e+3):
        '''
        qubit relaxation (T1, T2 in nanoseconds)
        '''
        gamma_relax = 2*np.pi*(1/T1)
        gamma_echo = 2*np.pi*(1/Techo)
        gamma_phi = gamma_echo - (gamma_relax/2)
        #print(gamma_phi)
        
        term = np.sqrt(gamma_phi)*tensor(self.num_q, self.identity_c)
        self.c_ops.append(term)
        return None
    
    def add_cavity_relaxation(self, T1 = 10e+6):
        '''
        qubit relaxation (T1 in nanoseconds)
        
        in displaced frame, the a_c -> a_c + alpha but the alpha part
        can be taken care of by using classical EOM when designing pulses
        '''
        gamma_relax = 2*np.pi*(1/T1)
        term = np.sqrt(gamma_relax/2)*tensor(self.identity_q, self.a_c)
        self.c_ops.append(term)
        return None
    
#     def add_cavity_dephasing_old(self, T1, Techo, thermal = False, T_phi = None):
#         '''
#         Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
        
#         If thermal = true, adds dephasing contribution due to frequency shifts of oscillator due to thermal 
#         population in the qubit
#         '''

#         #Rates 
#         if T_phi != None: 
#             gamma_phi = 2*np.pi*(1/T_phi)
#         else:
#             gamma_relax= 2*np.pi*(1/T1)
#             gamma_echo = 2*np.pi*(1/Techo)
#             gamma_phi = gamma_echo - (gamma_relax/2)
#         gamma_total = gamma_phi

#         if thermal:
#             # Adding thermal cntribution
#             gamma_qubit = 1/self.T1qubit
#             n_thermal_qubit = 0.0092 # from Alec's param 
#                                         #1.2    #???   https://arxiv.org/pdf/2010.16382.pdf
#             gamma_thermal = gamma_qubit*( 
#                                 np.real(
#                                     np.sqrt(
#                                         (1 + (1.0j * self.chi/gamma_qubit))**2
#                                         +
#                                         (4.0j * self.chi * n_thermal_qubit / gamma_qubit)
#                                     )
#                                     -
#                                     1
#                                 ) / 2
#                                 )
#             gamma_total += gamma_thermal

        
#         #In transforming a -> a + alpha, the term a^dag a can be broken down as 
#         # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
#         # the latter term can be ignored cuz equal to identity
        
#         #gamma_total = 2*np.pi*1e-6*30
#         print(gamma_total)
#         term1 = gamma_total*tensor(self.identity_q, self.num_c)
#         term2 = gamma_total*tensor(self.identity_q, self.a_c)
#         term3 = gamma_total*tensor(self.identity_q, self.adag_c )
#         term4 = gamma_total*tensor(self.identity_q, self.identity_c )
        
#         #add to collapse operator list
#         self.c_ops.append(term1)
#         self.c_ops.append([term2, np.conjugate(self.alpha)]) # adding the time dependent coefficients
#         self.c_ops.append([term3, self.alpha])
        
#         alpha_sq =np.array([cnum*np.conjugate(cnum) for cnum in self.alpha])
#         print(np.max(alpha_sq))
#         self.c_ops.append([term4, alpha_sq])


        
        
        
#         return None
    
    def add_cavity_dephasing(self, T1, Techo, thermal = False, T_phi = None):
        '''
        Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
        '''
        #Rates 
        if T_phi != None: 
            gamma_phi = 2*np.pi*(1/T_phi)
        else:
            gamma_relax= 2*np.pi*(1/T1)
            gamma_echo = 2*np.pi*(1/Techo)
            gamma_phi = gamma_echo - (gamma_relax/2)
        
        
        #In transforming a -> a + alpha, the term a^dag a can be broken down as 
        # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
        # the latter term can be ignored cuz equal to identity
        term1 = np.sqrt(gamma_phi)*tensor(self.identity_q, self.num_c) 
        self.c_ops.append(term1)
        
        
        term2 = np.sqrt(gamma_phi)*tensor(self.identity_q, self.a_c)
        self.c_ops.append([term2, np.conjugate(self.alpha)])
        
        term3 = np.sqrt(gamma_phi)*tensor(self.identity_q, self.adag_c )
        self.c_ops.append([term3, self.alpha])
        
        #dissipators
        term4 = gamma_phi * lindblad_dissipator(tensor(self.identity_q, self.num_c), # a+ a,  a
                                                 tensor(self.identity_q, self.a_c))
        coeff =  np.array([ (np.abs(k))**2 for k in self.alpha])
        self.c_ops.append([term4,coeff])
#         print([ (np.abs(k))**2 for k in self.alpha])
#         print(term4)
        
        term5 =gamma_phi * lindblad_dissipator(tensor(self.identity_q, self.num_c),  # a+ a,  a+
                                                     tensor(self.identity_q, self.adag_c))
        self.c_ops.append([term5, np.array([ (np.abs(k))**2 for k in self.alpha])])
        
        term6 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, self.a_c),   #  a,  a+ a
                                                tensor(self.identity_q, self.num_c))
        self.c_ops.append([term6, np.array([ (np.abs(k))**2 for k in self.alpha])])
        
        term7 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, self.adag_c), #  a+,  a+ a
                                                tensor(self.identity_q, self.num_c))
        self.c_ops.append([term7, np.array([ (np.abs(k))**2 for k in self.alpha])])
        
        term8 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, self.a_c), #  a,  a+
                                                tensor(self.identity_q, self.adag_c))
        self.c_ops.append([term8, np.array([ k*k for k in np.conjugate(self.alpha)])])
        
        term9 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, self.adag_c), #  a+,  a
                                                tensor(self.identity_q, self.a_c))
        self.c_ops.append([term9, np.array([ k*k for k in self.alpha])])
        
#         #add to collapse operator list
#         self.c_ops.append(term1)
#         self.c_ops.append([term2, np.conjugate(self.alpha)]) # adding the time dependent coefficients
#         self.c_ops.append([term3, self.alpha])
        
        return None
    


    def me_solve(self, nsteps = 10000, initial = None): 
        '''
        Solve the master equation 
        '''
        T = len(self.alpha) # total time length in nanoseconds
        t_list = np.linspace(0, T, T)

        if initial == None: 
            initial = tensor(basis(self.n_q,0), basis(self.n_c,0))
            
        opts = Options(store_states=True, store_final_state=True, nsteps = nsteps, max_step = 10)
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
        state = self.output.states[-1]
        result = 0

        if (state.type == 'ket') and (target.type == 'ket'):
            result = self.dot(state, target)

        elif (state.type == 'oper') and (target.type == 'ket'): #density matrix alert
            target_rho= target*target.dag()
            result = np.sqrt(self.dot(state, target_rho)) # Hilbert schmidt prod is enough, no need for squaring (try to do this for pure states and you'll get why sqrt used here)

        elif (state.type == 'oper') and (target.type == 'oper'): #density matrix alert
            #target_rho= target*target.dag()
            result = np.sqrt(self.dot(state, target_rho)) # Hilbert schmidt prod is enough, no need for squaring (try to do this for pure states and you'll get why sqrt used here)

        
        return result
    
    
    def plot_populations_single_mode(self, figname = 'figure', title = None):
        '''
        Given output of mesolve, outputs populations with qubit as ground
        '''
        
        output_states = self.output.states
        fig, axs = plt.subplots(2,1, figsize=(10,8))
        probs = []
        times = [k/1000 for k in range(len(output_states))]
        max_num_levels = 10 # to be shown on the plot

        #qubit grounded
        for i in range(max_num_levels):
            target = tensor(basis(self.n_q,0), basis(self.n_c, i))
            pops = []
            for k in range(len(output_states)): 
                z = target.overlap(output_states[k])
                pops.append(z.real**2 + z.imag**2)
            axs[0].plot(times, pops, label = '|g,'+str(i)+'>')

        #qubit excited
        for i in range(max_num_levels):
            target = tensor(basis(self.n_q,1), basis(self.n_c, i))
            pops = []
            for k in range(len(output_states)): 
                z = target.overlap(output_states[k])
                pops.append(z.real**2 + z.imag**2)
            axs[1].plot(times, pops, linestyle = '--',  label = '|e,'+str(i) +'>')

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

    
