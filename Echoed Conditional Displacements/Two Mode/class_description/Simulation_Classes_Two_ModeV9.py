from MECD_pulseV1 import *
from qutip import *
import numpy as np
import h5py as hf

#V3: changed how angles are loaded 
#V4: Adding time dependence to collapse operator decay rates
#V5: Added thermal noise
#V6: Proper cavity dephasing under displaced frame transformations
#V7: Fidelity computation differs for density matrices and kets
#V8: Mode Mode coupling (cross kerr) under displaced frame transformation
#V9: WIth MECD code

class ecd_pulse_multimode: 
    
    def __init__(self, 
                 param_file = None, 
                 storages_params = None,
                 N_modes = 2,
                # storage2_params = None, 
                 betas = None, 
                 #gammas = None,
                 phis = None, 
                 thetas = None,
                 qubit_params = None, 
                 alpha_CD =30, 
                 kappa = [0.5e-6,0.5e-6], # T1 for both modes is 2ms
                # kappa2 = 0.5e-6,
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
        self.param_file = param_file # for loading parameters
        self.betas = betas
        #self.gammas = gammas
        self.phis = phis
        self.thetas = thetas
        self.load_params()
        self.N_modes = N_modes
        
        self.kappa = kappa
        
        #Pulse parameters

        ## modes
        self.storages_params = storages_params
        if storages_params == None: 
            self.storages_params = [{
                            "chi_kHz": -33, 
                            "chi_prime_Hz": 0, 
                            "Ks_Hz": 0,
                            "epsilon_m_MHz": 400, 
                            "unit_amp": 0.01, 
                            "sigma": 11, 
                            "chop": 4, 
                        } for _ in range(self.N_modes)]
        #self.storage2_params = storage2_params
        
            
        self.qubit_params = qubit_params
        if self.qubit_params == None: 
            self.qubit_params = {'unit_amp': 0.5,
                                 'sigma': 6, 
                                 'chop': 4} 
        self.storages = None
        #self.storage2 = None
        self.qubit = None
        
        
        #parameters obtained after get_pulse() is called
        self.modes_dac_pulse_GHz = None
        #self.cavity2_dac_pulse_Ghz = None
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
        #params = np.loadtxt(self.param_file)
        filename = self.param_file
        file = hf.File(filename, "r")
        timestamp = list(file.keys())[-1]
        fids = file[timestamp]['fidelities'][-1]
        print('fidelity for h5 param is ' + str(max(fids)))
        best_fid_idx = np.argmax(fids)
        print('index of fidelity for h5 param is ' + str(best_fid_idx))
        self.betas = file[timestamp]['betas'][-1][best_fid_idx]
        #self.gammas = file[timestamp]['gammas'][-1][best_fid_idx]
        self.phis = file[timestamp]['phis'][-1][best_fid_idx]
        self.thetas = file[timestamp]['thetas'][-1][best_fid_idx]
        return None

    def get_pulses(self): 
        '''
        Evaluates cavity and qubit pulses for the desired ECD simulation
        '''
        #Creates objects
        self.storages = [FakeStorage(**self.storages_params[m]) for m in range(self.N_modes)] 
        # = FakeStorage(**self.storage2_params)
        self.qubit = FakeQubit(**self.qubit_params)
        
        #Qubit pi pulse stuff ... calculating conversion between qubit DAC units and MHz (Omega)
        pi = rotate(np.pi, phi=0, sigma=self.qubit_params['sigma'], chop=self.qubit_params['chop'], dt=1)
        Omega_m = np.real(np.max(pi))/self.qubit_params['unit_amp']
        
        #get pulses
        pulse_dict = conditional_displacement_circuit(self.betas,
                                                     #self.gammas,
                                                     self.phis,
                                                     self.thetas,
                                                     self.storages, 
                                                    #  self.storage2, 
                                                     self.qubit,
                                                     alpha_CD = self.alpha_CD,
                                                    # self.alpha_CD ,
                                                     buffer_time=self.buffer_time, 
                                                     kerr_correction = False, 
                                                     kappa = self.kappa,
                                                   #  kappa2 = self.kappa2,
                                                     chi_prime_correction=True, 
                                                   #  final_disp=True, 
                                                     pad=True)
        modes_dac_pulse,  qubit_dac_pulse, = pulse_dict['cavity_dac_pulse'],  pulse_dict['qubit_dac_pulse']
    
        #Dac units to Ghz conversion

        self.modes_dac_pulse_GHz = [(2*np.pi)*(10**(-3))*self.storages[m].epsilon_m_MHz*modes_dac_pulse[m] for m in range(self.N_modes)] #convert from DAC to Mhz to Ghz
        #self.cavity2_dac_pulse_GHz = (2*np.pi)*(10**(-3))*self.storage2.epsilon_m_MHz*cavity2_dac_pulse #convert from DAC to Mhz to Ghz
        self.qubit_dac_pulse_GHz = (2*np.pi)*10**(-3)*self.qubit.Omega_m_MHz*qubit_dac_pulse #convert from DAC to Mhz to Ghz

        #alpha:  Solve equation of motion and get corresponding displacements for displaced frame simulations
        self.alpha = [self.alpha_from_epsilon_nonlinear_finite_difference(
            epsilon_array =self.modes_dac_pulse_GHz[m] , delta=0, kappa = self.kappa[m], alpha_init=0 + 0j) for m in range(self.N_modes)] 
        
        # self.alpha2 = self.alpha_from_epsilon_nonlinear_finite_difference(
        #     epsilon_array =self.cavity2_dac_pulse_GHz , delta=0, kappa = self.kappa2, alpha_init=0 + 0j)
        
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
        #fig, axs = plt.subplots(5,1)
        fig, axs = plt.subplots(3,1)
        fig, axs = plt.subplots(self.N_modes+1,1)
        for m_ in range(self.N_modes):
            axs[m_].plot(np.real(self.modes_dac_pulse_GHz[m_]))
            axs[m_].plot(np.imag(self.modes_dac_pulse_GHz[m_]))
            axs[m_].set_ylabel('Mode '+str(m_)+' Drive (Ghz)', fontsize = 10)
            
        axs[-1].plot(np.real(self.qubit_dac_pulse_GHz))
        axs[-1].plot(np.imag(self.qubit_dac_pulse_GHz))
        axs[-1].set_ylabel('qubit dac pulse (GHz)', fontsize = 10)
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
        #print('hi')
        self.n_q = n_q
        self.n_c1 = n_c1
        self.n_c2 = n_c2
        
        self.chi1 = -33*2*np.pi*(10**(-6)) # Alec's params
        self.chi2 = self.chi1
        self.qubit_anh = 150*2*np.pi*(10**(-3)) # units of ghz
        
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
        self.add_bare_qubit_mode_coupling()
        self.add_stark_shift()
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
        eta = self.chi1*self.chi2/self.qubit_anh   
        print('V8 mode-mode coupling')
        
        #a^dag a 
        
        ## bdag b
        self.H0+= eta*tensor(self.identity_q, self.num_c, self.num_c)
        ## beta * bdag
        term = tensor(self.identity_q, self.num_c, self.adag_c)
        self.Hd.append(
            [eta * term, self.alpha2 ]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.num_c, self.a_c)
        self.Hd.append(
            [eta * term, np.conjugate(self.alpha2 )]
        )
        ## |beta|^2
        term = tensor(self.identity_q, self.num_c, self.identity_c)
        self.Hd.append(
            [eta * term, np.array([(np.abs(amp))**2 for amp in self.alpha2])]
        )
        
        #a^dag alpha 
        
        ## bdag b
        term = tensor(self.identity_q, self.adag_c, self.num_c)
        self.Hd.append(
            [eta * term, self.alpha1 ]
        )
        ## beta * bdag
        term = tensor(self.identity_q, self.adag_c, self.adag_c)
        self.Hd.append(
            [eta * term, np.array([amp1 * amp2 for amp1, amp2 in zip(self.alpha1, self.alpha2) ])]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.adag_c, self.a_c)
        self.Hd.append(
            [eta * term, 
            np.array( [amp1 * amp2 for amp1, amp2 in zip(self.alpha1, np.conjugate(self.alpha2)) ])]
        )
        ## |beta|^2
        term = tensor(self.identity_q,self.adag_c, self.identity_c)
        self.Hd.append(
            [eta * term, 
             np.array([amp1 * amp2 for amp1, amp2 in zip(self.alpha1, 
                                                [(np.abs(amp))**2 for amp in self.alpha2]
                                               ) ])]
                    )
        #a alpha ^*
        
        ## bdag b
        term = tensor(self.identity_q, self.a_c, self.num_c)
        self.Hd.append(
            [eta * term, np.conjugate(self.alpha1 )]
        )
        ## beta * bdag
        term = tensor(self.identity_q, self.a_c, self.adag_c)
        self.Hd.append(
            [eta * term, np.array([amp1 * amp2 for amp1, amp2 in zip(np.conjugate(self.alpha1 ), 
                                                            self.alpha2) ])]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.a_c, self.a_c)
        self.Hd.append(
            [eta * term, 
             np.array([amp1 * amp2 for amp1, amp2 in zip(np.conjugate(self.alpha1 ),
                                                np.conjugate(self.alpha2)) ])]
        )
        ## |beta|^2
        term = tensor(self.identity_q, self.a_c, self.identity_c)
        self.Hd.append(
            [eta * term, 
             np.array([amp1 * amp2 for amp1, amp2 in zip(np.conjugate(self.alpha1 ), 
                                                [(np.abs(amp))**2 for amp in self.alpha2]
                                               ) ])]
                    )
        
         #|alpha|^2
        
        ## bdag b
        term = tensor(self.identity_q, self.identity_c, self.num_c)
        self.Hd.append(
            [eta * term, np.array([(np.abs(amp))**2 for amp in self.alpha1])]
        )
        ## beta * bdag
        term = tensor(self.identity_q, self.identity_c, self.adag_c)
        self.Hd.append(
            [eta * term, np.array([amp1 * amp2 for amp1, amp2 in zip(
                                    [(np.abs(amp))**2 for amp in self.alpha1], 
                                                            self.alpha2)
                         ])]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.identity_c, self.a_c)
        self.Hd.append(
            [eta * term, 
            np.array( [amp1 * amp2 for amp1, amp2 in zip([(np.abs(amp))**2 for amp in self.alpha1],
                                                np.conjugate(self.alpha2)) ])]
        )
        ## |beta|^2 just constant term
        
        return None
        
        
        
    
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
        term = np.sqrt(gamma_relax)*tensor(self.a_q, self.identity_c, self.identity_c)
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
        term1 = np.sqrt(gamma_relax_mode1)*tensor(self.identity_q, self.a_c, self.identity_c)

        gamma_relax_mode2 = 1/T1_mode2
        term2 = np.sqrt(gamma_relax_mode2)*tensor(self.identity_q, self.identity_c, self.a_c)


        self.c_ops.append(term1)
        self.c_ops.append(term2)
        return None
    
    
    
    
    def add_cavity_dephasing_for_given_mode(self, T1, Techo, alpha, mode_idx = 1, thermal = False, T_phi = None):
        '''
        Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
        '''
        print('hii')
        #Rates 
        if T_phi != None: 
            gamma_phi = 2*(1/T_phi)
        else:
            gamma_relax= (1/T1)
            gamma_echo = (1/Techo)
            gamma_phi = gamma_echo - (gamma_relax/2)
        print(gamma_phi)


        #In transforming a -> a + alpha, the term a^dag a can be broken down as 
        # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
        # the latter term can be ignored cuz equal to identity


        term1 = np.sqrt(gamma_phi)*tensor(self.identity_q, 
                                          self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                          self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)                                
                                         ) 
        self.c_ops.append(term1)


        term2 = np.sqrt(gamma_phi)*tensor(self.identity_q,
                                          self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2))
        self.c_ops.append([term2, np.conjugate(alpha)])

        term3 = np.sqrt(gamma_phi)*tensor(self.identity_q, 
                                          self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                          self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2))
        self.c_ops.append([term3, alpha])

        #dissipators
        term4 = gamma_phi * lindblad_dissipator(tensor(self.identity_q, 
                                                       self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                       self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ), # a+ a,  a
                                                 tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                       )
                                               )
        coeff =  np.array([ (np.abs(k))**2 for k in alpha])
        self.c_ops.append([term4,coeff])
        #         print([ (np.abs(k))**2 for k in self.alpha])
        #         print(term4)

        term5 =gamma_phi * lindblad_dissipator(tensor(self.identity_q, 
                                                   self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                   self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ),  # a+ a,  a+
                                               tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)))
        self.c_ops.append([term5, np.array([ (np.abs(k))**2 for k in alpha])])

        term6 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                           ),                                                           #  a,  a+ a
                                                tensor(self.identity_q, 
                                                       self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                       self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ))
        self.c_ops.append([term6, np.array([ (np.abs(k))**2 for k in alpha])])

        term7 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)), #  a+,  a+ a
                                                tensor(self.identity_q, 
                                                       self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                       self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ))
        self.c_ops.append([term7, np.array([ (np.abs(k))**2 for k in alpha])])

        term8 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                           ), #  a,  a+
                                                tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx != 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx != 2) + self.identity_c * (mode_idx != 2)))
        self.c_ops.append([term8, np.array([ k*k for k in np.conjugate(alpha)])])

        term9 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)), #  a+,  a
                                                tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                           ))
        self.c_ops.append([term9, np.array([ k*k for k in alpha])])

        #         #add to collapse operator list
        #         self.c_ops.append(term1)
        #         self.c_ops.append([term2, np.conjugate(self.alpha)]) # adding the time dependent coefficients
        #         self.c_ops.append([term3, self.alpha])

        return None
    
    
    def add_cavity_dephasing_for_given_mode_old(self, T1, Techo, alpha, mode_index = 1, thermal = False):
        '''
        Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
        '''
        print('hi')
        gamma_relax= 1/T1
        gamma_echo = 1/Techo
        gamma_phi = gamma_echo - (gamma_relax/2)
        gamma_total = gamma_phi
        
        if thermal:
            # Adding thermal cntribution
            gamma_qubit = 1/self.T1qubit
            n_thermal_qubit = 1.2    #???   https://arxiv.org/pdf/2010.16382.pdf
            gamma_thermal = gamma_qubit*( 
                                np.real(
                                    np.sqrt(
                                        (1 + (1.0j * self.chi/gamma_qubit))**2
                                        +
                                        (4.0j * self.chi * n_thermal_qubit / gamma_qubit)
                                    )
                                    -
                                    1
                                ) / 2
                                )
            gamma_total += gamma_thermal
        
        
        #In transforming a -> a + alpha, the term a^dag a can be broken down as 
        # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
        # the latter term can be ignored cuz equal to identity
        if mode_index == 1: 
            term1 = gamma_total*tensor(self.identity_q, self.num_c, self.identity_c)
            term2 = gamma_total*tensor(self.identity_q, self.a_c  , self.identity_c)
            term3 = gamma_total*tensor(self.identity_q, self.adag_c  , self.identity_c)
        else: #mode index = 2
            term1 = gamma_total*tensor(self.identity_q, self.identity_c, self.num_c)
            term2 = gamma_total*tensor(self.identity_q, self.identity_c, self.a_c )
            term3 = gamma_total*tensor(self.identity_q, self.identity_c, self.adag_c)
        
        #add to collapse operator list
        self.c_ops.append(term1)
        self.c_ops.append([term2, np.conjugate(alpha)]) # adding the time dependent coefficients
        self.c_ops.append([term3, alpha])
        
        return None
        
    
    def add_cavity_dephasing(self, T1_mode1 = 10e+6, Techo_mode1 = 10e+6, T1_mode2 = 10e+6, Techo_mode2 = 10e+6, thermal = None, T_phi = None):
        '''
        qubit dephasing (T1, Techo in nanoseconds)
        '''
       
        self.add_cavity_dephasing_for_given_mode( T1_mode1, Techo_mode1, self.alpha1, mode_idx = 1, thermal = thermal, T_phi = T_phi)
        self.add_cavity_dephasing_for_given_mode( T1_mode2, Techo_mode2, self.alpha2, mode_idx = 2, thermal = thermal, T_phi= T_phi)        
        return None
    
    
    def me_solve(self, nsteps = 10000, initial = None, e_ops = None): 
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
            
        self.output = mesolve(self.H, initial , t_list, self.c_ops, e_ops, options =opts)        
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
        
        
        return np.real(result) #result's imag [art should be 0
    
    
    
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


class qutip_sim_multimode_mode:
    
    def __init__(self, n_q, n_c, N_modes, chi = None,
                 alphas = [], qubit_pulse = [],
                 sim_params = None, save_states = False, states_filename = 'states store'):
        '''
        n_q, n_c = # of levels in qubit, cavity
        Assumes n_c1 = n_c2
        
        sim_params = sim_params = {'bare_qubit_mode_coupling':0,
        'Stark Shift': 0,  'transmon_relax': 0,'transmon_dephasing': 0,
        'cavity_relax': 0,'cavity_dephasing': 0}  # 0 means false, 1 means true or activates    
        '''
        #print('hi')
        self.n_q = n_q
        self.n_c = n_c
        # self.n_c2 = n_c2
        
        self.chi = [-33*2*np.pi*(10**(-6)) for _ in range(N_modes)] # Alec's params
        #self.chi2 = self.chi1
        self.qubit_anh = 150*2*np.pi*(10**(-3)) # units of ghz
        
        ##get pulses
        self.alphas = alphas
       # self.alpha2 = alpha2
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
        self.add_bare_qubit_mode_coupling()
        self.add_stark_shift()
        self.states_filename = states_filename
        self.save_states= save_states
        
    
    def get_basic_ops(self): 
        '''
        Creates identity, creation/annihilation for qubit/cavity

        Assumption n_c1 = n_c2
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
        self.H0 +=  (self.chi1/2)*tensor(sigmaz(), self.num_c, self.identity_c)
        self.H0 +=  (self.chi2/2)*tensor(sigmaz(), self.identity_c, self.num_c,)
        return None

    def add_mode_mode_coupling(self, eta = None):
        '''
        Add mode mode coupling term or cross kerr interaction

        cross kerr = sqrt(kerr_mode1  * kerr_mode2) = chi1 * chi2 /anharmonicity of qubit
        '''
        eta = self.chi1*self.chi2/self.qubit_anh   
        print('V8 mode-mode coupling')
        
        #a^dag a 
        
        ## bdag b
        self.H0+= eta*tensor(self.identity_q, self.num_c, self.num_c)
        ## beta * bdag
        term = tensor(self.identity_q, self.num_c, self.adag_c)
        self.Hd.append(
            [eta * term, self.alpha2 ]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.num_c, self.a_c)
        self.Hd.append(
            [eta * term, np.conjugate(self.alpha2 )]
        )
        ## |beta|^2
        term = tensor(self.identity_q, self.num_c, self.identity_c)
        self.Hd.append(
            [eta * term, np.array([(np.abs(amp))**2 for amp in self.alpha2])]
        )
        
        #a^dag alpha 
        
        ## bdag b
        term = tensor(self.identity_q, self.adag_c, self.num_c)
        self.Hd.append(
            [eta * term, self.alpha1 ]
        )
        ## beta * bdag
        term = tensor(self.identity_q, self.adag_c, self.adag_c)
        self.Hd.append(
            [eta * term, np.array([amp1 * amp2 for amp1, amp2 in zip(self.alpha1, self.alpha2) ])]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.adag_c, self.a_c)
        self.Hd.append(
            [eta * term, 
            np.array( [amp1 * amp2 for amp1, amp2 in zip(self.alpha1, np.conjugate(self.alpha2)) ])]
        )
        ## |beta|^2
        term = tensor(self.identity_q,self.adag_c, self.identity_c)
        self.Hd.append(
            [eta * term, 
             np.array([amp1 * amp2 for amp1, amp2 in zip(self.alpha1, 
                                                [(np.abs(amp))**2 for amp in self.alpha2]
                                               ) ])]
                    )
        #a alpha ^*
        
        ## bdag b
        term = tensor(self.identity_q, self.a_c, self.num_c)
        self.Hd.append(
            [eta * term, np.conjugate(self.alpha1 )]
        )
        ## beta * bdag
        term = tensor(self.identity_q, self.a_c, self.adag_c)
        self.Hd.append(
            [eta * term, np.array([amp1 * amp2 for amp1, amp2 in zip(np.conjugate(self.alpha1 ), 
                                                            self.alpha2) ])]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.a_c, self.a_c)
        self.Hd.append(
            [eta * term, 
             np.array([amp1 * amp2 for amp1, amp2 in zip(np.conjugate(self.alpha1 ),
                                                np.conjugate(self.alpha2)) ])]
        )
        ## |beta|^2
        term = tensor(self.identity_q, self.a_c, self.identity_c)
        self.Hd.append(
            [eta * term, 
             np.array([amp1 * amp2 for amp1, amp2 in zip(np.conjugate(self.alpha1 ), 
                                                [(np.abs(amp))**2 for amp in self.alpha2]
                                               ) ])]
                    )
        
         #|alpha|^2
        
        ## bdag b
        term = tensor(self.identity_q, self.identity_c, self.num_c)
        self.Hd.append(
            [eta * term, np.array([(np.abs(amp))**2 for amp in self.alpha1])]
        )
        ## beta * bdag
        term = tensor(self.identity_q, self.identity_c, self.adag_c)
        self.Hd.append(
            [eta * term, np.array([amp1 * amp2 for amp1, amp2 in zip(
                                    [(np.abs(amp))**2 for amp in self.alpha1], 
                                                            self.alpha2)
                         ])]
        )
        ## beta* * b
        term = tensor(self.identity_q, self.identity_c, self.a_c)
        self.Hd.append(
            [eta * term, 
            np.array( [amp1 * amp2 for amp1, amp2 in zip([(np.abs(amp))**2 for amp in self.alpha1],
                                                np.conjugate(self.alpha2)) ])]
        )
        ## |beta|^2 just constant term
        
        return None
        
        
        
    
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
        term = np.sqrt(gamma_relax)*tensor(self.a_q, self.identity_c, self.identity_c)
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
        term1 = np.sqrt(gamma_relax_mode1)*tensor(self.identity_q, self.a_c, self.identity_c)

        gamma_relax_mode2 = 1/T1_mode2
        term2 = np.sqrt(gamma_relax_mode2)*tensor(self.identity_q, self.identity_c, self.a_c)


        self.c_ops.append(term1)
        self.c_ops.append(term2)
        return None
    
    
    
    
    def add_cavity_dephasing_for_given_mode(self, T1, Techo, alpha, mode_idx = 1, thermal = False, T_phi = None):
        '''
        Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
        '''
        print('hii')
        #Rates 
        if T_phi != None: 
            gamma_phi = 2*(1/T_phi)
        else:
            gamma_relax= (1/T1)
            gamma_echo = (1/Techo)
            gamma_phi = gamma_echo - (gamma_relax/2)
        print(gamma_phi)


        #In transforming a -> a + alpha, the term a^dag a can be broken down as 
        # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
        # the latter term can be ignored cuz equal to identity


        term1 = np.sqrt(gamma_phi)*tensor(self.identity_q, 
                                          self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                          self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)                                
                                         ) 
        self.c_ops.append(term1)


        term2 = np.sqrt(gamma_phi)*tensor(self.identity_q,
                                          self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2))
        self.c_ops.append([term2, np.conjugate(alpha)])

        term3 = np.sqrt(gamma_phi)*tensor(self.identity_q, 
                                          self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                          self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2))
        self.c_ops.append([term3, alpha])

        #dissipators
        term4 = gamma_phi * lindblad_dissipator(tensor(self.identity_q, 
                                                       self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                       self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ), # a+ a,  a
                                                 tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                       )
                                               )
        coeff =  np.array([ (np.abs(k))**2 for k in alpha])
        self.c_ops.append([term4,coeff])
        #         print([ (np.abs(k))**2 for k in self.alpha])
        #         print(term4)

        term5 =gamma_phi * lindblad_dissipator(tensor(self.identity_q, 
                                                   self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                   self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ),  # a+ a,  a+
                                               tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)))
        self.c_ops.append([term5, np.array([ (np.abs(k))**2 for k in alpha])])

        term6 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                           ),                                                           #  a,  a+ a
                                                tensor(self.identity_q, 
                                                       self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                       self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ))
        self.c_ops.append([term6, np.array([ (np.abs(k))**2 for k in alpha])])

        term7 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)), #  a+,  a+ a
                                                tensor(self.identity_q, 
                                                       self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                       self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ))
        self.c_ops.append([term7, np.array([ (np.abs(k))**2 for k in alpha])])

        term8 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                           ), #  a,  a+
                                                tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx != 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx != 2) + self.identity_c * (mode_idx != 2)))
        self.c_ops.append([term8, np.array([ k*k for k in np.conjugate(alpha)])])

        term9 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
                                                     self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                      self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)), #  a+,  a
                                                tensor(self.identity_q, 
                                                         self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
                                                         self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
                                                           ))
        self.c_ops.append([term9, np.array([ k*k for k in alpha])])

        #         #add to collapse operator list
        #         self.c_ops.append(term1)
        #         self.c_ops.append([term2, np.conjugate(self.alpha)]) # adding the time dependent coefficients
        #         self.c_ops.append([term3, self.alpha])

        return None
    
    
    def add_cavity_dephasing_for_given_mode_old(self, T1, Techo, alpha, mode_index = 1, thermal = False):
        '''
        Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
        '''
        print('hi')
        gamma_relax= 1/T1
        gamma_echo = 1/Techo
        gamma_phi = gamma_echo - (gamma_relax/2)
        gamma_total = gamma_phi
        
        if thermal:
            # Adding thermal cntribution
            gamma_qubit = 1/self.T1qubit
            n_thermal_qubit = 1.2    #???   https://arxiv.org/pdf/2010.16382.pdf
            gamma_thermal = gamma_qubit*( 
                                np.real(
                                    np.sqrt(
                                        (1 + (1.0j * self.chi/gamma_qubit))**2
                                        +
                                        (4.0j * self.chi * n_thermal_qubit / gamma_qubit)
                                    )
                                    -
                                    1
                                ) / 2
                                )
            gamma_total += gamma_thermal
        
        
        #In transforming a -> a + alpha, the term a^dag a can be broken down as 
        # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
        # the latter term can be ignored cuz equal to identity
        if mode_index == 1: 
            term1 = gamma_total*tensor(self.identity_q, self.num_c, self.identity_c)
            term2 = gamma_total*tensor(self.identity_q, self.a_c  , self.identity_c)
            term3 = gamma_total*tensor(self.identity_q, self.adag_c  , self.identity_c)
        else: #mode index = 2
            term1 = gamma_total*tensor(self.identity_q, self.identity_c, self.num_c)
            term2 = gamma_total*tensor(self.identity_q, self.identity_c, self.a_c )
            term3 = gamma_total*tensor(self.identity_q, self.identity_c, self.adag_c)
        
        #add to collapse operator list
        self.c_ops.append(term1)
        self.c_ops.append([term2, np.conjugate(alpha)]) # adding the time dependent coefficients
        self.c_ops.append([term3, alpha])
        
        return None
        
    
    def add_cavity_dephasing(self, T1_mode1 = 10e+6, Techo_mode1 = 10e+6, T1_mode2 = 10e+6, Techo_mode2 = 10e+6, thermal = None, T_phi = None):
        '''
        qubit dephasing (T1, Techo in nanoseconds)
        '''
       
        self.add_cavity_dephasing_for_given_mode( T1_mode1, Techo_mode1, self.alpha1, mode_idx = 1, thermal = thermal, T_phi = T_phi)
        self.add_cavity_dephasing_for_given_mode( T1_mode2, Techo_mode2, self.alpha2, mode_idx = 2, thermal = thermal, T_phi= T_phi)        
        return None
    
    
    def me_solve(self, nsteps = 10000, initial = None, e_ops = None): 
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
            
        self.output = mesolve(self.H, initial , t_list, self.c_ops, e_ops, options =opts)        
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
        
        
        return np.real(result) #result's imag [art should be 0
    
    
    
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