from MECD_pulseV4 import *
from qutip import *
import numpy as np
import h5py as hf
from scipy import interpolate
import matplotlib.pyplot as plt


#V3: changed how angles are loaded 
#V4: Adding time dependence to collapse operator decay rates
#V5: Added thermal noise
#V6: Proper cavity dephasing under displaced frame transformations
#V7: Fidelity computation differs for density matrices and kets
#V8: Mode Mode coupling (cross kerr) under displaced frame transformation
#V9: WIth MECD code and gf ECD
#V10: With Circle Grape (both ge and gef) compatibility for qutip sim 

#August 2, 2023
#V11: Code compatible with Qutrit ECD 

#August 11, 2023
#V12: each layer consists of ef rotation, ge rotation and then ge ECD

class ecd_pulse_multimode: 
    
    def __init__(self, 
                 param_file = None, 
                 storages_params = None,
                 N_modes = 2,
                 version = 'gef',
                # storage2_params = None,
                 chis = None, 
                 betas = None, 
                 #gammas = None,
                 phis = None, 
                 thetas = None,
                 qubit_params = None, 
                 alpha_CD =30, 
                 kappa = [0.5e-6,0.5e-6], # T1 for both modes is 2ms
                # kappa2 = 0.5e-6,
                 buffer_time = 0): 
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
        self.version = version
        
        self.kappa = kappa
        
        #Pulse parameters

        ## modes
        self.storages_params = storages_params
        
        if storages_params == None: 
            self.storages_params = [{
                            "chi_kHz": 1e+6 * np.array(chis[0]),  # from ECD paper  #chis (Assume same chis for all modes for now 
                            "chi_prime_Hz":np.array([0,0,0]),# np.array([0, 1.5, 3]), 
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
        if filename is None:
            return None
        file = hf.File(filename, "r")
        timestamp = list(file.keys())[-1]
        fids = file[timestamp]['fidelities'][-1]
        print('fidelity for h5 param is ' + str(max(fids)))
        best_fid_idx = np.argmax(fids)
        print('index of fidelity for h5 param is ' + str(best_fid_idx))
        self.betas = file[timestamp]['betas'][-1][best_fid_idx]
        #self.gammas = file[timestamp]['gammas'][-1][best_fid_idx]
        self.phis = file[timestamp]['phis'][-1][best_fid_idx]
        #bug in MECD param V2
        (m,n,s) = self.phis.shape
        for m_ in range(m):
            for n_ in range(n): 
                for s_ in range(s):
                    self.phis[m_,n_, s_] = self.phis[m_,n_,s_] - (np.pi/2)
        self.thetas = file[timestamp]['thetas'][-1][best_fid_idx]

        # self.betas = np.array([self.betas[0][:1]])
        # self.phis = np.array([self.phis[0][:1]])
        # self.thetas = np.array([self.thetas[0][:1]])
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
        pulse_dict = None
        if self.version == 'ge': 
            pulse_dict = conditional_displacement_circuit_old(self.betas,
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
                                                        pad=True, 
                                                        is_gf = False
                                                        )
        elif self.version == 'gef':
            pulse_dict = conditional_displacement_circuit_ge(self.betas,
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
                                                        pad=True, 
                                                        #is_gf = False
                                                        ) 

        #return pulse_dict
        modes_dac_pulse,  qubit_dac_pulse, = pulse_dict['cavity_dac_pulse'],  pulse_dict['qubit_dac_pulse']
    
        #Dac units to Ghz conversion

        self.modes_dac_pulse_GHz = [(2*np.pi)*(10**(-3))*self.storages[m].epsilon_m_MHz*modes_dac_pulse[m] for m in range(self.N_modes)] #convert from DAC to Mhz to Ghz
        #self.cavity2_dac_pulse_GHz = (2*np.pi)*(10**(-3))*self.storage2.epsilon_m_MHz*cavity2_dac_pulse #convert from DAC to Mhz to Ghz
        if self.version == 'ge':
            self.qubit_dac_pulse_GHz = (2*np.pi)*10**(-3)*self.qubit.Omega_m_MHz*qubit_dac_pulse #convert from DAC to Mhz to Ghz
        elif self.version == 'gef': 
            self.qubit_dac_pulse_GHz = [(2*np.pi)*10**(-3)*self.qubit.Omega_m_MHz*qubit_dac_pulse[i] for i in range(len(qubit_dac_pulse))] 

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
                        # probabily assuming that chi_g = 0 , same assumption in MECD V2 pulse file
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
        #fig, axs = plt.subplots(3,1)
        if self.version == 'ge':
            fig, axs = plt.subplots(self.N_modes+1,1)
            for m_ in range(self.N_modes):
                axs[m_].plot(np.real(self.modes_dac_pulse_GHz[m_]))
                axs[m_].plot(np.imag(self.modes_dac_pulse_GHz[m_]))
                axs[m_].set_ylabel('Mode '+str(m_)+' Drive (Ghz)', fontsize = 10)

            axs[-1].plot(np.real(self.qubit_dac_pulse_GHz))
            axs[-1].plot(np.imag(self.qubit_dac_pulse_GHz))
            axs[-1].set_ylabel('qubit dac pulse (GHz)', fontsize = 10)
            plt.xlabel('ns')
            
        elif self.version == 'gef':
            fig, axs = plt.subplots(self.N_modes+2,1)
            for m_ in range(self.N_modes):
                axs[m_].plot(np.real(self.modes_dac_pulse_GHz[m_]))
                axs[m_].plot(np.imag(self.modes_dac_pulse_GHz[m_]))
                axs[m_].set_ylabel('Mode '+str(m_)+' Drive (Ghz)', fontsize = 10)

            axs[-2].plot(np.real(self.qubit_dac_pulse_GHz[0]))
            axs[-2].plot(np.imag(self.qubit_dac_pulse_GHz[0]))
            axs[-2].set_ylabel('qubit ge (GHz)', fontsize = 10)
            
            axs[-1].plot(np.real(self.qubit_dac_pulse_GHz[1]))
            axs[-1].plot(np.imag(self.qubit_dac_pulse_GHz[1]))
            axs[-1].set_ylabel('qubit ef (GHz)', fontsize = 10)
            plt.xlabel('ns')
        
        return fig#None#fig
    
    
####################################################################################
#############     Qutip Now   ######################################################
class qutip_sim_multimode:
    
    def __init__(self, n_q, n_c, N_modes, method = 'ecd', # or 'cgrape'
                 version= 'ge', # or gef or gf
                 chis = None,
                 alphas = [], qubit_pulse = [],
                 sim_params = None, save_states = False, 
                 filename = '',
                 states_filename = 'states store'):
        '''
        n_q, n_c = # of levels in qubit, cavity
        
        chis = [[chi for mode 1], [chi for mode 2], ...]
        
        where each chi is a list [chi] = [chi_g, chi_e, ...]
        
        sim_params = sim_params = {'bare_qubit_mode_coupling':0,
        'Stark Shift': 0,  'transmon_relax': 0,'transmon_dephasing': 0,
        'cavity_relax': 0,'cavity_dephasing': 0}  # 0 means false, 1 means true or activates    
        '''
        #print('hi')
        self.n_q = n_q
        self.n_c = n_c
        self.N_modes = N_modes
        self.version = version
        self.method = method
        self.filename = filename # for the parameters
        
       
        self.chis= 2 * np.pi * np.transpose([[0,0], [-4.79148181e-05, -4.41176471e-05], [-8.75366869e-05, -8.08823529e-05]])
        if self.method == 'cgrape':
            self.detuning = [0.005 * 2 * np.pi  for _ in range(N_modes)] # all modes have 5 MHz of detuning
        else: 
            self.detuning = [0 for _ in range(N_modes)]
            
        self.qubit_anh = 150*2*np.pi*(10**(-3)) # units of ghz
        
        self.compute_chis()
        
        #load pulses from simulation files
        self.load_pulse()       
            
        #get operators
        self.get_basic_ops()
        
        ##H0 is identity
        self.H0 = 0 * tensor( self.identity_q , self.identity_mms ) # time independent part
        
        
        ## drive terms (ecd pulses)
        self.Hd = []
#         self.initialize_ECD_and_qubit_drive() #is_gf)
        
        self.c_ops = []
        self.states_filename = states_filename
        self.save_states= save_states
        
    def compute_chis(self): 
        '''
        Computes chi_g,e,f for 2 modes with specified detunings and alpha(anharmonicity) and coupling g's between 
        transmon and the cavity modes
        
        Assumes two modes
        
        Assume chi_g = 0
        '''
        
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

        chi_g = [0 for i in range(2)]
        chi_e = [chi_wrt_g(g, deltas[i], alpha, 1) for i in range(2)] 
        chi_f = [chi_wrt_g(g, deltas[i], alpha, 2) for i in range(2)]
        
        #Shape of chis is chi[mode][transmon level]
        self.chis =  2 * np.pi * np.transpose([chi_g, chi_e, chi_f])
        
        return None
        
    def load_pulse(self ):
        '''
        Returns qubit and alpha(t)
        '''
        # Two arrays for qubit pulse: 
        '''
        In ECD, [1] stores real and [2] stores imag
        In CGrape, [1] stores X, and [2] stores Y
        '''
        self.qubit_pulse1 = [[] for _ in range(self.n_q)]
        self.qubit_pulse2 = [[] for _ in range(self.n_q)]
        self.alphas = [[] for _ in range(self.N_modes)]

        self.qubit_pulse1[0].append([])   # no such thing as sigma_x_gg
        self.qubit_pulse2[0].append([])
        self.qubit_pulse1[1].append([])   # no such thing as sigma_x_eg # will only consider ge
        self.qubit_pulse2[1].append([])
        self.qubit_pulse1[1].append([])   # no such thing as sigma_x_ee
        self.qubit_pulse2[1].append([])

        if self.method == 'cgrape': 
            pulses = self.interpolate_circ_grape_pulses()
            #return pulses

            time_steps= len(pulses[0])
            self.alphas = np.array([np.array([30 for _ in range(time_steps)]) for __ in range(self.N_modes)])

            self.qubit_pulse1[0].append(pulses[0])   # sigma_x_ge
            self.qubit_pulse2[0].append(pulses[1])   # sigma_y_ge

            if self.version == 'gef':
                self.qubit_pulse1[1].append(pulses[2])   # sigma_x_ef
                self.qubit_pulse2[1].append(pulses[3])


        if self.method == 'ecd': 

            pulse_sim = ecd_pulse_multimode(param_file = self.filename,
                              kappa = [0,0],
                               N_modes= self.N_modes, 
                               version = self.version,
                               chis = self.chis/ (2* np.pi),
                                           )
                              # is_gf = False)
            pulse_sim.get_pulses()
            self.alphas = pulse_sim.alpha

            if self.version == 'ge':
                self.qubit_pulse1[0].append(np.conjugate(pulse_sim.qubit_dac_pulse_GHz))
                self.qubit_pulse2[0].append(pulse_sim.qubit_dac_pulse_GHz)

            elif self.version == 'gf':
                self.qubit_pulse1[0].append([])
                self.qubit_pulse2[0].append([])
                
                self.qubit_pulse1[0].append(np.conjugate(pulse_sim.qubit_dac_pulse_GHz))
                self.qubit_pulse2[0].append(pulse_sim.qubit_dac_pulse_GHz)
            #else: self.qubit_pulse[0].append([])
            
            elif self.version == 'gef': 
                
                # the ge pulse
                self.qubit_pulse1[0].append(np.conjugate(pulse_sim.qubit_dac_pulse_GHz[0]))
                self.qubit_pulse2[0].append(pulse_sim.qubit_dac_pulse_GHz[0])
                
                # the ef pulse    
                self.qubit_pulse1[1].append(np.conjugate(pulse_sim.qubit_dac_pulse_GHz[1]))
                self.qubit_pulse2[1].append(pulse_sim.qubit_dac_pulse_GHz[1])
            
            #Plot ECD pulses
            pulse_sim.plot_pulses()
            self.pulse_sim = pulse_sim
            

    def interpolate_circ_grape_pulses(self):
        '''
        Copied from Vatsan's Circ Grape Notebook
        '''
        fine_pulses = []
        #num_ops = #len(self.controlHs())
        f = hf.File(self.filename, 'r')
        num_ops = len(f['uks'][-1])
        total_time = f['total_time'][()]
        steps = f['steps'][()]
        dt = float(total_time) / steps
        fine_steps = total_time * 1 #self.SAMPLE_RATE =1 ns
        base_times = np.arange(steps + 1) * total_time / steps
        tlist = np.arange(fine_steps + 1) * total_time / fine_steps
        for i in range(num_ops):
            base_pulse = f['uks'][-1][i]  # final control pulse
            base_pulse = np.append(base_pulse, 0.0)  # add extra 0 on end of pulses for interpolation
            interpFun = interpolate.interp1d(base_times, base_pulse)
            pulse_interp = interpFun(tlist)
            fine_pulses.append(pulse_interp)

        return fine_pulses

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
        
        # Transmon state projection operators
        self.t_states = [ basis( self.n_q, j ) * (basis( self.n_q, j).dag()) for j in range(self.n_q)]
        
        #Transmon Sigma Matrices  (For Grape Code)

        #Sigma_x matrices 
        self.Q_sigmaXs = [] # sigma x for [[gg, ge, gf, ..], [eg, ee, ef, ..], ..]..
        zeroes = np.zeros((self.n_q, self.n_q))
        for num1 in range(0, self.n_q): 
            arr = []
            for num2 in range(0,self.n_q):
                sigma_x = zeroes.copy()
                sigma_x[num1, num2] = 1
                sigma_x[num2, num1] = 1
                arr.append(Qobj(sigma_x))
            self.Q_sigmaXs.append(arr)
        #print('loaded sigma Xs')
        #Sigma_y matrices 
        self.Q_sigmaYs = [] 
        for num1 in range(0, self.n_q): 
            arr = []
            for num2 in range(0,self.n_q):
                sigma_y = zeroes.copy()
                sigma_y[num2, num1] = 1      # assume num2>num1
                sigma_y[num1, num2] = -1
                arr.append(Qobj( (0+1j)* sigma_y))
            self.Q_sigmaYs.append(arr)

        # Multimode Identity Operator
        self.identity_mms = self.identity_c
        
        for _ in range(1, self.N_modes):
            
            self.identity_mms = tensor( self.identity_mms ,
                                       self.identity_c)
        
        # Multimode Operators Creation and Annhilation operators
        self.a_mms= []
        self.adag_mms = []
        
        for m in range(self.N_modes):
            
            ## Creating annihilation and creation operators for m'th mode (ignoring qubit)
            a_mm = self.a_c * (m == 0) + self.identity_c * (1 - (m == 0))
            adag_mm = self.adag_c * (m == 0) + self.identity_c * (1 - (m == 0))
            
            for n in range(1,self.N_modes):
                a_mm = tensor( a_mm , 
                                    self.a_c * (m == n) + self.identity_c * (1 - (m == n)) )
                adag_mm = tensor( adag_mm , 
                                    self.adag_c * (m == n) + self.identity_c * (1 - (m == n)) )
                
            ## Adding them to the list 
            self.a_mms.append( a_mm )
            self.adag_mms.append( adag_mm )
        
        return None
    
    def initialize_ECD_and_qubit_drive(self): 
        '''
        Adds the initial ECD time dependent displacement terms and qubit_drive to ham
        '''
        # first qubit drive 

        for t in range(self.n_q - 1): 
            for t_ in range(self.n_q):

                if t_<=t: continue # no such things as eg (ge is ok)

                #rules for diff mode (mode means whether ge, gf or gef .. like version)
                if self.version == 'ge' and ((t == 2) or (t_ == 2)): continue # ignore f 
                if self.version == 'gf' and ((t == 1) or (t_ == 1)): continue # ignore e
                if self.version == 'gef' and ((t == 0) and (t_ == 2)): continue # ignore gf

                if self.method == 'ecd':
#                     self.Hd.append([np.sqrt(t_) * tensor(basis(self.n_q, t) * basis(self.n_q, t_).dag(),
#                                             self.identity_mms), 
#                                     self.qubit_pulse1[t][t_]] )  
                    
#                     self.Hd.append([np.sqrt(t_) * tensor(basis(self.n_q, t_) * basis(self.n_q, t).dag(),
#                                             self.identity_mms), 
#                                             self.qubit_pulse2[t][t_] ])
                    self.Hd.append([ tensor(basis(self.n_q, t) * basis(self.n_q, t_).dag(),
                                            self.identity_mms), 
                                    self.qubit_pulse1[t][t_]] )  
                    
                    self.Hd.append([ tensor(basis(self.n_q, t_) * basis(self.n_q, t).dag(),
                                            self.identity_mms), 
                                            self.qubit_pulse2[t][t_] ])
#                     print('added qubit_drives')
#                     print('------------------------------------------------------------------')
#                     print(self.Hd)
                if self.method == 'cgrape':
                    self.Hd.append([tensor(self.Q_sigmaXs[t][t_],
                                            self.identity_mms), 
                                    self.qubit_pulse1[t][t_]] )  
                    
                    self.Hd.append([tensor(self.Q_sigmaYs[t][t_],
                                            self.identity_mms), 
                                            self.qubit_pulse2[t][t_] ])
          
        # Detuning terms
        if self.method == 'cgrape': 
            for m in range(self.N_modes): 
                self.H0+= self.detuning[m] * tensor(self.identity_q, self.adag_mms[m] * self.a_mms[m])
                
        elif self.method == 'ecd': #and self.version =='ge': (assuming cavity always driven at omega_g + omega_2 /2
            
            for m in range(self.N_modes):
                
                self.detuning = -1 * ( self.chis[m][0] + self.chis[m][1] ) / 2
                
                self.H0 += self.detuning * tensor(self.identity_q, self.adag_mms[m] * self.a_mms[m])
                
                self.Hd.append( [(self.detuning) * tensor( self.identity_q , self.a_mms[m] ),
                                  np.conjugate(self.alphas[m])] )
                    
                self.Hd.append( [(self.detuning)  * tensor( self.identity_q , self.adag_mms[m] ),
                                  self.alphas[m]] )
#                 print('------------------------------------------------------------------')
#                 print('added detuning')
#                 print(self.Hd)
                
        # ecd pulses
        for t in range(self.n_q):
            
            if self.version == 'gf' and t == 1: continue  # skip chi_e  terms (cuz assume e state not populated)
            if self.version == 'ge' and t == 2: continue  # skip chi_f  terms (cuz assume f state not populated)
                              
            for m in range(self.N_modes):
                 #print(t)
                 ## Bare Dispersive shift
                 self.H0 += self.chis[m][t] * tensor( self.t_states[t] , self.adag_mms[m] * self.a_mms[m] )

#                  print(self.chis[m][t] * tensor( self.t_states[t] , self.adag_mms[m] * self.a_mms[m] ))             
                 #Conditional Displacement Term
                 self.Hd.append( [(self.chis[m][t]) * tensor( self.t_states[t] , self.a_mms[m] ),
                                  np.conjugate(self.alphas[m])] )
                    
                 self.Hd.append( [(self.chis[m][t]) * tensor( self.t_states[t] , self.adag_mms[m] ),
                                  self.alphas[m]] )
                 
                #  ## Stark Shift
                 self.Hd.append( [self.chis[m][t] * tensor(self.t_states[t], self.identity_mms), 
                                 self.square_list( self.alphas[m] )] )
                    
#                  print('added cond disp')
#                  print(self.Hd)
        return None
                 
        
    def square_list(self, listy):
        '''
        Input: List of numbers [a,b,c,...]
        Output: [a^2, b^2, ...] (norm squared for complex numbers)
        '''
        return np.real( [np.real(i)**2 + np.imag(i)**2 for i in listy])
    

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
    
    # def add_cavity_relaxation(self, T1_mode1 = 10e+6, T1_mode2 = 10e+6):
    #     '''
    #     qubit relaxation (T1 in nanoseconds)
        
    #     in displaced frame, the a_c -> a_c + alpha but the alpha part
    #     can be taken care of by using classical EOM when designing pulses
    #     '''
    #     gamma_relax_mode1 = 1/T1_mode1
    #     term1 = np.sqrt(gamma_relax_mode1)*tensor(self.identity_q, self.a_c, self.identity_c)

    #     gamma_relax_mode2 = 1/T1_mode2
    #     term2 = np.sqrt(gamma_relax_mode2)*tensor(self.identity_q, self.identity_c, self.a_c)


    #     self.c_ops.append(term1)
    #     self.c_ops.append(term2)
    #     return None
    
    
    
    
    # def add_cavity_dephasing_for_given_mode(self, T1, Techo, alpha, mode_idx = 1, thermal = False, T_phi = None):
    #     '''
    #     Adds dephasing noise for a given mode (transforming the cavity dephosing noise in displaced frame)
    #     '''
    #     print('hii')
    #     #Rates 
    #     if T_phi != None: 
    #         gamma_phi = 2*(1/T_phi)
    #     else:
    #         gamma_relax= (1/T1)
    #         gamma_echo = (1/Techo)
    #         gamma_phi = gamma_echo - (gamma_relax/2)
    #     print(gamma_phi)


    #     #In transforming a -> a + alpha, the term a^dag a can be broken down as 
    #     # (a+ alpha)(a^dag + alpha^star) = a^adag + alpha^star * a + alpha* adag + |alpha|^2
    #     # the latter term can be ignored cuz equal to identity


    #     term1 = np.sqrt(gamma_phi)*tensor(self.identity_q, 
    #                                       self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                       self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)                                
    #                                      ) 
    #     self.c_ops.append(term1)


    #     term2 = np.sqrt(gamma_phi)*tensor(self.identity_q,
    #                                       self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                      self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2))
    #     self.c_ops.append([term2, np.conjugate(alpha)])

    #     term3 = np.sqrt(gamma_phi)*tensor(self.identity_q, 
    #                                       self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                       self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2))
    #     self.c_ops.append([term3, alpha])

    #     #dissipators
    #     term4 = gamma_phi * lindblad_dissipator(tensor(self.identity_q, 
    #                                                    self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                    self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ), # a+ a,  a
    #                                              tensor(self.identity_q, 
    #                                                      self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                      self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
    #                                                    )
    #                                            )
    #     coeff =  np.array([ (np.abs(k))**2 for k in alpha])
    #     self.c_ops.append([term4,coeff])
    #     #         print([ (np.abs(k))**2 for k in self.alpha])
    #     #         print(term4)

    #     term5 =gamma_phi * lindblad_dissipator(tensor(self.identity_q, 
    #                                                self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ),  # a+ a,  a+
    #                                            tensor(self.identity_q, 
    #                                                  self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                   self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)))
    #     self.c_ops.append([term5, np.array([ (np.abs(k))**2 for k in alpha])])

    #     term6 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
    #                                                      self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                      self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
    #                                                        ),                                                           #  a,  a+ a
    #                                             tensor(self.identity_q, 
    #                                                    self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                    self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ))
    #     self.c_ops.append([term6, np.array([ (np.abs(k))**2 for k in alpha])])

    #     term7 =gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
    #                                                  self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                   self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)), #  a+,  a+ a
    #                                             tensor(self.identity_q, 
    #                                                    self.num_c * (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                    self.num_c * (mode_idx == 2) + self.identity_c * (mode_idx != 2)    ))
    #     self.c_ops.append([term7, np.array([ (np.abs(k))**2 for k in alpha])])

    #     term8 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
    #                                                      self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                      self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
    #                                                        ), #  a,  a+
    #                                             tensor(self.identity_q, 
    #                                                  self.adag_c* (mode_idx != 1) + self.identity_c * (mode_idx != 1),
    #                                                   self.adag_c* (mode_idx != 2) + self.identity_c * (mode_idx != 2)))
    #     self.c_ops.append([term8, np.array([ k*k for k in np.conjugate(alpha)])])

    #     term9 = gamma_phi * lindblad_dissipator( tensor(self.identity_q, 
    #                                                  self.adag_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                   self.adag_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)), #  a+,  a
    #                                             tensor(self.identity_q, 
    #                                                      self.a_c* (mode_idx == 1) + self.identity_c * (mode_idx != 1),
    #                                                      self.a_c* (mode_idx == 2) + self.identity_c * (mode_idx != 2)
    #                                                        ))
    #     self.c_ops.append([term9, np.array([ k*k for k in alpha])])

    #     #         #add to collapse operator list
    #     #         self.c_ops.append(term1)
    #     #         self.c_ops.append([term2, np.conjugate(self.alpha)]) # adding the time dependent coefficients
    #     #         self.c_ops.append([term3, self.alpha])

    #     return None
    
    
   
    
    # def add_cavity_dephasing(self, T1_mode1 = 10e+6, Techo_mode1 = 10e+6, T1_mode2 = 10e+6, Techo_mode2 = 10e+6, thermal = None, T_phi = None):
    #     '''
    #     qubit dephasing (T1, Techo in nanoseconds)
    #     '''
       
    #     self.add_cavity_dephasing_for_given_mode( T1_mode1, Techo_mode1, self.alpha1, mode_idx = 1, thermal = thermal, T_phi = T_phi)
    #     self.add_cavity_dephasing_for_given_mode( T1_mode2, Techo_mode2, self.alpha2, mode_idx = 2, thermal = thermal, T_phi= T_phi)        
    #     return None
    
    
    def me_solve(self, nsteps = 10000, initial = None, e_ops = None): 
        '''
        Solve the master equation 
        '''
        T = len(self.alphas[0]) # total time length in nanoseconds
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
    
    
    def plot_populations_single_mode(self, figname = 'figure', title = None):
        '''
        Given output of mesolve, outputs populations with qubit as ground
        '''
        
        output_states = self.output.states
        fig, axs = plt.subplots(self.n_q, 1, figsize=(10, 2 * (self.n_q + 1)))
        probs = []
        times = [k/1000 for k in range(len(output_states))]
        max_num_levels = self.n_c # to be shown on the plot
        pops_list = []

        q_state_labels = ['|g, ',
                          '|e, ', 
                          '|f, ']

        #qubit grounded
        for q_state_index in range(self.n_q): 
            # for mode_index in range(self.n_c): 
            for mode_level_index in range(max_num_levels): 

                #target state
                target = tensor(basis(self.n_q, q_state_index), basis(self.n_c, mode_level_index))
                pops = []
                for k in range(len(output_states)): 
                    z = target.overlap(output_states[k])
                    pops.append(z.real**2 + z.imag**2)
                axs[q_state_index].plot(times, pops, label = q_state_labels[q_state_index] + str(mode_level_index) +'>')


                axs[q_state_index].set_xlabel(r"Time ($\mu$s)", fontsize = 18)
                axs[q_state_index].set_ylabel("Populations", fontsize = 18)

                axs[q_state_index].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
                pops_list.append(pops)
        # axs[0].set_ylabel("Populations", fontsize = 18)
        # axs[0].tick_params(axis = 'both', which = 'major', labelsize = '15')
        # axs[0].tick_params(axis = 'both', which = 'minor', labelsize = '15')
        # axs[1].tick_params(axis = 'both', which = 'major', labelsize = '15')
        # axs[1].tick_params(axis = 'both', which = 'minor', labelsize = '15')
    #     axs[0].set_xticks(fontsize= 10)
    #     axs[1].set_yticks(fontsize= 10)
    #     axs[0].set_yticks(fontsize= 10)
    #     plt.legend(prop={'size': 20},  fontsize = 8, loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=3, fancybox=True, shadow=True)   
        # axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        # axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        # #plt.legend(fontsize = '15')
        #fig.suptitle(title, fontsize = 15)
        plt.tight_layout()
        #fig.savefig(figname, dpi = 1000)
        return fig , pops_list
    
    def plot_populations_two_mode(self, figname = 'figure', title = None):
        '''
        Given output of mesolve, outputs populations with qubit as ground
        '''
        
        output_states = self.output.states
        fig, axs = plt.subplots(self.n_q, 1, figsize=(10, 2 * (self.n_q + 1)))
        probs = []
        times = [k/1000 for k in range(len(output_states))]
        max_num_levels = 2#self.n_c # to be shown on the plot
        pops_list = []

        q_state_labels = ['|g, ',
                          '|e, ', 
                          '|f, ']

        #qubit grounded
        for q_state_index in range(self.n_q): 
            # for mode_index in range(self.n_c): 
            for mode_level_index1 in range(max_num_levels): 
                for mode_level_index2 in range(max_num_levels): 
                    #target state
                    target = tensor(basis(self.n_q, q_state_index), 
                                    basis(self.n_c, mode_level_index1), 
                                    basis(self.n_c, mode_level_index2))
                    pops = []
                    for k in range(len(output_states)): 
                        z = target.overlap(output_states[k])
                        pops.append(z.real**2 + z.imag**2)
                    axs[q_state_index].plot(times, pops, 
                                            label = q_state_labels[q_state_index] + str(mode_level_index1) + ', ' + str(mode_level_index2) +'>')


                    axs[q_state_index].set_xlabel(r"Time ($\mu$s)", fontsize = 18)
                    axs[q_state_index].set_ylabel("Populations", fontsize = 18)

                    axs[q_state_index].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
                    pops_list.append(pops)
        # axs[0].set_ylabel("Populations", fontsize = 18)
        # axs[0].tick_params(axis = 'both', which = 'major', labelsize = '15')
        # axs[0].tick_params(axis = 'both', which = 'minor', labelsize = '15')
        # axs[1].tick_params(axis = 'both', which = 'major', labelsize = '15')
        # axs[1].tick_params(axis = 'both', which = 'minor', labelsize = '15')
    #     axs[0].set_xticks(fontsize= 10)
    #     axs[1].set_yticks(fontsize= 10)
    #     axs[0].set_yticks(fontsize= 10)
    #     plt.legend(prop={'size': 20},  fontsize = 8, loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=3, fancybox=True, shadow=True)   
        # axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        # axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        # #plt.legend(fontsize = '15')
        #fig.suptitle(title, fontsize = 15)
        plt.tight_layout()
        #fig.savefig(figname, dpi = 1000)
        return fig , pops_list
    

    def plot_matrix(self, figname = 'figure', title = None):
        '''
        Computes the unitary matrix and shows as colorplot 
        '''
        # Generate all the possible states 
        states = []
        labels = []
        q_state_labels = ['|g, ',
                          '|e, ', 
                          '|f, ']
        for q_state_index in range(self.n_q): 
            for m_state_index in range(self.n_c): 

                state = tensor(basis(self.n_q, q_state_index), basis(self.n_c, m_state_index))
                states.append(state)
               
                #labels
                label = q_state_labels[q_state_index] + str(m_state_index) + '>'
                labels.append(label)
                


        
        # Now compute the matrix
        matrix = np.zeros(shape = (len(states), len(states)))
        for i in range(len(states)): 

            #act pulse on initial
            initial_state = states[i]
            self.me_solve(initial = initial_state)
            final_state = self.output.states[-1]

            #compute dot product with all other states
            for t in range(len(states)): 
                target_state = states[t]
                matrix[i,t] = np.abs( final_state.overlap(target_state) )
            

            # plt.rcParams["figure.figsize"] = [7.50, 3.50]
            # plt.rcParams["figure.autolayout"] = True

        fig, ax = plt.subplots()
        #matrix = np.random.randint(0, 5, size=(max_val, max_val))
        cax = ax.matshow(matrix, cmap='Blues')
        cbar = fig.colorbar(cax)

        ax.set_xticklabels([''] + labels, fontsize = 10)  
        ax.set_yticklabels([''] + labels, fontsize = 10)
        print(labels)
        return matrix, labels

        



        output_states = self.output.states
        fig, axs = plt.subplots(self.n_q, 1, figsize=(10, 2 * (self.n_q + 1)))
        probs = []
        times = [k/1000 for k in range(len(output_states))]
        max_num_levels = self.n_c # to be shown on the plot
        pops_list = []

        q_state_labels = ['|g, ',
                          '|e, ', 
                          '|f, ']

        #qubit grounded
        for q_state_index in range(self.n_q): 
            # for mode_index in range(self.n_c): 
            for mode_level_index in range(max_num_levels): 

                #target state
                target = tensor(basis(self.n_q, q_state_index), basis(self.n_c, mode_level_index))
                pops = []
                for k in range(len(output_states)): 
                    z = target.overlap(output_states[k])
                    pops.append(z.real**2 + z.imag**2)
                axs[q_state_index].plot(times, pops, label = q_state_labels[q_state_index] + str(mode_level_index) +'>')


                axs[q_state_index].set_xlabel(r"Time ($\mu$s)", fontsize = 18)
                axs[q_state_index].set_ylabel("Populations", fontsize = 18)

                axs[q_state_index].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
                pops_list.append(pops)
        # axs[0].set_ylabel("Populations", fontsize = 18)
        # axs[0].tick_params(axis = 'both', which = 'major', labelsize = '15')
        # axs[0].tick_params(axis = 'both', which = 'minor', labelsize = '15')
        # axs[1].tick_params(axis = 'both', which = 'major', labelsize = '15')
        # axs[1].tick_params(axis = 'both', which = 'minor', labelsize = '15')
    #     axs[0].set_xticks(fontsize= 10)
    #     axs[1].set_yticks(fontsize= 10)
    #     axs[0].set_yticks(fontsize= 10)
    #     plt.legend(prop={'size': 20},  fontsize = 8, loc='upper center', bbox_to_anchor=(0.5, 1.05),
    #           ncol=3, fancybox=True, shadow=True)   
        # axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        # axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize = '15')
        # #plt.legend(fontsize = '15')
        #fig.suptitle(title, fontsize = 15)
        plt.tight_layout()
        #fig.savefig(figname, dpi = 1000)
        return pops_list
    
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
        max_num_levels = 5 # to be shown on the plot
        
        #qubit grounded
        for i in range(max_num_levels):
            for j in range(max_num_levels):
                target = tensor(basis(self.n_q,0), basis(self.n_c, i), basis(self.n_c, j))
                pops = []
                for k in range(len(output_states)): 
                    z = self.dot(target ,output_states[k])
                    pops.append(z)
                axs[0].plot(times, pops, label = '|g,'+str(i)+',' + str(j)+'>')
        
        #qubit excited
        for i in range(max_num_levels):
            for j in range(max_num_levels):
                target = tensor(basis(self.n_q,1), basis(self.n_c, i), basis(self.n_c, j))
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

print('hi')