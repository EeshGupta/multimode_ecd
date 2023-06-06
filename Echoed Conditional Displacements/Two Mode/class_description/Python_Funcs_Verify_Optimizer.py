from qutip import *
import h5py as hf
import numpy as np
import scipy
# def mod_disp_op(disp, n_q, n_c):
#     '''
#     Returns displacement operator using baken campbell formula
#     '''
#     pauli_like_x = (create(n_c) + destroy(n_c))
#     pauli_like_y = (1j)*(create(n_c) - destroy(n_c))
#     comm = (1/2)*((pauli_like_x*pauli_like_y) - (pauli_like_y*pauli_like_x))
#     re = np.real(disp)
#     im = np.imag(disp)

#     first = (1j*im*pauli_like_x).expm()
#     second = (-1j*re*pauli_like_y).expm()
#     third = ((im*re)*(-1)*comm).expm()
#     return first*second*third

class Calculator(): 
    def __init__(self, n_q, n_c, filename):
        '''
        n_q : # of levels in the qubit
        n_c : # of levels in cavitymode
        N_modes: num of modes
        filename: param_filename
        '''
        self.n_q = n_q
        self.n_c = n_c
        #self.N_modes = N_modes
        self.param_file = filename
        
        self.load_params()
        
        self.N_modes = len(self.betas)
        self.N_layers = len(self.betas[0])
        
        #important matrices
        self.identity = qeye(self.n_c)
        self.identity_mm = self.identity
        for i in range(1, self.N_modes):
            self.identity_mm = np.kron(self.identity_mm, self.identity)
        
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

        

    def disp_op(self, disp, mode_idx):
        '''
        Returns displacement operator for specified displacement (in numpy form)
        '''
        #disp = normalize_complex(disp)
        exponent = (disp*create(self.n_c))- (np.conjugate(disp)*destroy(self.n_c))
        d_op = (exponent.expm()).full()

        ii= mode_idx
        d = d_op*(ii==0) + self.identity*(1-(ii==0))
        for m in np.arange(1,self.N_modes):
            d = np.kron(d,d_op*(ii==m) + self.identity*(1-(ii==m)))
        return d


    def cond_disp_op(self, beta, mode):#, use_mod = False):
        '''
        Returns cond displacement operator for specified real displacement
        '''
        disp = beta/2

    #     if use_mod: 
    #         d = mod_disp_op(disp= disp, n_q = n_q, n_c = n_c) #Baker Campbell Approx
    #     else: 
        d = self.disp_op(disp= disp, mode_idx = mode)
        d_adjoint = ((Qobj(d)).dag()).full()

        left = np.kron(create(self.n_q), d) #D(beta/2)|e><g|
        right = np.kron(destroy(self.n_q), d_adjoint) #D(-beta/2)|g><e|   ...not sure d_adjoint(alpha) = d(-alpha) if use Baker
        return Qobj(left+right)

    
    def qubit_rot(self, phi, theta):
        '''
        Returns qubit rotation
        '''
        phi = phi - (np.pi/2)
        rot = (np.cos(phi)*sigmax()) + (np.sin(phi)*sigmay())
        rot = rot.full() #convert to numpy
        exp = scipy.linalg.expm((-1.0j)*(theta/2)*(rot))
       # exp = (((-1.0j)*(theta/2)*(rot)).expm()).full()
        exp_kron = np.kron(exp, self.identity_mm) 
        return  Qobj(exp_kron)

    def normalize_complex(number):
        '''
        Returns radius r of complex number z = r*e^iphi
        '''
        return np.sqrt(number.real**2 + (number .imag**2))

    def dot(self, state1, state2):
        '''
        dotting both states
        '''
        state1 = Qobj(state1.full())   # for reshaping purposes
        state2 = Qobj(state2.full())   # for reshaping purposes
        
        fid = state1.overlap(state2)
        return np.real(fid*np.conjugate(fid))
    def evolve(self, initial_state):
        '''
        Operates on initial_state with ECD(beta_n)*R(phi_n, theta_n) *...........*ECD(beta_1)*R(phi_1, theta_1)
        '''
        state = Qobj(initial_state.full())   # for reshaping purposes
        for l in range(self.N_layers):
            
            block = qeye(self.n_q * (self.n_c ** self.N_modes))
            for m in range(self.N_modes):

                beta = self.betas[m][l]
                phi = self.phis[m][l]
                theta = self.thetas[m][l]

                block_ = self.cond_disp_op(beta, mode = m)*self.qubit_rot(phi, theta)
                #print(block)
                block = block*block_
            #print(state)
                
            state = block*state
            #print(state)

        return state
        
        