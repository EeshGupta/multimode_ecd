'''
Change: can accept any number of initial params as long as they are under the required multistart count
V2: Forbidden state occupation penalty 
V3: Exact matrix multiplication functions 
V4: Exact displacement operator
'''
#%%
# note: timestamp can't use "/" character for h5 saving.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
#from selectors import EpollSelector
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # supress warnings
import h5py

print(
    "\nNeed tf version 2.3.0 or later. Using tensorflow version: "
    + tf.__version__
    + "\n"
)
import ECD_control.ECD_optimization.tf_quantum as tfq
from ECD_control.ECD_optimization.visualization import VisualizationMixin
import qutip as qt
import datetime
import time


class BatchOptimizer(VisualizationMixin):

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(
        self,
        optimization_type="state transfer",
        target_unitary=None,
        P_cav=None,
        N_cav=None,
        initial_states=None,
        target_states=None,
        expectation_operators=None,
        target_expectation_values=None,
        N_multistart=10,
        N_blocks=20,
        term_fid=0.99,  # can set >1 to force run all epochs
        dfid_stop=1e-4,  # can be set= -1 to force run all epochs
        learning_rate=0.01,
        epoch_size=10,
        epochs=100,
        beta_scale=1.0,
        alpha_scale=1.0,
        theta_scale=np.pi,
        use_displacements=False,
        no_CD_end=False,
        beta_mask=None,
        phi_mask=None,
        theta_mask=None,
        alpha_mask=None,
        name="ECD_control",
        filename=None,
        user_angles= None, #provided if manually want to set initial angles
        comment="",
        use_phase=False,  # include the phase in the optimization cost function. Important for unitaries.
        timestamps=[],
        allowable_states = [],
        forbidden_state_weight = 100,
        **kwargs
    ):
        self.parameters = {
            "optimization_type": optimization_type,
            "N_multistart": N_multistart,
            "N_blocks": N_blocks,
            "term_fid": term_fid,
            "dfid_stop": dfid_stop,
            "no_CD_end": no_CD_end,
            "learning_rate": learning_rate,
            "epoch_size": epoch_size,
            "epochs": epochs,
            "beta_scale": beta_scale,
            "alpha_scale": alpha_scale,
            "theta_scale": theta_scale,
            "use_displacements": use_displacements,
            "use_phase": use_phase,
            "name": name,
            "comment": comment,
            "user_angles": user_angles, 
            "allowable_states": allowable_states, 
            "forbidden_state_weight": forbidden_state_weight
        }
        self.allowable_states = tf.stack(
            [tf.linalg.adjoint(tfq.qt2tf(state)) for state in allowable_states])
        #print(self.allowable_states)
        self.user_angles = self.parameters["user_angles"]
        self.parameters.update(kwargs)
        if (
            self.parameters["optimization_type"] == "state transfer"
            or self.parameters["optimization_type"] == "analysis"
        ):
            self.batch_fidelities = (
                self.batch_state_transfer_fidelities
                # if self.parameters["use_phase"]
                # else self.batch_state_transfer_fidelities_real_part
            )
            # set fidelity function

            self.initial_states = tf.stack(
                [tfq.qt2tf(state) for state in initial_states]
            )

            self.target_unitary = tfq.qt2tf(target_unitary)

            # if self.target_unitary is not None: TODO
            #     raise Exception("Need to fix target_unitary multi-state transfer generation!")

            self.target_states = (  # store dag
                tf.stack([tfq.qt2tf(state) for state in target_states])
                if self.target_unitary is None
                else self.target_unitary @ self.initial_states
            )

            self.target_states_dag = tf.linalg.adjoint(
                self.target_states
            )  # store dag to avoid having to take adjoint

            N_cav = self.initial_states[0].numpy().shape[0] // 2
        elif self.parameters["optimization_type"] == "unitary":
            self.target_unitary = tfq.qt2tf(target_unitary)
            N_cav = self.target_unitary.numpy().shape[0] // 2
            P_cav = P_cav if P_cav is not None else N_cav
            raise Exception("Need to implement unitary optimization")

        elif self.parameters["optimization_type"] == "expectation":
            raise Exception("Need to implement expectation optimization")
        elif (
            self.parameters["optimization_type"] == "calculation"
        ):  # using functions but not doing opt
            pass
        else:
            raise ValueError(
                "optimization_type must be one of {'state transfer', 'unitary', 'expectation', 'analysis', 'calculation'}"
            )


        self.parameters["N_cav"] = N_cav
        if P_cav is not None:
            self.parameters["P_cav"] = P_cav

        # TODO: handle case when you pass initial params. In that case, don't randomize, but use "set_tf_vars()"
        self.randomize_and_set_vars()
        # self.set_tf_vars(betas=betas, alphas=alphas, phis=phis, thetas=thetas)

        self._construct_needed_matrices()
        self._construct_optimization_masks(beta_mask, alpha_mask, phi_mask, theta_mask)

        # opt data will be a dictionary of dictonaries used to store optimization data
        # the dictionary will be addressed by timestamps of optmization.
        # each opt will append to opt_data a dictionary
        # this dictionary will contain optimization parameters and results

        self.timestamps = timestamps
        self.filename = (
            filename
            if (filename is not None and filename is not "")
            else self.parameters["name"]
        )
        path = self.filename.split(".")
        if len(path) < 2 or (len(path) == 2 and path[-1] != ".h5"):
            self.filename = path[0] + ".h5"

    def modify_parameters(self, **kwargs):
        # currently, does not support changing optimization type.
        # todo: update for multi-state optimization and unitary optimziation
        parameters = kwargs
        for param, value in self.parameters.items():
            if param not in parameters:
                parameters[param] = value
        # handle things that are not in self.parameters:
        parameters["initial_states"] = (
            parameters["initial_states"]
            if "initial_states" in parameters
            else self.initial_states
        )
        parameters["target_states"] = (
            parameters["target_states"]
            if "target_states" in parameters
            else self.target_states
        )
        parameters["filename"] = (
            parameters["filename"] if "filename" in parameters else self.filename
        )
        parameters["timestamps"] = (
            parameters["timestamps"] if "timestamps" in parameters else self.timestamps
        )
        self.__init__(**parameters)

    def _construct_needed_matrices(self):
        N_cav = self.parameters["N_cav"]
        q = tfq.position(N_cav)  # Pauli x if N_cav = 2
        p = tfq.momentum(N_cav)  #Pauli y if N_cav = 2
        self.a= tfq.destroy(N_cav)
        self.adag= tfq.create(N_cav)

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)  # eig vals and eig vec matrix
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)

        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)

        if self.parameters["optimization_type"] == "unitary":
            P_cav = self.parameters["P_cav"]
            partial_I = np.array(qt.identity(N_cav))
            for j in range(P_cav, N_cav):
                partial_I[j, j] = 0
            partial_I = qt.Qobj(partial_I)
            self.P_matrix = tfq.qt2tf(qt.tensor(qt.identity(2), partial_I))

    def _construct_optimization_masks(
        self, beta_mask=None, alpha_mask=None, phi_mask=None, theta_mask=None
    ):
        if beta_mask is None:
            beta_mask = np.ones(
                shape=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                dtype=np.float32,
            )
            if self.parameters["no_CD_end"]:
                beta_mask[-1, :] = 0  # don't optimize final CD
        else:
            # TODO: add mask to self.parameters for saving if it's non standard!
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if alpha_mask is None:
            alpha_mask = np.ones(
                shape=(1, self.parameters["N_multistart"]), dtype=np.float32,
            )
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if phi_mask is None:
            phi_mask = np.ones(
                shape=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                dtype=np.float32,
            )
            phi_mask[0, :] = 0  # stop gradient on first phi entry
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if theta_mask is None:
            theta_mask = np.ones(
                shape=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                dtype=np.float32,
            )
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        self.beta_mask = beta_mask
        self.alpha_mask = alpha_mask
        self.phi_mask = phi_mask
        self.theta_mask = theta_mask

    @tf.function
    def test_batch_construct_displacement_operators(self, alphas):
        print(alphas)

        # Reshape amplitudes for broadcast against diagonals
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        re_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.real(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )
        im_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.imag(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )

        # Exponentiate diagonal matrices
        #expm_q = tf.linalg.diag(tf.math.exp(1.0j * im_a * self._eig_q))
        expm_q = tf.linalg.diag(tf.math.exp(1.0j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1.0j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))

        # Apply Baker-Campbell-Hausdorff
        # test = tf.cast(
        #     expm_c,
        #     dtype=tf.complex64,
        # )
        return expm_c

    @tf.function
    def batch_construct_displacement_operators(self, alphas):
        print(alphas)

        # Reshape amplitudes for broadcast against diagonals
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        re_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.real(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )
        im_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.imag(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )

        # Exponentiate diagonal matrices
        #expm_q = tf.linalg.diag(tf.math.exp(1.0j * im_a * self._eig_q))
        expm_q = tf.linalg.diag(tf.math.exp(1.0j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1.0j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))

        # Apply Baker-Campbell-Hausdorff
        # test = tf.cast(
        #     self._U_q
        #     @ expm_q
        #     @ tf.linalg.adjoint(self._U_q),
        #     dtype=tf.complex64,
        # )
        # print(test)
        return tf.cast(
            self._U_q
            @ expm_q
            @ tf.linalg.adjoint(self._U_q)
            @ self._U_p
            @ expm_p
            @ tf.linalg.adjoint(self._U_p)
            @ expm_c,
            dtype=tf.complex64,
        )

    @tf.function
    def batch_construct_exact_displacement_operators(self, alphas):
        #print(alphas)

        # Reshape amplitudes for broadcast against diagonals
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        re_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.real(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )
        im_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.imag(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )

        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1.0j * im_a * self._eig_q))
        expm_q = tf.linalg.diag(tf.math.exp(1.0j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1.0j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))

        # Apply Baker-Campbell-Hausdorff
        # test = tf.cast(
        #     self._U_q
        #     @ expm_q
        #     @ tf.linalg.adjoint(self._U_q),
        #     dtype=tf.complex64,
        # )
        # print(test)
        # print(alphas.shape)
        # print(im_a.shape)
        # print((im_a*self._eig_q).shape)
        # print(expm_q.shape)
        alphas_star = tf.math.conj(alphas)
        exponent = tf.einsum('ij,kl->ijkl', alphas, self.adag) - tf.einsum('ij,kl->ijkl', alphas_star, self.a)
        return tf.linalg.expm(exponent)
        #return tf.math.exp((alphas*))
        #return tf.cast(
        #     self._U_q
        #     @ expm_q
        #     @ tf.linalg.adjoint(self._U_q)
        #     @ self._U_p
        #     @ expm_p
        #     @ tf.linalg.adjoint(self._U_p)
        #     @ expm_c,
        #     dtype=tf.complex64,
        # )

    @tf.function
    def batch_construct_block_operators(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        # conditional displacements
        Bs = (
            tf.cast(betas_rho, dtype=tf.complex64)
            / tf.constant(2, dtype=tf.complex64)
            * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(betas_angle, dtype=tf.complex64)
            )
        )

        # final displacement
        D = tf.cast(alphas_rho, dtype=tf.complex64) * tf.math.exp(
            tf.constant(1j, dtype=tf.complex64)
            * tf.cast(alphas_angle, dtype=tf.complex64)
        )

        ds_end = self.batch_construct_exact_displacement_operators(D)
        ds_g = self.batch_construct_exact_displacement_operators(Bs)
        ds_e = tf.linalg.adjoint(ds_g)

        #EG: IDK why this phase is added; -1j from qubit pi pulse is accounted for already later in blocks coonstruction
        # Phis = phis - tf.constant(np.pi, dtype=tf.float32) / tf.constant(
        #     2, dtype=tf.float32
        # )
        Phis = phis
        Thetas = thetas / tf.constant(2, dtype=tf.float32)
        Phis = tf.cast(
            tf.reshape(Phis, [Phis.shape[0], Phis.shape[1], 1, 1]), dtype=tf.complex64
        )
        Thetas = tf.cast(
            tf.reshape(Thetas, [Thetas.shape[0], Thetas.shape[1], 1, 1]),
            dtype=tf.complex64,
        )

        exp = tf.math.exp(tf.constant(1j, dtype=tf.complex64) * Phis)
        exp_dag = tf.linalg.adjoint(exp)
        cos = tf.math.cos(Thetas)
        sin = tf.math.sin(Thetas)

        # constructing the blocks of the matrix
        ul = cos * ds_g
        ll = tf.constant(-1j, dtype=tf.complex64)* exp * sin * ds_e
        ur = tf.constant(-1j, dtype=tf.complex64) * exp_dag * sin * ds_g
        lr = cos * ds_e

        # without pi pulse, block matrix is:
        # (ul, ur)
        # (ll, lr)
        # however, with pi pulse included:
        # (ll, lr)
        # (ul, ur)
        # pi pulse also adds -i phase, however don't need to trck it unless using multiple oscillators.a
        # append a final block matrix with a single displacement in each quadrant
        blocks = tf.concat(
            [
                -1j * tf.concat([tf.concat([ll, lr], 3), tf.concat([ul, ur], 3)], 2),
                tf.concat(
                    [
                        tf.concat([ds_end, tf.zeros_like(ds_end)], 3),
                        tf.concat([tf.zeros_like(ds_end), ds_end], 3),
                    ],
                    2,
                ),
            ],
            0,
        )
        return  tf.concat([tf.concat([ll, lr], 3), tf.concat([ul, ur], 3)], 2)#, ds_g, ds_e#, ll

    # batch computation of <D>
    # todo: handle non-pure states (rho)
    def characteristic_function(self, psi, betas):
        psi = tfq.qt2tf(psi)
        betas_flat = betas.flatten()
        betas_tf = tf.constant(
            [betas_flat]
        )  # need to add extra dimension since it usually batches circuits
        Ds = tf.squeeze(self.batch_construct_displacement_operators(betas_tf))
        num_pts = betas_tf.shape[1]
        psis = tf.constant(np.array([psi] * num_pts))
        C = tf.linalg.adjoint(psis) @ Ds @ psis
        return np.squeeze(C.numpy()).reshape(betas.shape)

    def characteristic_function_rho(self, rho, betas):
        rho = tfq.qt2tf(rho)
        betas_flat = betas.flatten()
        betas_tf = tf.constant(
            [betas_flat]
        )  # need to add extra dimension since it usually batches circuits
        Ds = tf.squeeze(self.batch_construct_displacement_operators(betas_tf))
        num_pts = betas_tf.shape[1]
        rhos = tf.constant(np.array([rho] * num_pts))
        C = tf.linalg.trace(Ds @ rhos)
        return np.squeeze(C.numpy()).reshape(betas.shape)

    """
    @tf.function
    def state(
        self,
        i=0,
        betas_rho=None,
        betas_angle=None,
        alphas_rho=None,
        alphas_angle=None,
        phis=None,
        thetas=None,
    ):
        betas_rho = self.betas_rho if betas_rho is None else betas_rho
        betas_angle = self.betas_angle if betas_angle is None else betas_angle
        alphas_rho = self.alphas_rho if alphas_rho is None else alphas_rho
        alphas_angle = self.alphas_angle if alphas_angle is None else alphas_angle
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        if self.parameters["use_displacements"]:
            bs = self.construct_block_operators_with_displacements(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            )
        else:
            bs = self.construct_block_operators(betas_rho, betas_angle, phis, thetas)
        psi = self.initial_states[0]
        for U in bs[:i]:
            psi = U @ psi
        return psi
    """

    @tf.function
    def batch_state_transfer_fidelities(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        bs = self.batch_construct_block_operators(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        psis = tf.stack([self.initial_states] * self.parameters["N_multistart"])
        inter_layer_vecs = tf.TensorArray(dtype = psis.dtype, size=0, dynamic_size=True)
        for U in bs:
            psis = tf.einsum(
                "mij,msjk->msik", U, psis
            )  # m: multistart, s:multiple states (when have multiple initial states)
            inter_layer_vecs = inter_layer_vecs.write(inter_layer_vecs.size(), psis) 

        #print(inter_layer_vecs)
        inter_layer_vecs = inter_layer_vecs.stack()

        overlaps = self.target_states_dag @ psis  # broadcasting
        overlaps = tf.reduce_mean(overlaps, axis=1) # here they get rid of starting states
        overlaps = tf.squeeze(overlaps)
        # squeeze after reduce_mean which uses axis=1,
        # which will not exist if squeezed before for single state transfer
        fids = tf.cast(overlaps * tf.math.conj(overlaps), dtype=tf.float32)

        
       

        return fids, inter_layer_vecs#, psis

    # here, including the relative phase in the cost function by taking the real part of the overlap then squaring it.
    # need to think about how this is related to the fidelity.
    # @tf.function
    # def batch_state_transfer_fidelities_real_part(
    #     self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    # ):
    #     bs = self.batch_construct_block_operators(
    #         betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    #     )
    #     total_fids = None
    #     for k in range(len(self.initial_states)):
    #         initial_state = self.initial_states[i]
    #         target_state_dag = self.target_states_dag[i]
    #         psis = tf.stack([initial_state] * self.parameters["N_multistart"])
    #         for U in bs:
    #             psis = tf.einsum(
    #                 "mij,msjk->msik", U, psis
    #             )  # m: multistart, s:multiple states
    #         overlaps = target_state_dag @ psis  # broadcasting
    #         overlaps = tf.reduce_mean(tf.math.real(overlaps), axis=1)
    #         overlaps = tf.squeeze(overlaps)
    #         # squeeze after reduce_mean which uses axis=1,
    #         # which will not exist if squeezed before for single state transfer
    #         # don't need to take the conjugate anymore
    #         fids = tf.cast(overlaps * overlaps, dtype=tf.float32)
    #         if total_fids is None:
    #             total_fids = fids
    #         else: 
    #             total_fids += fids
    #     return (1/len(self.initial_states))*total_fids

    @tf.function
    def higher_fock_space_penalty_all_layers(self, inter_layer_vecs):
        '''
        Computes penalty for all intermediate states

        Currently only supports 1 starting state; if more than one, just need to divide penalty by that amount
        '''
        if self.allowable_states != None:
            penalty =  tf.TensorArray(dtype = tf.float32, size=0, dynamic_size=True)
            for layer_vec in inter_layer_vecs:
                penalty= penalty.write(penalty.size(), self.higher_fock_space_penalty_per_state(layer_vec))
                # if penalty == None: 
                #     penalty = self.higher_fock_space_penalty(layer_vec)
                # else: 
                #     penalty = tf.sum(penalty, self.higher_fock_space_penalty(layer_vec))
            
            penalty = penalty.stack()
            penalty = tf.reduce_sum(penalty, axis = 0) # add up all layers
            penalty = tf.reduce_sum(penalty, axis = 1) # add up all starting states
            penalty = tf.multiply(penalty, 
                                      tf.cast(1/self.parameters["N_blocks"], dtype = tf.float32)
                                      )

            return penalty
    @tf.function
    def higher_fock_space_penalty_per_state(self, given_states): 
        '''
        Computes the penalty for "how much" of the given states live in unallowable space
        '''

        dot_prods= tf.einsum(
                "tij,msjk->tmsik", self.allowable_states, given_states
            ) 
        #t is number of allowable states
        dot_prods = tf.reduce_sum(dot_prods, axis = 4) # get rid of k
        dot_prods = tf.reduce_sum(dot_prods, axis = 3) # get rid of i 
        # taking mod ^2 now
        mod_dot_prods = tf.math.multiply(dot_prods, tf.math.conj(dot_prods))
        mod_dot_prods = tf.cast(mod_dot_prods, dtype = tf.float32)
        cost = 1 - tf.reduce_sum(mod_dot_prods, axis = 0)
        #print(cost)

        
        return cost

    # @tf.function
    # def batch_state_transfer_fidelities_real_part(
    #     self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    # ):
    #     bs = self.batch_construct_block_operators(
    #         betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    #     )
    #     psis = tf.stack([self.initial_states] * self.parameters["N_multistart"])
    #     inter_layer_vecs = []
    #     initial = True # whether first layer or not
    #     for U in bs:
    #         psis = tf.einsum(
    #             "mij,msjk->msik", U, psis
    #         )  # m: multistart, s:multiple states (when have multiple initial states)
    #         if initial: 
    #             initial = False # for later layers
    #             inter_layer_vecs = [psis]
    #         else: 
    #             inter_layer_vecs.append(psis)

    #     overlaps = self.target_states_dag @ psis  # broadcasting
    #     # tf.compat.v1.enable_eager_execution(
    #     #         config=None, device_policy=None, execution_mode=None)

    #     #EG: this makes no sense, taking real part of overlap and squaring it 
    #     overlaps = tf.reduce_mean(tf.math.real(overlaps), axis=1)
    #     #overlaps = tf.reduce_mean(overlaps, axis=1) # ignoring the real part
    #     overlaps = tf.squeeze(overlaps)

    #     # squeeze after reduce_mean which uses axis=1,
    #     # which will not exist if squeezed before for single state transfer
    #     # don't need to take the conjugate anymore
    #     fids = tf.cast(overlaps * overlaps, dtype=tf.float32)
    #     return fids, overlaps

    @tf.function
    def mult_bin_tf(self, a):
        while a.shape[0] > 1:
            if a.shape[0] % 2 == 1:
                a = tf.concat(
                    [a[:-2], [tf.matmul(a[-2], a[-1])]], 0
                )  # maybe there's a faster way to deal with immutable constants
            a = tf.matmul(a[::2, ...], a[1::2, ...])
        return a[0]

    @tf.function
    def U_tot(self,):
        bs = self.batch_construct_block_operators(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        # U_c = tf.scan(lambda a, b: tf.matmul(b, a), bs)[-1]
        U_c = self.mult_bin_tf(
            tf.reverse(bs, axis=[0])
        )  # [U_1,U_2,..] -> [U_N,U_{N-1},..]-> U_N @ U_{N-1} @ .. @ U_1
        # U_c = self.I
        # for U in bs:
        #     U_c = U @ U_c
        return U_c

    """
    @tf.function
    def unitary_fidelity(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        U_circuit = self.U_tot(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        D = tf.constant(self.parameters["P_cav"] * 2, dtype=tf.complex64)
        overlap = tf.linalg.trace(
            tf.linalg.adjoint(self.target_unitary) @ self.P_matrix @ U_circuit
        )
        return tf.cast(
            (1.0 / D) ** 2 * overlap * tf.math.conj(overlap), dtype=tf.float32
        )
    """

    def optimize(self, do_prints=True):

        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        self.timestamps.append(timestamp)
        print("Start time: " + timestamp)
        # start time
        start_time = time.time()
        optimizer = tf.optimizers.Adam(self.parameters["learning_rate"])
        if self.parameters["use_displacements"]:
            variables = [
                self.betas_rho,
                self.betas_angle,
                self.alphas_rho,
                self.alphas_angle,
                self.phis,
                self.thetas,
            ]
        else:
            variables = [self.betas_rho, self.betas_angle, self.phis, self.thetas]

        @tf.function
        def entry_stop_gradients(target, mask):
            mask_h = tf.abs(mask - 1)
            return tf.stop_gradient(mask_h * target) + mask * target

        """
        if self.optimize_expectation:

            @tf.function
            def loss_fun(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            ):
                expect = self.expectation_value(
                    betas_rho,
                    betas_angle,
                    alphas_rho,
                    alphas_angle,
                    phis,
                    thetas,
                    self.O,
                )
                return tf.math.log(1 - tf.math.real(expect))

        if self.unitary_optimization:
            if self.unitary_optimization == "states":

                @tf.function
                def loss_fun(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                ):
                    fid = self.unitary_fidelity_state_decomp(
                        betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                    )
                    return tf.math.log(1 - fid)

            else:

                @tf.function
                def loss_fun(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                ):
                    fid = self.unitary_fidelity(
                        betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                    )
                    return tf.math.log(1 - fid)

        else:

            @tf.function
            def loss_fun(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            ):
                fid = self.state_fidelity(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                )e
                return tf.math.log(1 - fid)
        """

        @tf.function
        def loss_fun(fids, inter_layer_vecs):
            # I think it's important that the log is taken before the avg
            infid = 1-fids
            if self.allowable_states != None: 
                infid = infid + tf.multiply(self.higher_fock_space_penalty_all_layers(inter_layer_vecs),
                                            tf.cast(self.parameters["forbidden_state_weight"], dtype = tf.float32))
            losses = tf.math.log(infid)
            avg_loss = tf.reduce_sum(losses) / self.parameters["N_multistart"]
            return avg_loss

        def callback_fun(obj, fids, inter_layer_vecs, dfids, epoch):
            elapsed_time_s = time.time() - start_time
            time_per_epoch = elapsed_time_s / epoch if epoch is not 0 else 0.0
            epochs_left = self.parameters["epochs"] - epoch
            expected_time_remaining = epochs_left * time_per_epoch
            fidelities_np = np.squeeze(np.array(fids))
            penalties_np  = self.higher_fock_space_penalty_all_layers(inter_layer_vecs=inter_layer_vecs)
            betas_np, alphas_np, phis_np, thetas_np = self.get_numpy_vars()
            if epoch == 0:
                self._save_optimization_data(
                    timestamp,
                    fidelities_np,
                    penalties_np,
                    betas_np,
                    alphas_np,
                    phis_np,
                    thetas_np,
                    elapsed_time_s,
                    append=False,
                )
            else:
                self._save_optimization_data(
                    timestamp,
                    fidelities_np,
                    penalties_np,
                    betas_np,
                    alphas_np,
                    phis_np,
                    thetas_np,
                    elapsed_time_s,
                    append=True,
                )
            #print('inside callback function')
            #print(fids)
            avg_fid = tf.reduce_sum(fids) / self.parameters["N_multistart"]
            max_fid = tf.reduce_max(fids)
            avg_dfid = tf.reduce_sum(dfids) / self.parameters["N_multistart"]
            max_dfid = tf.reduce_max(dfids)
            penalties = self.higher_fock_space_penalty_all_layers(inter_layer_vecs)
            avg_penalty = tf.reduce_sum(penalties) / self.parameters["N_multistart"]
            extra_string = " (real part)" if self.parameters["use_phase"] else ""
            if do_prints:
                print(
                    "\r Epoch: %d / %d Avg Penalty: %.6f Max Fid: %.6f Avg Fid: %.6f Max dFid: %.6f Avg dFid: %.6f"
                    % (
                        epoch,
                        self.parameters["epochs"],
                        avg_penalty,
                        max_fid,
                        avg_fid,
                        max_dfid,
                        avg_dfid,
                    )
                    + " Elapsed time: "
                    + str(datetime.timedelta(seconds=elapsed_time_s))
                    + " Remaing time: "
                    + str(datetime.timedelta(seconds=expected_time_remaining))
                    + extra_string,
                    end="",
                )

        initial_fids, initial_inter_layer_vecs = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        fids = initial_fids
        inter_layer_vecs = initial_inter_layer_vecs
        callback_fun(self, fids, inter_layer_vecs, 0, 0)
        for epoch in range(self.parameters["epochs"] + 1)[1:]:
            for _ in range(self.parameters["epoch_size"]):
                with tf.GradientTape() as tape:
                    betas_rho = entry_stop_gradients(self.betas_rho, self.beta_mask)
                    betas_angle = entry_stop_gradients(self.betas_angle, self.beta_mask)
                    if self.parameters["use_displacements"]:
                        alphas_rho = entry_stop_gradients(
                            self.alphas_rho, self.alpha_mask
                        )
                        alphas_angle = entry_stop_gradients(
                            self.alphas_angle, self.alpha_mask
                        )
                    else:
                        alphas_rho = self.alphas_rho
                        alphas_angle = self.alphas_angle
                    phis = entry_stop_gradients(self.phis, self.phi_mask)
                    thetas = entry_stop_gradients(self.thetas, self.theta_mask)
                    new_fids, new_inter_layer_vecs = self.batch_fidelities(
                        betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas,
                    )
                    new_loss = loss_fun(new_fids, new_inter_layer_vecs)
                    dloss_dvar = tape.gradient(new_loss, variables)
                optimizer.apply_gradients(zip(dloss_dvar, variables))
            dfids = new_fids - fids
            fids = new_fids
            inter_layer_vecs = new_inter_layer_vecs
            callback_fun(self, fids,inter_layer_vecs, dfids, epoch)
            condition_fid = tf.greater(fids, self.parameters["term_fid"])
            condition_dfid = tf.greater(dfids, self.parameters["dfid_stop"])
            if tf.reduce_any(condition_fid):
                print("\n\n Optimization stopped. Term fidelity reached.\n")
                termination_reason = "term_fid"
                break
            if not tf.reduce_any(condition_dfid):
                print("\n max dFid: %6f" % tf.reduce_max(dfids).numpy())
                print("dFid stop: %6f" % self.parameters["dfid_stop"])
                print("\n\n Optimization stopped.  No dfid is greater than dfid_stop\n")
                termination_reason = "dfid"
                break

        if epoch == self.parameters["epochs"]:
            termination_reason = "epochs"
            print(
                "\n\nOptimization stopped.  Reached maximum number of epochs. Terminal fidelity not reached.\n"
            )
       #_termination_reason(timestamp, termination_reason)
        timestamp_end = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        elapsed_time_s = time.time() - start_time
        epoch_time_s = elapsed_time_s / epoch
        step_time_s = epoch_time_s / self.parameters["epochs"]
        self.print_info()
        print("all data saved as: " + self.filename)
        print("termination reason: " + termination_reason)
        print("optimization timestamp (start time): " + timestamp)
        print("timestamp (end time): " + timestamp_end)
        print("elapsed time: " + str(datetime.timedelta(seconds=elapsed_time_s)))
        print(
            "Time per epoch (epoch size = %d): " % self.parameters["epoch_size"]
            + str(datetime.timedelta(seconds=epoch_time_s))
        )
        print(
            "Time per Adam step (N_multistart = %d, N_cav = %d): "
            % (self.parameters["N_multistart"], self.parameters["N_cav"])
            + str(datetime.timedelta(seconds=step_time_s))
        )
        print(END_OPT_STRING)
        return timestamp

    # if append is True, it will assume the dataset is already created and append only the
    # last aquired values to it.
    # TODO: if needed, could use compression when saving data.
    

    def _save_optimization_data(
        self,
        timestamp,
        fidelities_np,
        penalties_np,
        betas_np,
        final_disp_np,
        phis_np,
        thetas_np,
        elapsed_time_s,
        append,
    ):
        # print('size of penalties')
        # print(penalties_np.shape)
        if not append:
            with h5py.File(self.filename, "a") as f:
                grp = f.create_group(timestamp)
                # for parameter, value in self.parameters.items():
                #     grp.attrs[parameter] = value
                grp.attrs["termination_reason"] = "outside termination"
                grp.attrs["elapsed_time_s"] = elapsed_time_s
                if self.target_unitary is not None:
                    grp.create_dataset(
                        "target_unitary", data=self.target_unitary.numpy()
                    )
                grp.create_dataset("initial_states", data=self.initial_states.numpy())
                grp.create_dataset("target_states", data=self.target_states.numpy())
                # dims = [[2, int(self.initial_states[0].numpy().shape[0] / 2)], [1, 1]]
                grp.create_dataset(
                    "fidelities",
                    chunks=True,
                    data=[fidelities_np],
                    maxshape=(None, self.parameters["N_multistart"]),
                )
                grp.create_dataset(
                    "penalties",
                    chunks=True,
                    data=[penalties_np],
                    maxshape=(None, self.parameters["N_multistart"]),
                )
                grp.create_dataset(
                    "betas",
                    data=[betas_np],
                    chunks=True,
                    maxshape=(
                        None,
                        self.parameters["N_multistart"],
                        self.parameters["N_blocks"],
                    ),
                )
                grp.create_dataset(
                    "final_disp",
                    data=[final_disp_np],
                    chunks=True,
                    maxshape=(None, self.parameters["N_multistart"], 1,),
                )
                grp.create_dataset(
                    "phis",
                    data=[phis_np],
                    chunks=True,
                    maxshape=(
                        None,
                        self.parameters["N_multistart"],
                        self.parameters["N_blocks"],
                    ),
                )
                grp.create_dataset(
                    "thetas",
                    data=[thetas_np],
                    chunks=True,
                    maxshape=(
                        None,
                        self.parameters["N_multistart"],
                        self.parameters["N_blocks"],
                    ),
                )
        else:  # just append the data
            with h5py.File(self.filename, "a") as f:
                f[timestamp]["fidelities"].resize(
                    f[timestamp]["fidelities"].shape[0] + 1, axis=0
                )
                f[timestamp]["penalties"].resize(
                    f[timestamp]["penalties"].shape[0] + 1, axis=0
                )
                f[timestamp]["betas"].resize(f[timestamp]["betas"].shape[0] + 1, axis=0)
                f[timestamp]["final_disp"].resize(
                    f[timestamp]["final_disp"].shape[0] + 1, axis=0
                )
                f[timestamp]["phis"].resize(f[timestamp]["phis"].shape[0] + 1, axis=0)
                f[timestamp]["thetas"].resize(
                    f[timestamp]["thetas"].shape[0] + 1, axis=0
                )

                f[timestamp]["fidelities"][-1] = fidelities_np
                f[timestamp]["penalties"][-1] = penalties_np
                f[timestamp]["betas"][-1] = betas_np
                f[timestamp]["final_disp"][-1] = final_disp_np
                f[timestamp]["phis"][-1] = phis_np
                f[timestamp]["thetas"][-1] = thetas_np
                f[timestamp].attrs["elapsed_time_s"] = elapsed_time_s

    def insert_user_initial_param(self, current_angles, user_angles_specfic):
        '''
        Input: Current angles are  N_blocks x N_multistarts shape and user angles have N_something x N_blocks shape
        Output: Replaces some of current angles with user's initial angles
        
        '''
       # print(user_angles_specfic)
        N_provided_ms = len(user_angles_specfic[0])
        N_blocks = self.parameters["N_blocks"]
        
        for i in range(N_blocks):
            for j in range(N_provided_ms):
                try:
                    current_angles[i][j] = user_angles_specfic[i][j]
                except IndexError: 
                    #Checking if this happens because of N_blocks mismatch
                    print('Index Error happened, user angles had '+ str(len(user_angles_specfic[0]))+' blocks but required ' + str(N_blocks))
                    current_angles[i][j] = 0.0
                
        return current_angles
    
    def insert_angles_in_layers(self, good_angles, angles_to_insert, index, indexx,current_insertion, total_insertions):
        '''
        Inserts angles_to_insert[indexx] into good_angles[index]

        First index is for identifying whether angle is beta, phi, theta, etc.
        Second index is for angles that correspond to some fidelity > threshold
        Helper function for return_good_angels
        '''
        n = self.parameters["N_blocks"]
        if len(good_angles[index]) is not n:
            array = np.zeros(shape =(n, total_insertions))
            #array = [[] for i in range(n)]
            good_angles[index] = array
        for i in range(n):
            good_angles[index][i][current_insertion] = angles_to_insert[i][indexx]
    
        return good_angles

    def return_good_angles(self, threshold): 
        '''
        Output: Returns angles with fidelities abpve a certain threshold
        '''
        fids, new_inter_layer_vecs = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        good_angles = [[] for i in range(6)] # one array for each parameter
        #First count how many angles to be inserted
        n = 0
        for k in range(len(fids)): 
            fid = fids[k]
            if fid> threshold:
                n+=1


        p= 0 #tracks index of inserted angle into good angles
        for i in range(len(fids)):
            fid = fids[i]
            if fid> threshold:
                good_angles = self.insert_angles_in_layers( good_angles,self.betas_rho.numpy() , 0, i ,p,n)
                good_angles  = self.insert_angles_in_layers( good_angles,self.betas_angle.numpy() , 1, i,p,n )
                #good_angles = insert_angles_in_layers(self, good_angles,self.betas_rho.numpy() , 2, i )
                #good_angles = insert_angles_in_layers(self, good_angles,self.betas_rho.numpy() , 3, i )
                good_angles = self.insert_angles_in_layers( good_angles,self.phis.numpy() , 4, i ,p,n)
                good_angles = self.insert_angles_in_layers( good_angles,self.thetas.numpy() , 5, i ,p,n)
                p+=1
        return np.array(good_angles)
        
    
    def randomize_and_set_vars(self):
        beta_scale = self.parameters["beta_scale"]
        alpha_scale = self.parameters["alpha_scale"]
        theta_scale = self.parameters["theta_scale"]
        betas_rho = np.random.uniform(
            0,
            beta_scale,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        betas_angle = np.random.uniform(
            -np.pi,
            np.pi,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        if self.parameters["use_displacements"]:
            alphas_rho = np.random.uniform(
                0, alpha_scale, size=(1, self.parameters["N_multistart"]),
            )
            alphas_angle = np.random.uniform(
                -np.pi, np.pi, size=(1, self.parameters["N_multistart"]),
            )
        phis = np.random.uniform(
            -np.pi,
            np.pi,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        thetas = np.random.uniform(
            -1 * theta_scale,
            theta_scale,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        
        #Now we replace some of the randomized param with those provided by the user
        #print('hello')
        print(self.user_angles)
        if self.user_angles is not None:
            #print(self.user_angles)
            betas_rho = self.insert_user_initial_param(betas_rho, self.user_angles[0])
            betas_angle = self.insert_user_initial_param(betas_angle, self.user_angles[1])
            #alphas_rho = self.insert_user_initial_param(alphas_rho, self.user_angles[2])
            #alphas_angle = self.insert_user_initial_param(alphas_angle, self.user_angles[3])
            phis = self.insert_user_initial_param(phis, self.user_angles[4])
            thetas = self.insert_user_initial_param(thetas, self.user_angles[5])
        
        phis[0] = 0  # everything is relative to first phi
        if self.parameters["no_CD_end"]:
            betas_rho[-1] = 0
            betas_angle[-1] = 0
        self.betas_rho = tf.Variable(
            betas_rho, dtype=tf.float32, trainable=True, name="betas_rho",
        )
        self.betas_angle = tf.Variable(
            betas_angle, dtype=tf.float32, trainable=True, name="betas_angle",
        )
        if self.parameters["use_displacements"]:
            self.alphas_rho = tf.Variable(
                alphas_rho, dtype=tf.float32, trainable=True, name="alphas_rho",
            )
            self.alphas_angle = tf.Variable(
                alphas_angle, dtype=tf.float32, trainable=True, name="alphas_angle",
            )
        else:
            self.alphas_rho = tf.constant(
                np.zeros(shape=((1, self.parameters["N_multistart"]))),
                dtype=tf.float32,
            )
            self.alphas_angle = tf.constant(
                np.zeros(shape=((1, self.parameters["N_multistart"]))),
                dtype=tf.float32,
            )
        self.phis = tf.Variable(
            phis, dtype=tf.float32, trainable=True, name="betas_rho",
        )
        self.thetas = tf.Variable(
            thetas, dtype=tf.float32, trainable=True, name="betas_angle",
        )

    def get_numpy_vars(
        self,
        betas_rho=None,
        betas_angle=None,
        alphas_rho=None,
        alphas_angle=None,
        phis=None,
        thetas=None,
    ):
        betas_rho = self.betas_rho if betas_rho is None else betas_rho
        betas_angle = self.betas_angle if betas_angle is None else betas_angle
        alphas_rho = self.alphas_rho if alphas_rho is None else alphas_rho
        alphas_angle = self.alphas_angle if alphas_angle is None else alphas_angle
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas

        betas = betas_rho.numpy() * np.exp(1j * betas_angle.numpy())
        alphas = alphas_rho.numpy() * np.exp(1j * alphas_angle.numpy())
        phis = phis.numpy()
        thetas = thetas.numpy()
        # now, to wrap phis and thetas so it's in the range [-pi, pi]
        phis = (phis + np.pi) % (2 * np.pi) - np.pi
        thetas = (thetas + np.pi) % (2 * np.pi) - np.pi

        # these will have shape N_multistart x N_blocks
        return betas.T, alphas.T, phis.T, thetas.T

    def set_tf_vars(self, betas=None, alphas=None, phis=None, thetas=None):
        # reshaping for N_multistart = 1
        if betas is not None:
            if len(betas.shape) < 2:
                betas = betas.reshape(betas.shape + (1,))
                self.parameters["N_multistart"] = 1
            betas_rho = np.abs(betas)
            betas_angle = np.angle(betas)
            self.betas_rho = tf.Variable(
                betas_rho, dtype=tf.float32, trainable=True, name="betas_rho"
            )
            self.betas_angle = tf.Variable(
                betas_angle, dtype=tf.float32, trainable=True, name="betas_angle",
            )
        if alphas is not None:
            if len(alphas.shape) < 2:
                alphas = alphas.reshape(alphas.shape + (1,))
                self.parameters["N_multistart"] = 1
            alphas_rho = np.abs(alphas)
            alphas_angle = np.angle(alphas)
            if self.parameters["use_displacements"]:
                self.alphas_rho = tf.Variable(
                    alphas_rho, dtype=tf.float32, trainable=True, name="alphas_rho",
                )
                self.alphas_angle = tf.Variable(
                    alphas_angle, dtype=tf.float32, trainable=True, name="alphas_angle",
                )
            else:
                self.alphas_rho = tf.constant(
                    np.zeros(shape=((1, self.parameters["N_multistart"],))),
                    dtype=tf.float32,
                )
                self.alphas_angle = tf.constant(
                    np.zeros(shape=((1, self.parameters["N_multistart"],))),
                    dtype=tf.float32,
                )

        if phis is not None:
            if len(phis.shape) < 2:
                phis = phis.reshape(phis.shape + (1,))
                self.parameters["N_multistart"] = 1
            self.phis = tf.Variable(
                phis, dtype=tf.float32, trainable=True, name="betas_rho",
            )
        if thetas is not None:
            if len(thetas.shape) < 2:
                thetas = thetas.reshape(thetas.shape + (1,))
                self.parameters["N_multistart"] = 1
            self.thetas = tf.Variable(
                thetas, dtype=tf.float32, trainable=True, name="betas_angle",
            )

    def best_circuit(self):
        fids, new_inter_layer_vecs = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        #print('shape of fids')
        #print(fids.shape)
        fids = np.atleast_1d(fids.numpy())
        
        max_idx = np.argmax(fids)
        all_betas, all_alphas, all_phis, all_thetas = self.get_numpy_vars(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        max_fid = fids[max_idx]
        betas = all_betas[max_idx]
        alphas = all_alphas[max_idx]
        phis = all_phis[max_idx]
        thetas = all_thetas[max_idx]
        return {
            "fidelity": max_fid,
            "betas": betas,
            "alphas": alphas,
            "phis": phis,
            "thetas": thetas,
        }

    def all_fidelities(self):
        fids, new_inter_layer_vecs = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        return fids.numpy()

    def best_fidelity_and_corresponding_pen(self):
        fids, inter_layer_vecs = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        # print(self.betas_rho.dtype)
        # print(self.betas_angle.dtype)
        max_idx = tf.argmax(fids).numpy()
        max_fid = fids[max_idx].numpy()

        pens = self.higher_fock_space_penalty_all_layers(inter_layer_vecs)
        pen = pens[max_idx].numpy()
        return max_fid, pen

    def print_info(self):
        best_circuit = self.best_circuit()
        with np.printoptions(precision=5, suppress=True):
            for parameter, value in self.parameters.items():
                print(parameter + ": " + str(value))
            print("filename: " + self.filename)
            print("\nBest circuit parameters found:")
            print("betas:         " + str(best_circuit["betas"]))
            print("alphas:        " + str(best_circuit["alphas"]))
            print("phis (deg):    " + str(best_circuit["phis"] * 180.0 / np.pi))
            print("thetas (deg):  " + str(best_circuit["thetas"] * 180.0 / np.pi))
            print("Max Fidelity:  %.6f" % best_circuit["fidelity"])
            print("\n")
    def save_angles(self, filename):
        betas = self.best_circuit()['betas']
        phis = self.best_circuit()['phis']
        thetas = self.best_circuit()['thetas']

        params = [np.real(betas), np.imag(betas), phis, thetas]
        for i in range(len(params)):
            params[i] = [float(k) for k in params[i]]

        a_file = open(filename, "w")
        np.savetxt(a_file, params)
        a_file.close()
        return None

###################### Exact matrix multiplication code #######################

    def mod_disp_op(disp, n_q, n_c):
        '''
        Returns displacement operator using baken campbell formula
        '''
        pauli_like_x = (create(n_c) + destroy(n_c))
        pauli_like_y = (1j)*(create(n_c) - destroy(n_c))
        comm = (1/2)*((pauli_like_x*pauli_like_y) - (pauli_like_y*pauli_like_x))
        re = np.real(disp)
        im = np.imag(disp)

        first = (1j*im*pauli_like_x).expm()
        second = (-1j*re*pauli_like_y).expm()
        third = ((im*re)*(-1)*comm).expm()
        return first*second*third


    def disp_op(self, disp, n_q, n_c):
        '''
        Returns displacement operator for specified displacement
        '''
        #disp = normalize_complex(disp)
        exponent = (disp*qt.create(n_c))- (np.conjugate(disp)*qt.destroy(n_c))
        return  exponent.expm()

    def cond_disp_op(self, beta, n_q, n_c):#, use_mod):
        '''
        Returns cond displacement operator for specified displacement
        '''
        disp = beta/2
        
        #n_c = n_c1 if mode == 1 else n_c2
        
        # if use_mod: 
        #     d = mod_disp_op(disp= disp, n_q = n_q, n_c = n_c) #Baker Campbell Approx
        # else: 
        
        d = self.disp_op(disp= disp, n_q = n_q, n_c = n_c)
        d_adjoint = d.dag()
        
        #if mode == 1: 
        left = qt.tensor(qt.create(n_q), d)#, qeye(n_c2)) #D(beta/2)|e><g|
        right = qt.tensor(qt.destroy(n_q), d_adjoint)#, qeye(n_c2)) #D(-beta/2)|g><e|   ...not sure d_adjoint(alpha) = d(-alpha) if use Baker
        
        # else: #mode ==2 
        #     left = tensor(create(n_q), qeye(n_c2), d)
        #     right = tensor(destroy(n_q),  qeye(n_c2), d_adjoint)
        return left+right

    def qubit_rot(self, phi, theta, n_q, n_c):#1, n_c2):
        '''
        Returns qubit rotation
        '''
        rot = (np.cos(phi)*qt.sigmax()) + (np.sin(phi)*qt.sigmay())
        exp = (-1.0j)*(theta/2)*(rot)
        return qt.tensor(exp.expm(), qt.qeye(n_c))#, qeye(n_c2) )

    def normalize_complex(self, number):
        '''
        Returns radius r of complex number z = r*e^iphi
        '''
        return np.sqrt(number.real**2 + (number .imag**2))

    def dot(self, state1, state2):
        '''
        dotting both states
        '''
        fid = state1.overlap(state2)
        return fid*np.conjugate(fid)
    def state(self, fock, qubit_g, N): 
        '''
        Returns fock states for qubit and coupled cavity mode
        '''
        if qubit_g:
            return qt.tensor(qt.basis(2,0), qt.basis(N, fock))
        else:
            return qt.tensor(qt.basis(2,1), qt.basis(N, fock))
    def evolve(self,  betas, phis, thetas, n_q, n_c, target):
        '''
        Operates on initial_state with ECD(beta_n)*R(phi_n, theta_n) *...........*ECD(beta_1)*R(phi_1, theta_1)
        '''
        n = len(betas)
        #initial_state = qt.Qobj(self.initial_states.numpy()[0]) # assuming only 1 initial state
        initial_state = self.state(0,True, n_c)
        state = initial_state
       # print(state)
        for i in range(n):
            beta = betas[i]
            #gamma = gammas[i]
            phi = phis[i]
            theta = thetas[i]
            
    
        
            #state = cond_disp_op(gamma, n_q, n_c1,n_c2,mode = 2,use_mod = use_mod)*cond_disp_op(beta, n_q, n_c1,n_c2, mode = 1,use_mod = use_mod)*qubit_rot(phi, theta, n_q, n_c1, n_c2)*state
            state = self.cond_disp_op(beta, n_q, n_c)*self.qubit_rot(phi, theta, n_q, n_c)*state
            
        return state, self.dot(state, self.state(target, True, n_c))

    def exact_fids_for_multistarts(self, n_q, n_c, target): 
        '''
        target is fock number for target state

        Returns exact fidelities (exact in terms of how displacement operator 
        is realized) for all multistarts from the latest iteration
        '''
        #step 1: convert angles to complex numbers
        all_betas, all_alphas, all_phis, all_thetas = self.get_numpy_vars(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        # print(all_betas.shape)
        # print(self.betas_rho.shape)
        #Shape of all_betas = N_Multistart x N_blocks
        #step 2: use evolve for each multistart
        exact_fids = []
        for i in range(self.parameters['N_multistart']): 
            # print(i)
            # print(all_betas.shape)
            # print(tf.transpose(all_betas).shape)
            final_state, ex_fid = self.evolve(betas=all_betas[i],
                                      phis =all_phis[i],
                                      thetas =all_thetas[i], 
                                       n_q = n_q, 
                                        n_c = n_c , target = target)
            exact_fids.append(ex_fid)
        return exact_fids
    
    def return_all_fids_pens(self): 
        '''
        Returns all fidelities and multistarts 
        '''
        fids, inter_layer_vecs = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        pens = self.higher_fock_space_penalty_all_layers(inter_layer_vecs)
        return fids.numpy(), pens.numpy()