import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from scipy.optimize import fmin

# note that some pulse functions also in fpga_lib are repeated here so this file can be somewhat standalone.

#EG: I have taken Simplified_ECD_pulse_constructionV2.py from single mode and converted it to two mode here
#V2: adding f state
#V3: supports ge ECD with three level ancilla
#V4: each layer consists of ef rotation, ge rotation and then ge ECD
#V5: Constructs ECD_ge in qutrit ancilla subspace
    # Uses the 3 block structure to echo out f state population
    # More modularized code 
    # Not compatible with qubit ECD

'''
V2 Notes: 
1. Dispersive shifts Chi (first order) and Chi' (second order) are 
   arrays where the i't value is the shift correponding to transmon 
   being in the i'th state.

2. Created separate gf ECD phase space trajectory functions

3. TODO: Change the  pi pulse inside cond disp


'''
'''
V4 Notes (August 11, 2023)
'''


def helloworld():
    print('hello world')
def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop / 2 * sigma, chop / 2 * sigma, chop * sigma)
    P = np.exp(-(ts ** 2) / (2.0 * sigma ** 2))
    ofs = P[0]
    return (P - ofs) / (1 - ofs)


def ring_up_smootherstep(length):
    dt = 1.0 / length
    ts = np.arange(length) * dt
    return 6 * ts ** 5 - 15 * ts ** 4 + 10 * ts ** 3


def ring_up_smoothstep(length):
    dt = 1.0 / length
    ts = np.arange(length) * dt
    return 3 * ts ** 2 - 2 * ts ** 3


def rotate(theta, phi=0, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx=dt)
    amp = 1 / energy
    wave = (1 + 0j) * wave
    return (theta / (2.0)) * amp * np.exp(1j * phi) * wave


def rotate_echoed(theta, phi=0, sigma=8, chop=6, dt=1):
    wave_1 = rotate(theta / 2.0, phi, sigma, chop, dt)
    wave_2 = rotate(np.pi, phi, sigma, chop, dt)
    wave_3 = rotate(-theta / 2.0, phi, sigma, chop, dt)
    return np.concatenate([wave_1, wave_2, wave_3])


# displace cavity by an amount alpha
def disp_gaussian(alpha, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx=dt)
    wave = (1 + 0j) * wave
    return (
        (np.abs(alpha) / energy) * np.exp(1j * (np.pi / 2.0 + np.angle(alpha))) * wave
    )


class FakePulse:
    def __init__(self, unit_amp, sigma, chop, detune=0):
        self.unit_amp = unit_amp
        self.sigma = sigma
        self.chop = chop
        self.detune = detune

    def make_wave(self, pad=False):
        wave = gaussian_wave(sigma=self.sigma, chop=self.chop)
        return np.real(wave), np.imag(wave)


class FakeStorage:
    def __init__(
        self,
        chi_kHz=-30.0,
        chi_prime_Hz=1.0,
        Ks_Hz=-2.0,
        epsilon_m_MHz=400.0,
        T1_us=340.0,
        unit_amp=0.05,
        sigma=15,
        chop=4,
        max_dac=0.6,
    ):
        self.chi_kHz = chi_kHz
        self.chi_prime_Hz = chi_prime_Hz
        self.Ks_Hz = Ks_Hz
        self.epsilon_m_MHz = epsilon_m_MHz
        self.max_dac = max_dac
        self.T1_us = T1_us

        self.displace = FakePulse(unit_amp=unit_amp, sigma=sigma, chop=chop)

        #calculating conversion between DAC and Hamiltonian drive amplitude
        disp = disp_gaussian(alpha=1.0, sigma=sigma, chop=chop, dt=1)
        self.epsilon_m_MHz = 1e3*np.real(np.max(np.abs(disp)))/unit_amp/2/np.pi


class FakeQubit:
    def __init__(self, unit_amp, sigma, chop, detune=0):
        self.pulse = FakePulse(unit_amp=unit_amp, sigma=sigma, chop=chop, detune=detune)
        #calculating conversion between DAC and Hamiltonian drive amplitude
        pi = rotate(np.pi, phi=0, sigma=sigma, chop=chop, dt=1)
        self.Omega_m_MHz = 1e3*np.real(np.max(np.abs(pi)))/unit_amp/2/np.pi


# Solution to linear differential equation
def alpha_from_epsilon_linear(epsilon, delta=0, kappa=0, dt=1, alpha_init=0 + 0j):
    ts = np.arange(0, len(epsilon)) * dt
    integrand = np.exp((1j * delta + kappa / 2.0) * ts) * epsilon
    return np.exp(-1 * (1j * delta + kappa / 2.0) * ts) * (
        alpha_init - 1j * np.cumsum(integrand)
    )


# Later: include chi prime for deviation during the displacements?
def interp(data_array, dt=1):
    ts = np.arange(0, len(data_array)) * dt
    return interp1d(
        ts, data_array, kind="cubic", bounds_error=False
    )  # can test different kinds


def get_flip_idxs(qubit_dac_pulse):
    return find_peaks(qubit_dac_pulse, height=np.max(qubit_dac_pulse) * 0.975)[0]




# solution to nonlinear differential equation
def alpha_from_epsilon_nonlinear_finite_difference(
    epsilon_array, delta=0, Ks=0, kappa=0, alpha_init=0 + 0j
):
    dt = 1
    alpha = np.zeros_like(epsilon_array)
    alpha[0] = alpha_init
    alpha[1] = alpha_init
    for j in range(1, len(epsilon_array) - 1):
        alpha[j + 1] = (
            2
            * dt
            * (
                -1j * delta * alpha[j]
                - 2j * Ks * np.abs(alpha[j]) ** 2 * alpha[j]
                - (kappa / 2.0) * alpha[j]
                - 1j * epsilon_array[j]
            )
            + alpha[j - 1]
        )
    return alpha




def alpha_from_epsilon_ge_finite_difference(
    epsilon_array,
    delta=0,
    chi=[],
    chi_prime=[],
    Ks=0,
    kappa=0,
    alpha_g_init=0 + 0j,
    alpha_e_init=0 + 0j,
):
    dt = 1
    alpha_g = np.zeros_like(epsilon_array)
    alpha_e = np.zeros_like(epsilon_array)
    # todo: can handle initial condition with finite difference better...
    alpha_g[0], alpha_g[1] = alpha_g_init, alpha_g_init
    alpha_e[0], alpha_e[1] = alpha_e_init, alpha_e_init
    for j in range(1, len(epsilon_array) - 1):
        alpha_g[j + 1] = (
            2
            * dt
            * (
                -1j * delta * alpha_g[j]
                + 2j * Ks * np.abs(alpha_g[j]) ** 2 * alpha_g[j]
                - (kappa / 2.0) * alpha_g[j]
                -1j * epsilon_array[j]
                - 1j * (chi[0] + 2 * chi_prime[0] * np.abs(alpha_g[j]) ** 2)* alpha_g[j]
            )
            + alpha_g[j - 1]
        )
        alpha_e[j + 1] = (
            2
            * dt
            * (
                 -1j * delta * alpha_e[j]
                 + 2j * Ks * np.abs(alpha_e[j]) ** 2 * alpha_e[j]
                 - (kappa / 2.0) * alpha_e[j]
                -1j * epsilon_array[j]
               # - 1j * (chi) * alpha_e[j]
                - 1j * (chi[1] + 2 * chi_prime[1] * np.abs(alpha_e[j]) ** 2) * alpha_e[j]
            )
            + alpha_e[j - 1]
        )

        # alpha_f[j + 1] = (
        #     2
        #     * dt
        #     * (
        #          -1j * delta * alpha_f[j]
        #          - 2j * Ks * np.abs(alpha_f[j]) ** 2 * alpha_f[j]
        #          - (kappa / 2.0) * alpha_f[j]
        #         -1j * epsilon_array[j]
        #        # - 1j * (chi) * alpha_e[j]
        #         - 1j * 2 * (chi + 2 * chi_prime * np.abs(alpha_f[j]) ** 2) * alpha_f[j]
        #     )
        #     + alpha_e[j - 1]
        # )
    return alpha_g, alpha_e
def alpha_from_epsilon_ef_finite_difference(
    epsilon_array,
    delta=0,
    chi=[],
    chi_prime=[],
    Ks=0,
    kappa=0,
    alpha_e_init=0 + 0j,
    alpha_f_init=0 + 0j,
):
    dt = 1
    alpha_e = np.zeros_like(epsilon_array)
    alpha_f = np.zeros_like(epsilon_array)
    # todo: can handle initial condition with finite difference better...
    alpha_e[0], alpha_e[1] = alpha_e_init, alpha_e_init
    alpha_f[0], alpha_f[1] = alpha_f_init, alpha_f_init
    for j in range(1, len(epsilon_array) - 1):
        alpha_e[j + 1] = (
            2
            * dt
            * (
                -1j * delta * alpha_e[j]
                + 2j * Ks * np.abs(alpha_e[j]) ** 2 * alpha_e[j]
                - (kappa / 2.0) * alpha_e[j]
                -1j * epsilon_array[j]
                - 1j * (chi[1] + 2 * chi_prime[1] * np.abs(alpha_e[j]) ** 2)* alpha_e[j]
            )
            + alpha_e[j - 1]
        )
        alpha_f[j + 1] = (
            2
            * dt
            * (
                 -1j * delta * alpha_f[j]
                 + 2j * Ks * np.abs(alpha_f[j]) ** 2 * alpha_f[j]
                 - (kappa / 2.0) * alpha_f[j]
                -1j * epsilon_array[j]
               # - 1j * (chi) * alpha_f[j]
                - 1j  * (chi[2] + 2 * chi_prime[2] * np.abs(alpha_f[j]) ** 2) * alpha_f[j]
            )
            + alpha_f[j - 1]
        )

    return alpha_e, alpha_f

def get_ge_trajectories(
    epsilon,
    delta=0,
    chi=[],
    chi_prime=[],
    Ks=0,
    kappa=0,
    flip_idxs=[],
    finite_difference=False,
):
    func = (
        alpha_from_epsilon_ge_finite_difference
        # if finite_difference
        # else alpha_from_epsilon_ge
    )
    f = lambda epsilon, alpha_g_init, alpha_e_init: func(
        epsilon,
        delta=delta,
        chi=chi,
        chi_prime=chi_prime,
        Ks=Ks,
        kappa=kappa,
        alpha_g_init=alpha_g_init,
        alpha_e_init=alpha_e_init,
    )
    epsilons = np.split(epsilon, flip_idxs)
    alpha_g = []  # alpha_g defined as the trajectory that starts in g
    alpha_e = []
    g_state = 0  # this bit will track if alpha_g is in g (0) or e (1)
    alpha_g_init = 0 + 0j
    alpha_e_init = 0 + 0j
    for epsilon in epsilons:
        alpha_g_current, alpha_e_current = f(epsilon, alpha_g_init, alpha_e_init)
        if g_state == 0:
            alpha_g.append(alpha_g_current)
            alpha_e.append(alpha_e_current)
        else:
            alpha_g.append(alpha_e_current)
            alpha_e.append(alpha_g_current)
        # because we will flip the qubit, the initial state for the next g trajectory will be the final from e of the current trajectory.
        alpha_g_init = alpha_e_current[-1]
        alpha_e_init = alpha_g_current[-1]
        g_state = 1 - g_state  # flip the bit
    alpha_g = np.concatenate(alpha_g)
    alpha_e = np.concatenate(alpha_e)
    return alpha_g, alpha_e

def get_ef_trajectories(
    epsilon,
    delta=0,
    chi=[],
    chi_prime=[],
    Ks=0,
    kappa=0,
    flip_idxs=[],
    finite_difference=False,
):
    func = (
        alpha_from_epsilon_ef_finite_difference
        # if finite_difference
        # else alpha_from_epsilon_ge
    )
    f = lambda epsilon, alpha_e_init, alpha_f_init: func(
        epsilon,
        delta=delta,
        chi=chi,
        chi_prime=chi_prime,
        Ks=Ks,
        kappa=kappa,
        alpha_e_init=alpha_e_init,
        alpha_f_init=alpha_f_init,
    )
    epsilons = np.split(epsilon, flip_idxs)
    alpha_e = []  # alpha_e defined as the trajectory that starts in e
    alpha_f = []
    e_state = 0  # this bit will track if alpha_e is in e (0) or f (1)
    alpha_e_init = 0 + 0j
    alpha_f_init = 0 + 0j
    for epsilon in epsilons:
        alpha_e_current, alpha_f_current = f(epsilon, alpha_e_init, alpha_f_init)
        if e_state == 0:
            alpha_e.append(alpha_e_current)
            alpha_f.append(alpha_f_current)
        else:
            alpha_e.append(alpha_e_current)
            alpha_f.append(alpha_f_current)
        # because we will flip the qubit, the initial state for the next g trajectory will be the final from e of the current trajectory.
        alpha_e_init = alpha_f_current[-1]
        alpha_f_init = alpha_e_current[-1]
        e_state = 1 - e_state  # flip the bit
    alpha_e = np.concatenate(alpha_e)
    alpha_f = np.concatenate(alpha_f)
    return alpha_e, alpha_f

# this will use the pre-calibrated pulses.
# note that this will return the DAC pulses, not the values of epsilon and Omega.
# Buffer time can be a negative number if you wish to perform the pi pulse while the cavity is being displaced
# conditional displacement is defined as:
# D(beta/2)|eXg| + D(-beta/2) |gXe|
def conditional_displacement(
    beta,
    alpha,
    storage,
    qubit,
    buffer_time=4,
    wait_time = 0,
    version = 'ge', # or 'ef'
    curvature_correction=True,
    chi_prime_correction=True,
    kerr_correction=True,
    kappa = 0,
    pad=True,
    finite_difference=True,
    output=False,
):
    #print('is_gf: ' + str(is_gf))
    #print('Modified conditional displacement called')
    beta = float(beta) if isinstance(beta, int) else beta
    alpha = float(alpha) if isinstance(alpha, int) else alpha
    chi = 2 * np.pi * 1e-6 * storage.chi_kHz
    chi_prime = 2 * np.pi * 1e-9 * storage.chi_prime_Hz if chi_prime_correction else 0.0
    Ks = 2 * np.pi * 1e-9 * storage.Ks_Hz

    # V1
    # delta is the Hamiltonian parameter, so if it is positive, that means the cavity
    # is detuned positivly relative to the current frame, so the drive is
    # below the cavity.
    # We expect chi to be negative, so we want to drive below the cavity, hence delta should
    # be positive.

    # V2
    # For drive freuency to be (omega_c_e +omega_c_f)/2  , 
    # w_c - w_d = - (chi_e + chi_f)/2

    if version == 'ge':
        delta = -1 * (chi[0] +chi[1])/2
    elif version == 'ef':
        delta = -1 * (chi[1] +chi[2])/2
    epsilon_m = 2 * np.pi * 1e-3 * storage.epsilon_m_MHz
    alpha = np.abs(alpha)
    beta_abs = np.abs(beta)
    beta_phase = np.angle(beta)

    # note, even with pad =False, there is a leading and trailing 0 because
    # every gaussian pulse will start / end at 0. Could maybe remove some of these
    # later to save a few ns.
    dr, di = storage.displace.make_wave(pad=False)
    d = storage.displace.unit_amp * (dr + 1j * di)
    pr, pi = qubit.pulse.make_wave(pad=False)
    # doing the same thing the FPGA does
    detune = qubit.pulse.detune
    if np.abs(detune) > 0:
        ts = np.arange(len(pr)) * 1e-9
        c_wave = (pr + 1j * pi) * np.exp(-2j * np.pi * ts * detune)
        pr, pi = np.real(c_wave), np.imag(c_wave)
    p = qubit.pulse.unit_amp * (pr + 1j * pi)

    # only add buffer time at the final setp
    def construct_CD(alpha, tw, r, r0, r1, r2, buf=0):

        cavity_dac_pulse = r * np.concatenate(
            [
                alpha * d * np.exp(1j * phase),
                np.zeros(tw),
                r0 * alpha * d * np.exp(1j * (phase + np.pi)),
                np.zeros(len(p) + 2 * buf),
                r1 * alpha * d * np.exp(1j * (phase + np.pi)),
                np.zeros(tw),
                r2 * alpha * d * np.exp(1j * phase),
            ]
        )
        qubit_dac_pulse = np.concatenate(
            [
                np.zeros(tw + 2 * len(d) + buf),
                p,
                np.zeros(tw + 2 * len(d) + buf),
            ]
        )
        # need to detune the pulse for chi prime

        # if chi_prime_correction:
        #    ts = np.arange(len(cavity_dac_pulse))
        #    cavity_dac_pulse = cavity_dac_pulse * np.exp(-1j * ts * chi_prime * n)
        return cavity_dac_pulse, qubit_dac_pulse

    def integrated_beta_and_displacement(epsilon):
        '''
        V2 Note: alpha_e represents the displacement conditioned on the excited state 
        whether that may be the e state or the f state, depending on type of ECD the 
        user wishes to enact.
        
        '''
        # note that the trajectories are first solved without kerr.
        flip_idx = int(len(epsilon) / 2)

        if version == 'ef': 

            #ignore that alpha_g and alpha_e should be alpha_e and alpha_f respectively
            # helps out later
            alpha_g, alpha_e = get_ef_trajectories(
            epsilon,
            delta=delta,
            chi=chi,
            kappa = kappa,
            chi_prime=chi_prime,
            Ks=Ks,
            flip_idxs=[flip_idx],
            finite_difference=finite_difference,
        )
        elif version == 'ge': 
            alpha_g, alpha_e = get_ge_trajectories(
                epsilon,
                delta=delta,
                chi=chi,
                kappa = kappa,
                chi_prime=chi_prime,
                Ks=Ks,
                flip_idxs=[flip_idx],
                finite_difference=finite_difference,
            )
        mid_disp = np.abs(alpha_g[flip_idx] + alpha_e[flip_idx])
        final_disp = np.abs(alpha_g[-1] + alpha_e[-1])
        first_radius = np.abs(
            (alpha_g[int(flip_idx / 2)] + alpha_e[int(flip_idx / 2)]) / 2.0
        )
        second_radius = np.abs(
            (alpha_g[int(3 * flip_idx / 2)] + alpha_e[int(3 * flip_idx / 2)]) / 2.0
        )
        if output:
            print(
                "\r  mid_disp: %.4f" % mid_disp
                + " final_disp: %.4f" % final_disp
                + " first_radius: %.4f" % first_radius
                + " second_radius: %.4f" % second_radius,
                # end="",
            )
        return np.abs(alpha_g[-1] - alpha_e[-1]), np.abs(alpha_g[-1] + alpha_e[-1])

    """
    def ratios(alpha, tw):
        n = np.abs(alpha) ** 2
        chi_effective = chi + 2 * chi_prime * n
        r = np.cos((chi_effective / 2.0) * tw)
        r2 = np.cos(chi_effective * tw)
        # r = np.cos((chi/2.0)*(tw + 2*tp))/np.cos((chi/2.0)*tp)
        # r2 = np.cos((chi/2.0)*(tw + 2*tp)) - np.cos((chi/2.0)*(tw + tp))
        return r, r2
    """
    # ratios will perform a minimization problem:
    # given the alpha, and the tw, find the ratio of the middle pulses and the
    # final pulse which:
    # 1. returns the state to the middle after the first half
    # 2.  returns the state to the middle after the full thing.
    # as a bonus:
    # 3. uses an equal radius in the second half as the first half.
    def ratios(alpha, tw):
        # guess ratios:
        n = np.abs(alpha) ** 2
        if version == 'ef': # gf pulse
            chi_effective = (chi[2] - chi[1]) + 2 * (chi_prime[2] - chi_prime[1]) * n
        elif version == 'ge': # ge pulse
            chi_effective = (chi[1] - chi[0]) + 2 * (chi_prime[1] - chi_prime[0]) * n
        r = 1.0
        r0 = np.cos((chi_effective / 2.0) * tw)
        r1 = r0
        r2 = np.cos(chi_effective * tw)

        # the cost function
        def cost(x):
            r = x[0]
            r0 = x[1]
            r1 = x[2]
            r2 = x[3]
            cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r0, r1, r2)
            epsilon = cavity_dac_pulse * epsilon_m
            flip_idx = int(len(epsilon) / 2)
            if version == 'ef':
                alpha_g, alpha_e = get_ef_trajectories(
                    epsilon,
                    delta=delta,
                    chi=chi,
                    kappa = kappa,
                    chi_prime=chi_prime,
                    Ks=Ks,
                    flip_idxs=[flip_idx],
                    finite_difference=finite_difference,
                )
            elif version == 'ge': 
                alpha_g, alpha_e = get_ge_trajectories(
                    epsilon,
                    delta=delta,
                    chi=chi,
                    kappa = kappa,
                    chi_prime=chi_prime,
                    Ks=Ks,
                    flip_idxs=[flip_idx],
                    finite_difference=finite_difference,
                )
            mid_disp = np.abs(alpha_g[flip_idx] + alpha_e[flip_idx])
            final_disp = np.abs(alpha_g[-1] + alpha_e[-1])
            first_radius = np.abs(
                (alpha_g[int(flip_idx / 2)] + alpha_e[int(flip_idx / 2)]) / 2.0
            )
            second_radius = np.abs(
                (alpha_g[int(3 * flip_idx / 2)] + alpha_e[int(3 * flip_idx / 2)]) / 2.0
            )
            if output:
                print(
                    "\r  mid_disp: %.4f" % mid_disp
                    + " final_disp: %.4f" % final_disp
                    + " first_radius: %.4f" % first_radius
                    + " second_radius: %.4f" % second_radius,
                    # end="",
                )
            return (
                np.abs(mid_disp)
                + np.abs(final_disp)
                + np.abs(first_radius - np.abs(alpha))
                + np.abs(second_radius - np.abs(alpha))
            )

        result = fmin(cost, x0=[r, r0, r1, r2], ftol=1e-3, xtol=1e-3, disp=False)
        r = result[0]
        r0 = result[1]
        r1 = result[2]
        r2 = result[3]
        return r, r0, r1, r2

    # the initial guesses
    # phase of the displacements.
    phase = beta_phase + np.pi / 2.0
    n = np.abs(alpha) ** 2
    if version == 'ef': # gf pulse
        chi_effective = (chi[2] - chi[1]) + 2 * (chi_prime[2] - chi_prime[1]) * n
    elif version == 'ge': # ge pulse
        chi_effective = (chi[1] - chi[0]) + 2 * (chi_prime[1] - chi_prime[0]) * n
    # initial tw
    tw = int(np.abs(np.arcsin(beta_abs / (2 * alpha)) / chi_effective))

    # ratios of the displacements
    r, r0, r1, r2 = ratios(alpha, tw)

    cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r0, r1, r2)

    if curvature_correction:

        epsilon = cavity_dac_pulse * epsilon_m
        current_beta, current_disp = integrated_beta_and_displacement(epsilon)
        diff = np.abs(current_beta) - np.abs(beta)
        ratio = np.abs(current_beta) / np.abs(beta)
        if output:
            print(
                "tw: "
                + str(tw)
                + "alpha: "
                + str(alpha)
                + "beta: "
                + str(current_beta)
                + "diff: "
                + str(diff)
            )
        #  could look at real/imag part...
        # for now, will only consider absolute value
        # first step: lower tw
        #
        if diff < 0:
            tw = int(tw * 1.5)
            ratio = 1.01
        tw_flag = True
        while np.abs(diff) / np.abs(beta) > 1e-3:
            if ratio > 1.0 and tw > 0 and tw_flag:
                tw = int(tw / ratio)
            else:
                tw_flag = False
                # if ratio > 1.02:
                #    ratio = 1.02
                # if ratio < 0.98:
                #    ratio = 0.98
                alpha = alpha / ratio

            # update the ratios for the new tw and alpha given chi_prime
            r, r0, r1, r2 = ratios(alpha, tw)
            cavity_dac_pulse, qubit_dac_pulse = construct_CD(
                alpha, tw, r, r0, r1, r2, buf=buffer_time
            )
            epsilon = cavity_dac_pulse * epsilon_m
            current_beta, current_disp = integrated_beta_and_displacement(epsilon)
            diff = np.abs(current_beta) - np.abs(beta)
            ratio = np.abs(current_beta) / np.abs(beta)
            if output:
                print(
                    "tw: "
                    + str(tw)
                    + "alpha: "
                    + str(alpha)
                    + "beta: "
                    + str(current_beta)
                    + "diff: "
                    + str(diff)
                )
    """
    # now, correct for the the displacement
    while current_disp > 0.05:
        #without correction, it overshoots the origin of phase space.
        r2 = 0.99*r2
        cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r2, buf=buffer_time)
        epsilon = cavity_dac_pulse * epsilon_m
        current_beta, current_disp = integrated_beta_and_displacement(epsilon)
        if output:
                print("tw: " + str(tw))
                print("alpha: " + str(alpha))
                print("disp: " + str(current_disp))
                print("beta: " + str(current_beta))
                print("diff: " + str(diff))
    """
    # need to add back in the buffer time to the pulse
    cavity_dac_pulse, qubit_dac_pulse = construct_CD(
        alpha, tw, r, r0, r1, r2, buf=buffer_time
    )
    epsilon = cavity_dac_pulse * epsilon_m
    current_beta, current_disp = integrated_beta_and_displacement(epsilon)

    # the final step is kerr correction. Now, the trajectories are solved with kerr, and there is a frame update.
    # This is not yet implemented/tested fully because Kerr correction is not important with Alec's parameters.
    # Can include it when using larger Kerr.
    # Don't trust the below code, it needs to be looked at in more detail. In particular, the rate of local rotation
    # And the rate of center of mass rotation differs by a factor of 2.
    """
    if kerr_correction:
        #here, we want to get the trajectory without kerr!
        alpha_g, alpha_e = get_ge_trajectories(epsilon, chi=chi, chi_prime=chi_prime, kerr=0.0, flip_half_way=True)
        nbar_g = np.abs(alpha_g)**2
        nbar_e = np.abs(alpha_e)**2
        det_g = kerr*nbar_g
        det_e = kerr*nbar_e
        avg_det = (det_g + det_e)/2.0 #note, that the dets should be the same
        accumulated_phase = np.cumsum(avg_det)
        cavity_dac_pulse = cavity_dac_pulse*np.exp(-1j*accumulated_phase)
    else:
        accumulated_phase = np.zeros_like(epsilon)
    """
    if pad:
        while len(cavity_dac_pulse) % 4 != 0:
            cavity_dac_pulse = np.pad(cavity_dac_pulse, (0, 1), mode="constant")
            qubit_dac_pulse = np.pad(qubit_dac_pulse, (0, 1), mode="constant")
            # accumulated_phase = np.pad(accumulated_phase, (0,1), mode='edge')

    print('---------------------------')
    print('Final Displacement: ' + str(current_beta))
    #cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r2, buf=buffer_time)
    cavity_dac_pulse = np.append(cavity_dac_pulse, np.zeros(wait_time)) 
    qubit_dac_pulse = np.append(qubit_dac_pulse, np.zeros(wait_time)) 
    # between ECD pulses
    return  cavity_dac_pulse, qubit_dac_pulse,alpha, tw


def double_circuit(betas, phis, thetas, final_disp=True):
    phis = [phis] if type(phis) is not list else phis
    thetas = [thetas] if type(thetas) is not list else thetas
    betas2 = []
    phis2 = [[] for _ in phis]
    thetas2 = [[] for _ in thetas]
    for i, beta in enumerate(betas):
        if np.abs(beta) > 0 and not (i == len(betas) - 1 and final_disp):
            betas2.extend([beta / 2.0, beta / 2.0])
            for j in range(len(thetas)):
                phis2[j].extend([phis[j][i], 0])
                thetas2[j].extend([thetas[j][i], np.pi])
        else:
            betas2.extend([beta])
            for j in range(len(thetas)):
                phis2[j].extend([phis[j][i]])
                thetas2[j].extend([thetas[j][i]])
    return betas2, phis2, thetas2

def Qubit_rotation_post_process(
    phi,
    theta,
    pulse_dict,
    N_modes,
    qubit,
    buffer_time=0,
):
    '''
    1. Constructs the R_ef and R_ge gates
    2. Updates the dac lists storing the array
    '''
    
    # constructing qubit part rotation (ef rotaton)
    pr, pi = qubit.pulse.make_wave(pad=False)
    
    o_r_ef = (
        qubit.pulse.unit_amp
        * (theta[1] / np.pi)
        * (pr + 1j * pi)
        * np.exp(1j * phi[1])
    )
    pulse_dict['qubit_dac_pulse'][1].append(np.exp(-1j * pulse_dict['cumulative_qubit_phase'][1]) * o_r_ef)
    
    pulse_dict['qubit_dac_pulse'][0].append(np.zeros(len(o_r_ef)))
    # constructing qubit part rotation (ge rotaton)
        
    o_r_ge = (
        qubit.pulse.unit_amp
        * (theta[0] / np.pi)
        * (pr + 1j * pi)
        * np.exp(1j * phi[0])
    )
    pulse_dict['qubit_dac_pulse'][0].append(np.exp(-1j * pulse_dict['cumulative_qubit_phase'][0]) * o_r_ge)
    
    pulse_dict['qubit_dac_pulse'][1].append(np.zeros(len(o_r_ge)))

    for m_ in range(N_modes): 
        pulse_dict['cavity_dac_pulse'][m_].append(np.zeros(len(o_r_ge) + len(o_r_ef)))  # qubit rotation happening, all modes on standby

    #adding buffer time
    for m_ in range(N_modes): 
         pulse_dict['cavity_dac_pulse'][m_].append(np.zeros(buffer_time))  # all modes on standby

    pulse_dict['qubit_dac_pulse'][0].append(np.zeros(buffer_time))
    pulse_dict['qubit_dac_pulse'][1].append(np.zeros(buffer_time))
    
    return pulse_dict

def ECD_ef_post_process(
    beta,
    mode,
    pulse_dict,
    storage,
    N_modes,
    qubit,
    alpha_CD,
    buffer_time=0,
    wait_time = 0,
    curvature_correction=True,
    chi_prime_correction=True,
    kerr_correction=False,
    kappa = [0,0],
    finite_difference=True,
    output=False,
):
    '''
    1. Constructs the ECD_ef gate
    2. Updates the dac lists storing the array
    3. Updates the phases of ge and ef rotation
    '''
    #Constructing the ECD gate in 2 level subspace      
    e_cd, o_cd, alpha, tw = conditional_displacement(
        beta,
        alpha=alpha_CD,
        storage=storage,
        version = 'ef',
        qubit=qubit,
        buffer_time=buffer_time,
        wait_time = wait_time, 
        curvature_correction=curvature_correction,
        chi_prime_correction=chi_prime_correction,
        kerr_correction=kerr_correction,
        kappa = kappa[mode],
        finite_difference=finite_difference,
        output=output,
    )

    pulse_dict['alphas'].append(alpha)
    pulse_dict['tws'].append(tw)

    ## getting the phase for the phase correction
    analytic_dict = analytic_CD_ef(
        -1j * 2 * np.pi * 1e-3 * storage.epsilon_m_MHz * e_cd,
        o_cd,
        2 * np.pi * 1e-6 * storage.chi_kHz
    )

            # print(np.shape(cd_qubit_phases))
            # print(analytic_dict["qubit_phase"])
    pulse_dict['cd_qubit_phases'][0].append(analytic_dict["qubit_phase"][0])
    pulse_dict['cd_qubit_phases'][1].append(analytic_dict["qubit_phase"][1])   
    pulse_dict['analytic_betas'].append(analytic_dict["beta"])        

    # Construct ECD
    m = mode # mode on which ECD acts
    pulse_dict['cavity_dac_pulse'][m].append(np.zeros(buffer_time))
    pulse_dict['cavity_dac_pulse'][m].append(pulse_dict['beta_sign'][m] * e_cd) # ecd happening on m'th mode
    pulse_dict['cavity_dac_pulse'][m].append(np.zeros(buffer_time))

    for m_ in range(N_modes):
        if m_ is not m:  # all other modes besides m'th on standby 
            pulse_dict['cavity_dac_pulse'][m_].append(np.zeros(len(e_cd) + (2 * buffer_time)))  # qubit rotation happening, all modes on standby
    
    #ef qubit drive        
    pulse_dict['qubit_dac_pulse'][1].append(np.zeros(buffer_time))
    pulse_dict['qubit_dac_pulse'][1].append(np.exp(-1j * pulse_dict['cumulative_qubit_phase'][1]) * o_cd)
    pulse_dict['qubit_dac_pulse'][1].append(np.zeros(buffer_time))

    #ge qubit drive
    pulse_dict['qubit_dac_pulse'][0].append(np.zeros(len(o_cd) + (2 * buffer_time)))# gf part is 0 for this duration

    return pulse_dict

# The following function is used for 3 level ancilla 
def conditional_displacement_circuit(
    betas,
    phis,
    thetas,
    storages,
    qubit,
    alpha_CD,
    final_disp=True,
    buffer_time=0,
    wait_time = 0,
    curvature_correction=True,
    qubit_phase_correction=True,
    chi_prime_correction=True,
    kerr_correction=False,
    kappa = [0,0],
    pad=True,
    finite_difference=True,
    output=False,
):
    '''
    MASTER function
    Converts ECD params to pulse sequencies
    '''

    N_modes = len(storages)
    N_layers = len(betas[0])

    pulse_dict = {
            'alphas' : [], 
            'tws' : [],
            'cd_qubit_phases' : [[], []],# ge and ef phase,
            'cumulative_qubit_phase' : [0, 0],  # ge and ef phase
            'analytic_betas' : [],
            'beta_sign': [1 for _ in range(N_modes)],
            'cavity_dac_pulse':[[] for _ in range(N_modes)], # pulse for [[mode 1], [mode 2], ...]
            'qubit_dac_pulse': [[], []] # first is ge and the other is ef

    }


    for l in range(N_layers): #construct each layer first 
        for m__ in range(N_modes): # currently assuming N_modes =1 
            m = N_modes - m__ -1
            

            #Getting appropiate parameters
            beta = betas[m][l]
            phi = phis[m][l]
            theta = thetas[m][l]
            storage = storages[m]

            
            # First the rotations
            pulse_dict = Qubit_rotation_post_process(
                            phi,
                            theta,
                            pulse_dict,
                            N_modes,
                            qubit,
                            buffer_time,
                        )

            # Then ECD_ef(-beta/2)
            pulse_dict = ECD_ef_post_process(
                            beta = -1 * beta/2,
                            mode =m ,
                            pulse_dict = pulse_dict,
                            storage = storage,
                            N_modes = N_modes,
                            qubit = qubit,
                            alpha_CD = alpha_CD,
                            buffer_time= buffer_time,
                            wait_time =  wait_time,
                            curvature_correction= curvature_correction,
                            chi_prime_correction= chi_prime_correction,
                            kerr_correction=kerr_correction,
                            kappa = kappa,
                            finite_difference=finite_difference,
                            output= output,
                        )
            #Then pi_ef pulse 
            pulse_dict = Qubit_rotation_post_process(
                            phi = [0,0],
                            theta= [np.pi, 0],
                            pulse_dict = pulse_dict,
                            N_modes = N_modes,
                            qubit = qubit,
                            buffer_time = buffer_time,
                        )
            #Then ECD_ef(beta/2)
            pulse_dict = ECD_ef_post_process(
                            beta =  beta/2,
                            mode =m ,
                            pulse_dict = pulse_dict,
                            storage = storage,
                            N_modes = N_modes,
                            qubit = qubit,
                            alpha_CD = alpha_CD,
                            buffer_time= buffer_time,
                            wait_time =  wait_time,
                            curvature_correction= curvature_correction,
                            chi_prime_correction= chi_prime_correction,
                            kerr_correction=kerr_correction,
                            kappa = kappa,
                            finite_difference=finite_difference,
                            output= output,
                        )
            #Then pi_ef pulse 
            pulse_dict = Qubit_rotation_post_process(
                            phi = [0,0],
                            theta= [np.pi, 0 ],
                            pulse_dict = pulse_dict,
                            N_modes = N_modes,
                            qubit = qubit,
                            buffer_time = buffer_time,
                        )
            #Then ECD_ef(beta/2)
            pulse_dict = ECD_ef_post_process(
                            beta =  -beta/2,
                            mode =m ,
                            pulse_dict = pulse_dict,
                            storage = storage,
                            N_modes = N_modes,
                            qubit = qubit,
                            alpha_CD = alpha_CD,
                            buffer_time= buffer_time,
                            wait_time =  wait_time,
                            curvature_correction= curvature_correction,
                            chi_prime_correction= chi_prime_correction,
                            kerr_correction=kerr_correction,
                            kappa = kappa,
                            finite_difference=finite_difference,
                            output= output,
                        )
            
            # updating phases
            # summing phases across all 3 ecds
            pulse_dict['cumulative_qubit_phase'][0] += sum(pulse_dict['cd_qubit_phases'][0][-3:])
            pulse_dict['cumulative_qubit_phase'][1] += sum(pulse_dict['cd_qubit_phases'][1][-3:])

    pulse_dict['cavity_dac_pulse'] = np.array([np.concatenate(pulse_dict['cavity_dac_pulse'][m_]) for m_ in range(N_modes)])
    print('len of qubit dac pulse is ' + str(len(pulse_dict['qubit_dac_pulse'])))
    pulse_dict['qubit_dac_pulse'][0] = np.concatenate(pulse_dict['qubit_dac_pulse'][0])
    pulse_dict['qubit_dac_pulse'][1] = np.concatenate(pulse_dict['qubit_dac_pulse'][1])
    pulse_dict['qubit_dac_pulse'] = np.array(pulse_dict['qubit_dac_pulse'])

    # flip idxs for ef since ef is flipping (ef ecds)
    flip_idxs = find_peaks(np.abs(pulse_dict['qubit_dac_pulse'][1]), height=np.max(np.abs(pulse_dict['qubit_dac_pulse'][1])) * 0.975)[0]
       # for qp in qubit_dac_pulse
    

    if kerr_correction:
        print("Kerr correction not implemented yet!")
    pulse_dict['accumulated_phase'] = np.zeros_like(pulse_dict['cavity_dac_pulse'])


    return pulse_dict



def analytic_CD_ef(epsilon, Omega, chi):
    '''
    Solves EOM as in Alec's S4A
    Computes phase accrued by e and f state (relative to e and g state) during a ECD_ge gate
    '''

    flip_idxs = get_flip_idxs(Omega)
    pm = +1
    z = []
    for i in range(len(flip_idxs) + 1):
        l_idx = 0 if i == 0 else flip_idxs[i - 1]
        r_idx = len(Omega) if i == len(flip_idxs) else flip_idxs[i]
        z.append(pm * np.ones(r_idx - l_idx))
        pm = -1 * pm
    z = np.concatenate(z)
    '''
    The chi below comes from dispersive hamiltonian when transmon is treated 
    as a two level system. See supplementary section 3/4 of ECD paper. If working
    with gf ECD, this chi will have a different value. Recall that 
    chi = chi_e - chi_g
    '''
    # if is_gf:
    #     chi_ = (chi[2] - chi[0])
    # else:
    chi_ef = (chi[2] - chi[1])/2
    chi_g_prime = (chi[0] - (( chi[1] + chi[2])/2))
    
    # New variables to encode chis
    ones = np.array([1 for i in range(len(z))])
    a = -1 * chi_ef * z 
    b = -1 * ( chi_g_prime * ones) 

    #EOMs for phis
    phi_ef_dot =  a
    phi_g_dot =  b

    # Solution of EOM for phis
    phi_ef = np.cumsum(phi_ef_dot)
    phi_g = np.cumsum(phi_g_dot)

    #EOMs for gamma, delta_e, delta_f
    gamma = np.zeros_like(phi_ef, dtype=np.complex64)
    delta_ef = np.zeros_like(gamma)
    delta_g = np.zeros_like(gamma)

    for i in range(len(phi_ef)):
        
        delta_ef[i] = 1j * np.sum((1 - np.exp(1j * (phi_ef[:i+1] - phi_ef[i]))) * epsilon[: i + 1])
        
        delta_g[i] = 1j * np.sum((1 - np.exp(1j * (phi_g[:i+1] - phi_g[i]))) * epsilon[: i + 1])
        
        gamma[i] = -1j * np.sum( epsilon[: i + 1])

    theta_ef = -2 * np.cumsum(np.real(np.conj(epsilon) * delta_ef))
    theta_g = -2 * np.cumsum(np.real(np.conj(epsilon) * delta_g))

    correction_ef = 2 * np.imag(gamma[-1] * delta_ef[-1])
    correction_g = 2 * np.imag(gamma[-1] * delta_g[-1])

    theta_prime_ef = theta_ef[-1] + correction_ef
    theta_prime_g = theta_g[-1] + correction_g

    beta = 2 * (delta_ef[-1])# - delta_f[-1])
    print('-----------')
    print('Analytic Values')
    print('delta_g : ' + str(delta_g[-1]))
    print('delta_e : ' + str(delta_ef[-1]))
    print('theta_prime_g : ' + str(theta_prime_g))
    print('theta_prime_ef : ' + str(theta_prime_ef))
    print('gamma : ' + str(gamma[-1]))
    print('phi_g : ' + str(phi_g[-1] * 180 /np.pi) + str(' deg'))
    print('phi_ef : ' + str(phi_ef[-1] * 180 /np.pi) + str(' deg'))
    #print('predicted_beta : ' + str(np.abs(-delta_g[-1] + 2*delta_ef[-1])))
    print('-----------')

    # print('--------------------------------------')
    # print({
    #     "z": z,
    #     "phi": [phi_ge, phi_f],
    #     "delta": [delta_ge, delta_f],
    #     "gamma": gamma,
    #     "theta": [theta_ge, theta_f],
    #     "correction": [correction_ge, correction_f],
    #     "theta_prime": [theta_prime_ge, theta_prime_f],
    #     "qubit_phase": [theta_prime_ge + (theta_prime_f/3),  (2*theta_prime_f/3)],
    #     "beta": beta,
    # })

    return {
        "z": z,
        "phi": [phi_ef, phi_g],
        "delta": [delta_ef, delta_g],
        "gamma": gamma,
        "theta": [theta_ef, theta_g],
        "correction": [correction_ef, correction_g],
        "theta_prime": [theta_prime_ef, theta_prime_g],
        "qubit_phase":[theta_prime_ef,  theta_prime_g],
        #"qubit_phase": [theta_prime_ef, (theta_prime_g - theta_prime_ef)/2],
        #"qubit_phase":[theta_prime_ef + (theta_prime_g/3),  (2*theta_prime_g/3)],
        "beta": beta,
    }



# def analytic_CD_ge(epsilon, Omega, chi, is_gf):

    '''
    Computes phase accrued by e and f state (relative to e and g state) during a ECD_ge gate
    '''

    flip_idxs = get_flip_idxs(Omega)
    pm = +1
    z = []
    for i in range(len(flip_idxs) + 1):
        l_idx = 0 if i == 0 else flip_idxs[i - 1]
        r_idx = len(Omega) if i == len(flip_idxs) else flip_idxs[i]
        z.append(pm * np.ones(r_idx - l_idx))
        pm = -1 * pm
    z = np.concatenate(z)
    '''
    The chi below comes from dispersive hamiltonian when transmon is treated 
    as a two level system. See supplementary section 3/4 of ECD paper. If working
    with gf ECD, this chi will have a different value. Recall that 
    chi = chi_e - chi_g
    '''
    # if is_gf:
    #     chi_ = (chi[2] - chi[0])
    # else:
    chi_ge = (chi[1] - chi[0])/2
    chi_f_prime = chi[2] - (( chi[0] + chi[1])/2)
    
    # New variables to encode chis
    ones = np.array([1 for i in range(len(z))])
    a = -1 * chi_ge * z 
    b = (-1/2) * ( ( chi_ge * z) + ( chi_f_prime * ones) )
    c = (+1/2) * ( ( chi_ge * z) + ( chi_f_prime * ones) )

    #EOMs for phis
    phi_e_dot = -1 * a
    phi_f_dot = -1 * b
    
    ## auxiliary variable for later
    phi_dot = [ np.sqrt( (phi_e_dot[i])**2 +
                        (phi_f_dot[i])**2
                        )  for i in range(len(phi_e_dot))
                ]

    # Solution of EOM for phis
    phi_e = np.cumsum(phi_e_dot)
    phi_f = np.cumsum(phi_f_dot)

    ## auxiliary variable for later
    phi = [ np.sqrt( (phi_e[i])**2 +
                        (phi_f[i])**2
                        )  for i in range(len(phi_e))
                ]

    #EOMs for gamma, delta_e, delta_f
    gamma = np.zeros_like(phi_e, dtype=np.complex64)
    delta_e = np.zeros_like(gamma)
    delta_f = np.zeros_like(gamma)

    for i in range(len(phi_e)):
        
        delta_e[i] = -1 * (phi_e_dot[i]/phi_dot[i]) * np.sum(np.sin(phi[: i + 1] - phi[i]) * epsilon[: i + 1])
        
        delta_f[i] = -1 * (phi_f_dot[i]/phi_dot[i]) * np.sum(np.sin(phi[: i + 1] - phi[i]) * epsilon[: i + 1])
        
        gamma[i] = -1j * np.sum(np.cos(phi[: i + 1] - phi[i]) * epsilon[: i + 1])

    theta_e = -2 * np.cumsum(np.real(np.conj(epsilon) * delta_e))
    theta_f = -2 * np.cumsum(np.real(np.conj(epsilon) * delta_f))

    correction = 2 * np.imag(gamma[-1] * delta[-1])
    theta_prime = theta[-1] + correction
    beta = 2 * delta[-1]
    return {
        "z": z,
        "phi": phi,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "correction": correction,
        "theta_prime": theta_prime,
        "qubit_phase": theta_prime,
        "beta": beta,
    }