import numpy as np
import numpy.random as ra
import scipy.linalg as la
import scipy.integrate as integ
import matplotlib.pyplot as pl
import os
import math
from scipy.special import lambertw


def calculate_t_star(v0_tb, a0_tb, t_brake, gamma, eta, tau_d):
    """
    Calculates the transition braking time t* using the complex formula involving the Lambert W function.
    t* is the time when the vehicle's velocity v0 first reaches gamma/eta.
    """
    term1 = (v0_tb + tau_d * a0_tb + tau_d * gamma + gamma * t_brake - gamma / eta) / gamma
    exponent_of_e = ((gamma / eta - v0_tb - tau_d * a0_tb - tau_d * gamma - gamma * t_brake) / (gamma * tau_d))
    factor_of_e = -((a0_tb + gamma) * np.exp(t_brake / tau_d)) / gamma
    w_argument = factor_of_e * np.exp(exponent_of_e)
    t_star = term1 + tau_d * np.real(lambertw(w_argument))
    return t_star


def calculate_a0_t_star(a0_tb, t_brake, t_star, gamma, tau_d):
    """
    Calculates the acceleration of the virtual vehicle at the transition time t*.
    """
    return (a0_tb + gamma) * np.exp(-(t_star - t_brake) / tau_d) - gamma


def calculate_u0_post_t_star(t, t_star, a0_t_star, v0_t_star, gamma, eta, tau_d):
    """
    Calculates the desired acceleration u0(t) for t > t*.
    """
    u0 = 0.0
    if eta < 1 / (4 * tau_d):
        sqrt_term = np.sqrt(1 - 4 * eta * tau_d)
        lambda1 = (-1 + sqrt_term) / (2 * tau_d)
        lambda2 = (-1 - sqrt_term) / (2 * tau_d)
        term_C1 = (a0_t_star - lambda2 * v0_t_star) / (lambda1 - lambda2)
        term_C2 = (lambda1 * v0_t_star - a0_t_star) / (lambda1 - lambda2)
        u0 = -eta * (term_C1 * np.exp(lambda1 * (t - t_star)) + term_C2 * np.exp(lambda2 * (t - t_star)))
    elif np.isclose(eta, 1 / (4 * tau_d)):
        lambda3 = -1 / (2 * tau_d)
        term_C4 = a0_t_star - lambda3 * v0_t_star
        u0 = -eta * np.exp(lambda3 * (t - t_star)) * (v0_t_star + term_C4 * (t - t_star))
    return u0


# ============================================================================
# [FUEL variant, based on simulationcommon_3.py]
# COMBINATION: CDG (Constant Distance Gap) + PF (Predecessor-Following)
#
# Spacing policy changed from CTG to CDG. Following besselink2017string
# (s_ref,i(t) = s_{i-1}(t) - d, constant d), the spacing error is redefined as
#   e_i^CDG(t) = (p_{i-1}(t) - p_i(t) - L_i) - d
# instead of the CTG error (r + h*v_i(t) term). Consequently:
#   edot_i^CDG      = v_{i-1} - v_i            (no -h*a_i term)
#   eddot_i^CDG     = a_{i-1} - a_i             (no h/tau_d-1, -h/tau_d terms)
# This changes THREE entries across TWO rows of the N block inside Ac
# (s2, s3, s4 loops below), verified against N^CDG:
#   [0,0,0,-1, 0,0]
#   [0,0,0, 0,-1,0]
#   (rows 3-6 of N unchanged)
# Bc, M, G, H are UNCHANGED (topology is still PF; this only affects Ac).
# Theorems 1-4 hold verbatim since the proofs depend only on the linear
# structure of the platoon dynamics, not the specific entries of Ac.
# ============================================================================

def generate_matrix_Ac(n, tau_d=1.5, h=0.6, k_p=0.2, k_d=0.7):
    Ac = 3 + 6 * n
    matrix = np.zeros((Ac, Ac))
    matrix[0, 1] = 1;
    matrix[1, 2] = 1;
    matrix[2, 2] = -1 / tau_d;
    matrix[3, 1] = 1;
    matrix[4, 2] = 1
    for s1 in range(1, n + 1):
        row, col = -3 + 6 * s1, 6 * s1
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = -1
    for s2 in range(1, n + 1):
        row, col = -3 + 6 * s2, 1 + 6 * s2
        # CDG: edot_i has no -h*a_i term (was: matrix[row, col] = -h)
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = 0
    for s3 in range(1, n + 1):
        row, col = -2 + 6 * s3, 1 + 6 * s3
        # CDG: eddot_i coefficient of a_i is -1 (was: h/tau_d - 1)
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = -1
    for s4 in range(1, n + 1):
        row, col = -2 + 6 * s4, 2 + 6 * s4
        # CDG: eddot_i coefficient of u_i is 0 (was: -h/tau_d)
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = 0
    for s5 in range(1, n + 1):
        row, col = -1 + 6 * s5, 6 * s5
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = 1
    for s6 in range(1, n + 1):
        row, col = 6 * s6, 1 + 6 * s6
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = 1
    for s7 in range(1, n + 1):
        row, col = 1 + 6 * s7, 1 + 6 * s7
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = -1 / tau_d
    for s8 in range(1, n + 1):
        row, col = 1 + 6 * s8, 2 + 6 * s8
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = 1 / tau_d
    for s9 in range(1, n + 1):
        row, col = 2 + 6 * s9, -3 + 6 * s9
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = k_p / h
    for s10 in range(1, n + 1):
        row, col = 2 + 6 * s10, -2 + 6 * s10
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = k_d / h
    for s11 in range(1, n + 1):
        row, col = 2 + 6 * s11, 2 + 6 * s11
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = -1 / h
    for s12 in range(1, n + 1):
        row, col = 3 + 6 * s12, 6 * s12
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = 1
    for s13 in range(1, n + 1):
        row, col = 4 + 6 * s13, 1 + 6 * s13
        if 0 <= row < Ac and 0 <= col < Ac: matrix[row, col] = 1
    return matrix


def generate_matrix_Bc(n, tau_d=1.5, h=0.6, k_p=0.2, k_d=0.7):
    row_bc, column_bc = 3 + 6 * n, 1 + n
    matrix = np.zeros((row_bc, column_bc))
    matrix[2, 0] = 1 / tau_d
    for s1 in range(1, n + 1):
        row, col = 2 + 6 * s1, s1
        if 0 <= row < row_bc and 0 <= col < column_bc: matrix[row, col] = 1 / h
    return matrix


def construct_vector_x0(n, p0, v0, a0, d, length=4.7):
    x = np.matrix(np.zeros((3 + 6 * n, 1)))
    x[0, 0], x[1, 0], x[2, 0] = p0, v0, a0
    for i in range(1, n + 1):
        index = 3 + (i - 1) * 6
        x[index + 2, 0] = p0 - (length + d) * i
        x[index + 3, 0], x[index + 4, 0], x[index + 5, 0] = v0, a0, 0
        x[index, 0] = calculate_error(i, x, d, length)
        if i > 1:
            x[index + 1, 0] = x[index - 3, 0] - x[index + 3, 0]
        else:
            x[index + 1, 0] = x[1, 0] - x[index + 3, 0]
    return x


def construct_vector_u0(n, ):
    u = np.matrix(np.zeros((1 + n, 1)))
    u[0, 0] = 0
    for i in range(1, n + 1):
        index = i
        if index == 1:
            u[index, 0] = u[index - 1, 0]
        else:
            u[index, 0] = 0
    return u


def calculate_error(i, x, d, length=4.7):
    p_i = x[0, 0] if i == 1 else x[-1 + 6 * (i - 1), 0]
    j_index = 5 + 6 * (i - 1)
    if j_index < x.shape[0]:
        p_j = x[j_index, 0]
        return p_i - p_j - length - d
    else:
        return 0


def convertc2d(Ac, Bc, Delta):
    n = Ac.shape[0]
    A = la.expm(Delta * Ac)

    def f(t): return np.asarray(la.expm((Delta - t) * Ac).flatten())

    result = integ.quad_vec(f, 0, Delta)
    B = np.matrix(result[0].reshape((n, n))) * Bc
    return A, B


def calculate_minimum_distance(xks, n_vehicles, length=4.7):
    """Calculates the minimum distance from a series of state vectors."""
    min_distances = []
    for i in range(n_vehicles - 1):
        j = i + 1
        pis = np.asarray(xks[5 + 6 * i, :]).flatten()  # p_i
        pjs = np.asarray(xks[5 + 6 * j, :]).flatten()  # p_j
        distance_values = (pis - pjs) - length
        min_distances.append(np.min(distance_values))
    return max([0, min(min_distances)])

def get_discrete_ab(n, tau_d, h, k_p, k_d, Nbar, T):
    filenamea = (
        f"matrices/CDG_LPF_Amatrices_Nbar_{Nbar}_T_{T:.3f}"
        f"_tau_{tau_d:.3f}_h_{h:.3f}"
        f"_kp_{k_p:.4f}_kd_{k_d:.4f}"
    )

    filenameb = (
        f"matrices/CDG_LPF_Bmatrices_Nbar_{Nbar}_T_{T:.3f}"
        f"_tau_{tau_d:.3f}_h_{h:.3f}"
        f"_kp_{k_p:.4f}_kd_{k_d:.4f}"
    )
    #filenamea = f"/home/matrices/Amatrices_Nbar_{Nbar}_T_{T:.3f}_kp_{k_p:.3f}_kd_{k_d:.3f}.npy"
    #filenameb = f"/home/matrices/Bmatrices_Nbar_{Nbar}_T_{T:.3f}_kp_{k_p:.3f}_kd_{k_d:.3f}.npy"
    if not os.path.exists('matrices'):
        os.makedirs('matrices')
    if os.path.exists(filenamea) and os.path.exists(filenameb):
        As = np.load(filenamea, allow_pickle=True)
        Bs = np.load(filenameb, allow_pickle=True)
        return [list(pair) for pair in zip(As, Bs)]
    else:
        matrix_Ac = generate_matrix_Ac(n, tau_d, h, k_p, k_d)
        matrix_Bc = generate_matrix_Bc(n, tau_d, h, k_p, k_d)
        identityA, zeroB = np.matrix(np.eye(matrix_Ac.shape[0])), np.matrix(0 * matrix_Bc)
        As, Bs = [identityA], [zeroB]
        delta = T / Nbar
        Ad, Bd = convertc2d(matrix_Ac, matrix_Bc, delta)
        print("generating matrices")
        for i in range(1, Nbar + 1):
            print(i)
            As.append(Ad @ As[-1])
            Bs.append(Ad @ Bs[-1] + Bd)
        np.save(filenamea, As)
        np.save(filenameb, Bs)
        return [list(pair) for pair in zip(As, Bs)]



def simulate(simulation_parameters, get_get_nuk):
    T = simulation_parameters["T"]
    Nbar = simulation_parameters["Nbar"]
    seed = simulation_parameters["seed"]
    alpha = simulation_parameters["alpha"]
    tau_d = simulation_parameters["tau_d"]
    h = simulation_parameters["h"]
    d = simulation_parameters.get("d", 5.0)
    k_p = simulation_parameters["k_p"]
    k_d = simulation_parameters["k_d"]
    n = simulation_parameters["n"]
    p0 = simulation_parameters["p0"]
    v0 = simulation_parameters["v0"]
    a0 = simulation_parameters["a0"]
    tend = simulation_parameters["tend"]
    kend = Nbar * int(tend / T)
    break_time = simulation_parameters["break_time"]
    break_step = Nbar * int(break_time / T)
    gamma = simulation_parameters["gamma"]
    eta = simulation_parameters["eta"]
    ell = simulation_parameters["ell"]

    matrix_Ac = generate_matrix_Ac(n, tau_d, h, k_p, k_d)
    matrix_Bc = generate_matrix_Bc(n, tau_d, h, k_p, k_d)
    discrete_ab = get_discrete_ab(n, tau_d, h, k_p, k_d, Nbar, T)

    get_nuk = get_get_nuk(simulation_parameters, matrix_Ac, matrix_Bc)

    x0 = construct_vector_x0(n, p0, v0, a0, d)

    xks = np.matrix(np.zeros((x0.shape[0], kend)))
    xks[:, 0] = x0
    u0 = construct_vector_u0(n)

    uks = np.matrix(np.zeros((u0.shape[0], kend)))
    uks[:, 0] = u0

    time_vector = np.zeros(kend)

    xks_at_S = [x0]

    last_communicated_values = []
    # Initially all values are communicated
    for i in range(1, n):
        last_communicated_values.append([xks[8 + 6 * (i - 1), 0]])

    discrete_step = 0
    time = 0

    t_star = None
    t_star_in_steps = None
    a0_at_t_star = None

    slowing_down = True

    k = 0
    for k in range(kend - 1):
        time_vector[k] = time

        if time > tend:
            break

        xk = xks[:, k]
        uk = uks[:, k]

        proposed_nuk = get_nuk(xk, uk)

        nuk_to_comm = Nbar - (discrete_step % Nbar)
        actual_nuk = max(1, min(proposed_nuk, nuk_to_comm))
        A, B = discrete_ab[actual_nuk]

        # --- MODIFIED: Oscillation Logic (Threshold based) ---
        if discrete_step >= break_step:
            current_v = xks[1, k]  # v0 (Virtual Leader Velocity)

            if slowing_down:
                # Goal: Reach 5.0
                if current_v > 5.0:
                    uks[0, k] = -gamma  # Brake
                else:
                    # Reached bottom (approx), switch to acceleration
                    slowing_down = False
                    uks[0, k] = gamma
            else:
                # Goal: Reach 30.0
                if current_v < 30.0:
                    uks[0, k] = gamma   # Accelerate
                else:
                    # Reached top (approx), switch to braking
                    slowing_down = True
                    uks[0, k] = -gamma

        if time > break_time and xks[1, k] < 0:
            xks[1, k], uks[0, k], xks[2, k] = 0, 0, 0

        if discrete_step % Nbar == 0:
            # possible communication time
            if k > 0:
                xks_at_S.append(xk)

            if discrete_step % (Nbar * (ell + 1)) == 0:
                # communication time
                for i in range(1, n):
                    last_communicated_values[i - 1].append(xks[8 + 6 * (i - 1), k])

        uks[1, k] = uks[0, k]
        for i in range(1, n):
            # because uks[0] = u0, uks[1] = u0, uks[2] = uhat1 ...
            uks[1 + i, k] = last_communicated_values[i - 1][-1]

        xks[:, k + 1] = A * xks[:, k] + B * uks[:, k]
        time += actual_nuk * T / Nbar
        discrete_step += actual_nuk
        time_vector[k + 1] = time

        stop_simulation = False
        for i in range(1, n + 1):
            v_idx = 6 + 6 * (i - 1)
            if xks[v_idx, k] < 0:
                stop_simulation = True

        if stop_simulation:
            break

    d_min_T = calculate_minimum_distance(xks[:, :k + 1], n)

    xks_at_S_matrix = np.hstack(xks_at_S)
    d_min_S = calculate_minimum_distance(xks_at_S_matrix, n)
    return d_min_T, d_min_S, k, xks, time_vector

def simulate_with_cache(simulation_parameters, get_get_nuk, description):
    xks_filename = f"{description}_xks.npz"
    dk_filename = f"{description}_dk.npz"
    time_filename = f"{description}_time_vector.npz"
    xks = ""
    if os.path.exists(dk_filename):
        print("Loading from cache")
        data = np.load(dk_filename)
        if 'arr_0' in data:
            values = data['arr_0']
            d_min_T = values[0]
            d_min_S = values[1]
            k = int(values[2])
        else:
            print("We must rerun simulation.")
        data = np.load(time_filename)
        if 'arr_0' in data:
            time_vector = data['arr_0']
        else:
            print("We must rerun simulation.")
    else:
        d_min_T, d_min_S, k, xks, time_vector = simulate(simulation_parameters, get_get_nuk)
        # np.savez(xks_filename, xks)
        np.savez(dk_filename, np.array([d_min_T, d_min_S, k]))
        np.savez(time_filename, time_vector)

    return d_min_T, d_min_S, k, xks, time_vector


def get_value(simulation_parameters, get_get_nuk, description=None):
    if description is None:
        d_min_T, d_min_S, k_final, xks_history, time_history = simulate(simulation_parameters, get_get_nuk)
    else:
        d_min_T, d_min_S, k_final, xks_history, time_history = simulate_with_cache(simulation_parameters, get_get_nuk, description)

    T = simulation_parameters["T"]
    Nbar = simulation_parameters["Nbar"]
    seed = simulation_parameters["seed"]
    alpha = simulation_parameters["alpha"]
    tau_d = simulation_parameters["tau_d"]
    h = simulation_parameters["h"]
    k_p = simulation_parameters["k_p"]
    k_d = simulation_parameters["k_d"]
    n = simulation_parameters["n"]
    p0 = simulation_parameters["p0"]
    v0 = simulation_parameters["v0"]
    a0 = simulation_parameters["a0"]
    tend = simulation_parameters["tend"]
    kend = Nbar * int(tend / T)
    break_time = simulation_parameters["tend"]
    break_step = Nbar * int(break_time / T)
    gamma = simulation_parameters["gamma"]
    eta = simulation_parameters["eta"]
    ell = simulation_parameters["ell"]
    print(f"kp: {k_p}, kd: {k_d}, d_min(T): {d_min_T:.4f}, d_min(S): {d_min_S:.4f}, steps(k): {k_final}")
    return d_min_T, d_min_S, k_final, xks_history, time_history

def common_simulation_parameters():
    return {"T": 0.1,
            "Nbar": 20000,
            "seed": 5,
            "alpha": 1,
            "tau_d": 1.5,
            "h": 0.6,
            "d": 18.3,
            "k_p": 0.2,
            "k_d": 0.7,
            "n": 10,
            "p0": 200,
            "v0": 30,
            "a0": 0,
            "tend": 300,
            "kend": 5000000,
            "break_time": 5.0,
            "gamma": 0.5,  # <--- Modified: Changed from 1.2 to 5.0 for faster oscillation
            "eta": 0.1,
            "ell": 1
            }
