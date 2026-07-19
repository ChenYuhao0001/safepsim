import numpy as np
import numpy.random as ra
import matplotlib.pyplot as pl
import os
import math
import scipy.linalg as la
from simulationcommon import *


try:
    pl.rcParams['text.usetex'] = True
    pl.rc('text.latex', preamble="\\usepackage{amsmath}\n \\usepackage{amssymb}")
except:
    print("LaTeX not found, using standard fonts.")


def get_get_nuk_thm7(simulation_parameters, matrix_Ac, matrix_Bc):
    T = simulation_parameters["T"]
    Nbar = simulation_parameters["Nbar"]
    alpha = simulation_parameters["alpha"]
    n = simulation_parameters["n"]

    dim_x = matrix_Ac.shape[0]
    dim_u = matrix_Bc.shape[1]

    tilde_A = np.block([
        [matrix_Ac, matrix_Bc],
        [np.zeros((dim_u, dim_x)), np.zeros((dim_u, dim_u))]
    ])

    symm_part = (tilde_A + tilde_A.T) / 2
    eigenvalues = la.eigvals(symm_part)
    mu_tilde_A = np.max(np.real(eigenvalues))

    phi_norms = []
    top_block_of_tilde_A = np.hstack([matrix_Ac, matrix_Bc])
    for i in range(1, n + 1):
        qi = np.zeros((dim_x, 1))
        if i == 1:
            p_i_minus_1_idx = 0
            p_i_idx = 5
        else:
            p_i_minus_1_idx = 5 + 6 * (i - 2)
            p_i_idx = 5 + 6 * (i - 1)
        qi[p_i_minus_1_idx, 0] = 1
        qi[p_i_idx, 0] = -1
        phi_i_row_vector = qi.T @ top_block_of_tilde_A
        phi_norms.append(la.norm(phi_i_row_vector))

    phi_new = np.max(phi_norms) if phi_norms else 0

    def get_nuk(xk, uk):
        if mu_tilde_A <= 1e-9:
            return Nbar
        tilde_xk = np.vstack([xk, uk])
        norm_tilde_xk = la.norm(tilde_xk)
        denominator = phi_new * norm_tilde_xk
        if denominator < 1e-9:
            return Nbar
        log_arg = (mu_tilde_A * alpha) / denominator + 1
        if log_arg <= 0:
            return 1
        value = (Nbar / T) * np.log(log_arg) / mu_tilde_A
        return math.floor(value)

    return get_nuk


def simulate_random_custom(simulation_parameters, get_get_nuk):
    T = simulation_parameters["T"]
    Nbar = simulation_parameters["Nbar"]
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

    break_time = simulation_parameters.get("break_time", 5.0)
    break_step = Nbar * int(break_time / T)
    gamma = simulation_parameters["gamma"]
    eta = simulation_parameters["eta"]

    packet_drop_probability = simulation_parameters.get("prob_drop", 0.0)

    matrix_Ac = generate_matrix_Ac(n, tau_d, h, k_p, k_d)
    matrix_Bc = generate_matrix_Bc(n, tau_d, h, k_p, k_d)
    discrete_ab = get_discrete_ab(n, tau_d, h, k_p, k_d, Nbar, T)
    get_nuk_func = get_get_nuk(simulation_parameters, matrix_Ac, matrix_Bc)

    x0 = construct_vector_x0(n, p0, v0, a0, h)
    xks = np.matrix(np.zeros((x0.shape[0], kend)))
    xks[:, 0] = x0
    uks = np.matrix(np.zeros((1 + n, kend)))

    last_communicated_values = []
    for i in range(1, n):
        last_communicated_values.append([xks[8 + 6 * (i - 1), 0]])

    time = 0
    discrete_step = 0
    t_star, t_star_in_steps, a0_at_t_star = None, None, None

    for k in range(kend - 1):
        if time > tend: break
        xk = xks[:, k]
        uk = uks[:, k]
        proposed_nuk = get_nuk_func(xk, uk)
        nuk_to_comm = Nbar - (discrete_step % Nbar)
        actual_nuk = max(1, min(proposed_nuk, nuk_to_comm))
        A, B = discrete_ab[actual_nuk]

        if discrete_step >= break_step:
            if t_star is None:
                v0_at_tb, a0_at_tb = xks[1, k], xks[2, k]
                t_star = calculate_t_star(v0_at_tb, a0_at_tb, break_time, gamma, eta, tau_d)
                a0_at_t_star = calculate_a0_t_star(a0_at_tb, break_time, t_star, gamma, tau_d)
                t_star_in_steps = Nbar * int(t_star / T)
            if discrete_step < t_star_in_steps:
                uks[0, k] = -gamma
            else:
                v0_at_t_star = gamma / eta
                current_time = (T / Nbar) * discrete_step
                t_star_val = (T / Nbar) * t_star_in_steps
                uks[0, k] = calculate_u0_post_t_star(current_time, t_star_val,
                                                     a0_at_t_star, v0_at_t_star, gamma, eta, tau_d)

        if time > break_time and xks[1, k] < 0:
            xks[1, k], uks[0, k], xks[2, k] = 0, 0, 0

        if discrete_step % Nbar == 0:
            for i in range(1, n):
                is_received = ra.choice([0, 1], p=[packet_drop_probability, 1.0 - packet_drop_probability])
                if is_received == 1:
                    last_communicated_values[i - 1].append(xks[8 + 6 * (i - 1), k])

        uks[1, k] = uks[0, k]
        for i in range(1, n):
            uks[1 + i, k] = last_communicated_values[i - 1][-1]

        xks[:, k + 1] = A * xks[:, k] + B * uks[:, k]
        time += actual_nuk * T / Nbar
        discrete_step += actual_nuk

    d_min_T = calculate_minimum_distance(xks[:, :k + 1], n)
    return d_min_T


def run_and_plot_probability_analysis(result_name, get_get_nuk, sample_size=100):
    probabilities = np.arange(0.6, 0.9, 0.01)
    medians = []
    percentiles_25 = []
    percentiles_75 = []

    folder_name = "Theorem7_probability_analysis"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print(f"--- Starting Probability Analysis for {result_name} ---")
    print(f"--- Output Folder: {folder_name} ---")
    print(f"--- Sample Size: {sample_size} ---")

    for p in probabilities:
        values = np.zeros(sample_size)
        filename = os.path.join(folder_name, f"d_min_data_samplesize_{sample_size}_p_{p:.2f}.npz")

        if os.path.exists(filename):
            print(f"[p={p:.2f}] Loading cached data...")
            data = np.load(filename)
            if 'arr_0' in data:
                values = data['arr_0']
            else:
                print(f"Warning: 'arr_0' not found in {filename}. Re-running simulation.")
                for i in range(sample_size):
                    if i % 10 == 0: print(f"  Simulation {i}/{sample_size}...")
                    sim_params = common_simulation_parameters()
                    sim_params["prob_drop"] = p
                    d_min = simulate_random_custom(sim_params, get_get_nuk)
                    values[i] = d_min
                np.savez(filename, values)
        else:
            print(f"[p={p:.2f}] Running simulations...")
            for i in range(sample_size):
                if i % 10 == 0: print(f"  Simulation {i}/{sample_size}...")
                sim_params = common_simulation_parameters()
                sim_params["prob_drop"] = p
                d_min = simulate_random_custom(sim_params, get_get_nuk)
                values[i] = d_min
            np.savez(filename, values)

        medians.append(np.median(values))
        percentiles_25.append(np.percentile(values, 25))
        percentiles_75.append(np.percentile(values, 75))

    print(f"Generating Summary Plot for {result_name} (Percentile Method)...")

    probabilities = np.array(probabilities)
    medians = np.array(medians)
    p25 = np.array(percentiles_25)
    p75 = np.array(percentiles_75)

    pl.figure(figsize=(6, 3))
    pl.fill_between(probabilities,
                    p25,
                    p75,
                    color='#1f77b4',
                    alpha=0.2,
                    label='IQR (25\%-75\%)')

    pl.plot(probabilities, medians,
            color='#1f77b4',
            linewidth=2,
            label="Median $d'_{\min}$")
    pl.grid(True, linestyle=':', alpha=0.6)

    pl.xlabel(r"Packet Drop Probability $p$")
    pl.ylabel(r"$d'_{\min}\,\,[\mathrm{m}]$")
    pl.legend(loc='best')
    pl.tight_layout()
    summary_filename = os.path.join(folder_name, f"summary_{result_name}_d_min_vs_p_percentile_plot.pdf")
    pl.savefig(summary_filename)
    pl.close()
    print(f"Saved {summary_filename}\n")


if __name__ == "__main__":
    run_and_plot_probability_analysis("Theorem7", get_get_nuk_thm7, sample_size=100)