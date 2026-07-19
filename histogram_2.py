from simulationcommon import *
import matplotlib.pyplot as pl
import numpy.random as ra

pl.rcParams['text.usetex'] = True
pl.rc('text.latex', preamble="\\usepackage{amsmath}\n \\usepackage{amssymb}")
pl.rcParams['font.size'] = 13
pl.rcParams['axes.labelsize'] = 13
pl.rcParams['xtick.labelsize'] = 13
pl.rcParams['ytick.labelsize'] = 13


def get_get_nuk_thm7(simulation_parameters, matrix_Ac, matrix_Bc):
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
            print("Abort! Error!")
            return 1

        value = (Nbar / T) * np.log(log_arg) / mu_tilde_A
        return math.floor(value)

    return get_nuk


def simulate_random(simulation_parameters, get_get_nuk):
    ra.seed = 5

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
    break_time = simulation_parameters["break_time"]
    break_step = Nbar * int(break_time / T)
    gamma = simulation_parameters["gamma"]
    eta = simulation_parameters["eta"]
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

    packet_drop_probability = 0.8

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
                uks[0, k] = calculate_u0_post_t_star((T / Nbar) * discrete_step, (T / Nbar) * t_star_in_steps,
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


def plot_histogram(result_name, get_get_nuk, kp_hist, kd_hist, sample_size=10000): # Modified default value to 10000
    print(f"--- Generating Histogram for {result_name} (kp={kp_hist}, kd={kd_hist}) ---")
    values = np.zeros(sample_size)

    import os
    filename = f"histogramvalues_samplesize_{sample_size}_kp_{kp_hist}_kd_{kd_hist}.npz"

    if os.path.exists(filename):
        data = np.load(filename)
        if 'arr_0' in data:
            values = data['arr_0']
        else:
            print("Warning: 'arr_0' not found in .npz file. Re-running simulation.")
            os.remove(filename)
            for i in range(sample_size):
                if i % 10 == 0: print(f"  Simulation {i}/{sample_size}...")
                simulation_parameters = common_simulation_parameters()
                simulation_parameters["k_p"] = kp_hist
                simulation_parameters["k_d"] = kd_hist

                d_min = simulate_random(simulation_parameters, get_get_nuk)
                values[i] = d_min
            np.savez(filename, values)
    else:
        for i in range(sample_size):
            if i % 10 == 0: print(f"  Simulation {i}/{sample_size}...")
            simulation_parameters = common_simulation_parameters()
            simulation_parameters["k_p"] = kp_hist
            simulation_parameters["k_d"] = kd_hist

            d_min = simulate_random(simulation_parameters, get_get_nuk)
            values[i] = d_min
        np.savez(filename, values)

    pl.figure(figsize=(6, 3))
    pl.hist(values, density=True, edgecolor="#5050AA", facecolor="#CCCCFF")
    pl.xlabel(r"$d_{\min}^{\prime}$")
    pl.tight_layout()

    filename_pdf = f"histogram_{result_name}_kp_{kp_hist}_kd_{kd_hist}.pdf"
    pl.savefig(filename_pdf, bbox_inches = "tight")
    pl.close()
    print(f"Saved {filename_pdf}\n")


if __name__ == "__main__":
    plot_histogram("Theorem7", get_get_nuk_thm7, 0.2, 1.2, 10000)
    plot_histogram("Theorem7", get_get_nuk_thm7, 0.25, 1.2, 10000)