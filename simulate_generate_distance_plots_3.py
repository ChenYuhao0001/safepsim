import time
from simulationcommon import *
import matplotlib.pyplot as pl
import numpy as np
import math
import scipy.linalg as la

pl.rcParams['text.usetex'] = True
pl.rc('text.latex', preamble="\\usepackage{amsmath}\n \\usepackage{amssymb}")
pl.rcParams['font.size'] = 13
pl.rcParams['axes.labelsize'] = 13
pl.rcParams['xtick.labelsize'] = 13
pl.rcParams['ytick.labelsize'] = 13

def get_get_nuk_thm1(simulation_parameters, matrix_Ac, matrix_Bc):
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

    norm_Ac = la.norm(matrix_Ac)
    dim_x = matrix_Ac.shape[0]
    dim_u = matrix_Bc.shape[1]
    tilde_A = np.block([
        [matrix_Ac, matrix_Bc],
        [np.zeros((dim_u, dim_x)), np.zeros((dim_u, dim_u))]
    ])

    symm_part = (tilde_A + tilde_A.T) / 2
    eigenvalues = la.eigvals(symm_part)
    mu_tilde_A = np.max(np.real(eigenvalues))

    #for i, val in enumerate(eigenvalues):
    #    print(f"  Eigenvalue {i + 1}: {val.real:.4f} + {val.imag:.4f}j")

    phi_norms = []
    top_block_of_tilde_A = np.hstack([matrix_Ac, matrix_Bc])
    if norm_Ac < 1e-9:
        def get_nuk_const_max(xk, uk):
            return Nbar

        return get_nuk_const_max

    def get_nuk(xk, uk):
        norm_xk = la.norm(xk)  # ||x(tk)||
        norm_Bc_uk = la.norm(matrix_Bc @ uk)  # ||Bc u(tk)||

        denominator_term = (norm_xk + norm_Bc_uk / norm_Ac)

        if denominator_term < 1e-9:
            return Nbar

        log_arg = (alpha / (np.sqrt(2) * denominator_term)) + 1

        if log_arg <= 0:
            print(" log_arg <= 0. return 1.")
            return 1

        t_delta = np.log(log_arg) / norm_Ac

        value = (Nbar / T) * t_delta

        return math.floor(value)

    return get_nuk


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


def plot_result(result, get_get_nuk):
    kps = [0.15, 0.2, 0.25]
    kds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
           1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]
    n_vehicles = 10

    print(f"--- Start generating {result} simulation plots (Trajectories) ---")

    for kp in kps:
        for kd in kds:
            print(f"Running: $k_p$={kp:.2f}, $k_d$={kd:.2f}")
            simulation_parameters = common_simulation_parameters()
            simulation_parameters["k_d"] = kd
            simulation_parameters["k_p"] = kp

            description = f"cache/k_p_{kp:.2f}_k_d={kd:.2f}_{get_get_nuk.__name__}"
            d_min_T, d_min_S, k_final, xks_history, time_history = get_value(simulation_parameters, get_get_nuk, description)
            t = time_history[:k_final + 1]
            xks_data = xks_history[:, :k_final + 1]

            pl.figure(figsize=(12, 8))

            p0 = xks_data[0, :].A1
            pl.plot(t, p0, label="$p_0$ (Virtual)", linestyle='--')

            for i in range(1, n_vehicles + 1):
                p_idx = 5 + 6 * (i - 1)
                if p_idx < xks_data.shape[0]:
                    pi = xks_data[p_idx, :].A1
                    pl.plot(t, pi, label=f"$p_{i}$")

            pl.title(f"{result}: Vehicle Positions ($k_p$={kp:.2f}, $k_d$={kd:.2f})")
            pl.xlabel("Time [s]")
            pl.ylabel("Position [m]")
            pl.grid(True, linestyle=":")
            pl.legend(loc="upper left", bbox_to_anchor=(1, 1.02))
            pl.tight_layout()

            filename = f"plot_{result}_positions_kp_{kp:.2f}_kd_{kd:.3f}.png"
            pl.savefig(filename, bbox_inches = "tight")
            pl.close()
    print(f"--- Plot generation complete ---")


def detect_backward_motion(xks_history, n_vehicles):
    for i in range(1, n_vehicles + 1):
        v_idx = 6 + 6 * (i - 1)
        if v_idx < xks_history.shape[0]:
            velocities = xks_history[v_idx, :].A1
            if np.min(velocities) < -0.01:
                return True
    return False

def plot_summary_with_check_test(result, get_get_nuk):
    kps = [0.15]
    kds = [0.50, 0.55]
    n_vehicles = 10

    print(f"--- Start generating {result} summary plots (Split into 3 PDFs) ---")

    fig1, ax1 = pl.subplots(figsize=(6, 3))
    fig2, ax2 = pl.subplots(figsize=(6, 3))
    fig3, ax3 = pl.subplots(figsize=(6, 3))

    for kp in kps:
        valid_kds = []
        valid_d_min_T = []
        valid_d_min_S = []
        valid_steps = []

        for kd in kds:
            print(f"Checking {result}: $k_p$={kp:.2f}, $k_d$={kd:.2f} ...")

            simulation_parameters = common_simulation_parameters()
            simulation_parameters["k_d"] = kd
            simulation_parameters["k_p"] = kp

            description = f"cache/k_p_{kp:.2f}_k_d={kd:.2f}_{get_get_nuk.__name__}"
            d_min_T, d_min_S, k_final, xks_history, time_history = get_value(simulation_parameters, get_get_nuk, description)

            valid_kds.append(kd)
            valid_d_min_T.append(d_min_T)
            valid_d_min_S.append(d_min_S)

            actual_end_time = time_history[k_final]
            if actual_end_time <= 1e-9:
                valid_steps.append(0)
            else:
                valid_steps.append(k_final)

        line_label = f'$k_{{\\mathrm{{p}}}}={kp:.2f}$'

        ax1.plot(valid_kds, valid_d_min_T, marker="o", markersize=6, label=line_label)
        ax2.plot(valid_kds, valid_d_min_S, marker="s", markersize=6, label=line_label)
        ax3.plot(valid_kds, valid_steps, marker="s", markersize=6, linestyle='--', label=line_label)

    ax1.set_ylabel(r'$d_{\min} [m]$ ')
    ax1.set_xlabel(r'$k_{\mathrm{d}}$')
    ax1.grid(True, linestyle=":")
    ax1.legend(loc="best")
    fig1.tight_layout()
    filename1 = f"plot_summary_{result}_dminT.pdf"
    fig1.savefig(filename1, bbox_inches = "tight")
    pl.close(fig1)
    print(f"--- Summary plot for d_min (T) saved to {filename1} ---")


    ax3.set_ylabel(r'Total Simulation Steps ($k^{\prime}_{\mathrm{end}}$)')
    ax3.set_xlabel(r'$k_{\mathrm{d}}$')
    ax3.grid(True, linestyle=":")
    ax3.legend(loc="best")
    fig3.tight_layout()
    filename3 = f"plot_summary_{result}_steps.pdf"
    fig3.savefig(filename3, bbox_inches = "tight")
    pl.close(fig3)
    print(f"--- All summary plots have been saved separately ---")


def plot_summary_with_check(result, get_get_nuk):
    kps = [0.15, 0.2, 0.25]
    kds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
           1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]
    n_vehicles = 10

    print(f"--- Start generating {result} summary plots (Split into 3 PDFs) ---")

    fig1, ax1 = pl.subplots(figsize=(6, 3))
    fig2, ax2 = pl.subplots(figsize=(6, 3))
    fig3, ax3 = pl.subplots(figsize=(6, 3))

    for kp in kps:
        valid_kds = []
        valid_d_min_T = []
        valid_d_min_S = []
        valid_steps = []

        for kd in kds:
            print(f"Checking {result}: $k_p$={kp:.2f}, $k_d$={kd:.2f} ...")

            simulation_parameters = common_simulation_parameters()
            simulation_parameters["k_d"] = kd
            simulation_parameters["k_p"] = kp

            description = f"cache/k_p_{kp:.2f}_k_d={kd:.2f}_{get_get_nuk.__name__}"
            d_min_T, d_min_S, k_final, xks_history, time_history = get_value(simulation_parameters, get_get_nuk, description)

            valid_kds.append(kd)
            valid_d_min_T.append(d_min_T)
            valid_d_min_S.append(d_min_S)

            actual_end_time = time_history[k_final]
            if actual_end_time <= 1e-9:
                valid_steps.append(0)
            else:
                valid_steps.append(k_final)

        line_label = f'$k_{{\\mathrm{{p}}}}={kp:.2f}$'

        ax1.plot(valid_kds, valid_d_min_T, marker="o", markersize=6, label=line_label)
        ax2.plot(valid_kds, valid_d_min_S, marker="s", markersize=6, label=line_label)
        ax3.plot(valid_kds, valid_steps, marker="s", markersize=6, linestyle='--', label=line_label)

    ax1.set_ylabel(r'$d^{\prime}_{\min}\,\,$ [$\mathrm{m}$]')
    ax1.set_xlabel(r'$k_{\mathrm{d}}$')
    ax1.grid(True, linestyle=":")
    ax1.legend(loc=2)
    fig1.tight_layout()
    filename1 = f"plot_summary_{result}_dminT.pdf"
    fig1.savefig(filename1, bbox_inches = "tight")
    pl.close(fig1)
    print(f"--- Summary plot for d_min (T) saved to {filename1} ---")

    ax3.set_ylabel(r'Total Simulation Steps ($k^{\prime}_{\mathrm{end}}$)')
    ax3.set_xlabel(r'$k_{\mathrm{d}}$')
    ax3.grid(True, linestyle=":")
    ax3.legend(loc="best")
    fig3.tight_layout()
    filename3 = f"plot_summary_{result}_steps.pdf"
    fig3.savefig(filename3, bbox_inches = "tight")
    pl.close(fig3)
    print(f"--- All summary plots have been saved separately ---")


def plot_dmin_difference(get_get_nuk1, get_get_nuk2):
    """
    Generates a figure showing the difference in minimum distance (d_min)
    between two different theorems (Theorem 1 - Theorem 7) across a range of kd values.
    """
    kps = [0.15, 0.2, 0.25]
    kds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
           1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]

    fig, ax = pl.subplots(figsize=(6, 3))

    for kp in kps:
        kds_data = []
        dmin_diffs = []

        for kd in kds:
            # Run Thm 1
            simulation_parameters1 = common_simulation_parameters()
            simulation_parameters1["k_d"] = kd
            simulation_parameters1["k_p"] = kp
            description = f"cache/k_p_{kp:.2f}_k_d={kd:.2f}_{get_get_nuk1.__name__}"
            d_min_T1, _, _, _, _ = get_value(simulation_parameters1, get_get_nuk1, description)

            # Run Thm 7
            simulation_parameters2 = common_simulation_parameters()
            simulation_parameters2["k_d"] = kd
            simulation_parameters2["k_p"] = kp
            description = f"cache/k_p_{kp:.2f}_k_d={kd:.2f}_{get_get_nuk2.__name__}"
            d_min_T2, _, _, _, _ = get_value(simulation_parameters2, get_get_nuk2, description)

            kds_data.append(kd)
            dmin_diffs.append(d_min_T1 - d_min_T2)

        line_label = f'$k_{{\\mathrm{{p}}}}={kp:.2f}$'
        ax.plot(kds_data, dmin_diffs, marker="o", markersize=6, label=line_label)
        print(f"difference", line_label)
        print(np.max(np.abs(dmin_diffs)))

    ax.set_ylabel(r'$d_{\min}^{\mathrm{Thm1}} - d_{\min}^{\mathrm{Thm2}}$ [$\mathrm{m}$] ')
    ax.set_xlabel(r'$k_{\mathrm{d}}$')
    ax.grid(True, linestyle=":")
    ax.legend(loc="best")
    fig.tight_layout()
    filename = "plot_summary_dmin_diff_T1_T7.pdf"
    fig.savefig(filename, bbox_inches = "tight")
    pl.close(fig)
    print(f"--- Difference plot saved to {filename} ---")


if __name__ == "1__main__":
    t1 = time.time()
    d_min_T, d_min_S, k_final, xks_history, time_history = get_value(common_simulation_parameters(), get_get_nuk_thm7, "TestCache")
    print(d_min_T)
    t2 = time.time()
    d_min_T, d_min_S, k_final, xks_history, time_history = get_value(common_simulation_parameters(), get_get_nuk_thm7, "TestCache")
    print(d_min_T)
    t3 = time.time()
    print(f"Initial run time: {t2-t1}")
    print(f"Second run time: {t3-t2}")


if __name__ == "1__main__":
    plot_summary_with_check_test("Theorem1", get_get_nuk_thm1)
    plot_summary_with_check_test("Theorem7", get_get_nuk_thm7)
if __name__ == "__main__":
    plot_summary_with_check("Theorem1", get_get_nuk_thm1)
    plot_summary_with_check("Theorem7", get_get_nuk_thm7)

    print("\n--- Running New Plot Functions ---")

    plot_dmin_difference(get_get_nuk_thm1, get_get_nuk_thm7)
