import time
from simulationcommon import *
import matplotlib.pyplot as pl
import numpy as np
import math
import scipy.linalg as la
import scipy.optimize
import os


if not os.path.exists('cache'):
    os.makedirs('cache')

if not os.path.exists('cache_test'):
    os.makedirs('cache_test')

if not os.path.exists('plots'):
    os.makedirs('plots')

pl.rcParams['text.usetex'] = True
pl.rc('text.latex', preamble="\\usepackage{amsmath}\n \\usepackage{amssymb}")
pl.rcParams['font.size'] = 13
pl.rcParams['axes.labelsize'] = 13
pl.rcParams['xtick.labelsize'] = 13
pl.rcParams['ytick.labelsize'] = 13


def ensure_flat(array_like):
    """Helper to ensure data is a 1D numpy array."""
    if hasattr(array_like, 'A1'):
        return array_like.A1
    return np.array(array_like).flatten()


def get_best_npz_filename(result, kp, kd, ell=None, eta=None, gamma=None, tend=None):
    """
    Smart helper to find existing NPZ files.
    """
    if ell is None and eta is None:
        f_old = f"data_{result}_positions_kp_{kp:.2f}_kd_{kd:.3f}.npz"
        f_new = f"cache_test/data_{result}_pos_kp_{kp:.2f}_kd_{kd:.2f}.npz"
        if os.path.exists(f_new): return f_new, True
        if os.path.exists(f_old): return f_old, True
        return f_new, False

    elif ell is not None:
        f_old = f"data_{result}_varying_ell_{ell}.npz"
        f_new = f"cache_test/data_{result}_ell_{ell}.npz"
        if os.path.exists(f_new): return f_new, True
        if os.path.exists(f_old): return f_old, True
        return f_new, False

    elif eta is not None:
        t_suffix = f"_t{tend}" if tend else ""
        f_old = f"data_{result}_brake_eta_{eta}_gamma_{gamma}{t_suffix}.npz"
        f_new = f"cache_test/data_{result}_brake_eta_{eta}_gamma_{gamma}{t_suffix}.npz"
        if os.path.exists(f_new): return f_new, True
        if os.path.exists(f_old): return f_old, True
        return f_new, False

    return "error.npz", False


def get_get_nuk_thm1(simulation_parameters, matrix_Ac, matrix_Bc):
    T = simulation_parameters["T"]
    Nbar = simulation_parameters["Nbar"]
    alpha = simulation_parameters["alpha"]
    norm_Ac = la.norm(matrix_Ac)

    if norm_Ac < 1e-9:
        return lambda xk, uk: Nbar

    def get_nuk(xk, uk):
        norm_xk = la.norm(xk)
        norm_Bc_uk = la.norm(matrix_Bc @ uk)
        denominator_term = (norm_xk + norm_Bc_uk / norm_Ac)
        if denominator_term < 1e-9: return Nbar
        log_arg = (alpha / (np.sqrt(2) * denominator_term)) + 1
        if log_arg <= 0: return 1
        return math.floor((Nbar / T) * (np.log(log_arg) / norm_Ac))

    return get_nuk


def get_get_nuk_thm7(simulation_parameters, matrix_Ac, matrix_Bc):
    T = simulation_parameters["T"]
    Nbar = simulation_parameters["Nbar"]
    alpha = simulation_parameters["alpha"]
    n = simulation_parameters["n"]
    dim_x = matrix_Ac.shape[0]
    dim_u = matrix_Bc.shape[1]

    tilde_A = np.block([[matrix_Ac, matrix_Bc], [np.zeros((dim_u, dim_x)), np.zeros((dim_u, dim_u))]])
    symm_part = (tilde_A + tilde_A.T) / 2
    mu_tilde_A = np.max(np.real(la.eigvals(symm_part)))

    phi_norms = []
    top_block = np.hstack([matrix_Ac, matrix_Bc])
    for i in range(1, n + 1):
        qi = np.zeros((dim_x, 1))
        p_i_minus_1_idx = 0 if i == 1 else 5 + 6 * (i - 2)
        p_i_idx = 5 + 6 * (i - 1)
        qi[p_i_minus_1_idx, 0] = 1
        qi[p_i_idx, 0] = -1
        phi_norms.append(la.norm(qi.T @ top_block))
    phi_new = np.max(phi_norms) if phi_norms else 0

    def get_nuk(xk, uk):
        if mu_tilde_A <= 1e-9: return Nbar
        tilde_xk = np.vstack([xk, uk])
        denominator = phi_new * la.norm(tilde_xk)
        if denominator < 1e-9: return Nbar
        log_arg = (mu_tilde_A * alpha) / denominator + 1
        if log_arg <= 0: return 1
        return math.floor((Nbar / T) * np.log(log_arg) / mu_tilde_A)

    return get_nuk


def plot_result(result, get_get_nuk):
    kps = [0.15, 0.2, 0.25]
    kds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
           1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]
    n_vehicles = 10

    print(f"--- Start {result} Trajectories (Saving to cache_test...) ---")

    for kp in kps:
        for kd in kds:
            npz_filename, exists = get_best_npz_filename(result, kp, kd)

            if exists:
                try:
                    data = np.load(npz_filename)
                    t = data['t']
                    xks_data = data['xks']
                except Exception as e:
                    print(f"  !!! Error loading {npz_filename}: {e}. Re-running.")
                    exists = False

            if not exists:
                print(f"  -> Running Sim: kp={kp:.2f}, kd={kd:.2f}")
                sim_params = common_simulation_parameters()
                sim_params["k_d"] = kd
                sim_params["k_p"] = kp
                desc = f"cache/k_p_{kp:.2f}_k_d={kd:.2f}_{get_get_nuk.__name__}"

                output = get_value(sim_params, get_get_nuk, desc)

                if isinstance(output, str) or len(output) < 5: continue
                d_min_T, d_min_S, k_final, xks_history, time_history = output
                if isinstance(xks_history, str): continue

                t = time_history[:k_final + 1]
                xks_data = xks_history[:, :k_final + 1]

                np.savez(npz_filename, t=t, xks=xks_data)

            pl.figure(figsize=(6, 3))
            p0 = ensure_flat(xks_data[0, :])
            pl.plot(t, p0, label="$p_0$", linestyle='--')

            for i in range(1, n_vehicles + 1):
                p_idx = 5 + 6 * (i - 1)
                if p_idx < xks_data.shape[0]:
                    pi = ensure_flat(xks_data[p_idx, :])
                    pl.plot(t, pi, label=f"$p_{{{i}}}$")

            pl.xlabel(r"Time $t$ [s]")
            pl.ylabel(r"Position $[\mathrm{m}]$")

            pl.grid(True, linestyle=":")
            pl.legend(loc="upper left", bbox_to_anchor=(1, 1.02))

            img_filename = f"plots/plot_{result}_pos_kp_{kp:.2f}_kd_{kd:.2f}.pdf"
            pl.savefig(img_filename, bbox_inches="tight")
            pl.close()

    print(f"--- {result} Trajectories Done ---")


def plot_varying_ell_combined(result, get_get_nuk):
    kp_fixed = 0.25
    kd_fixed = 0.55
    n_vehicles = 10
    ell_values = [0, 4, 7]

    print(f"--- Start {result} Varying Ell Combined (ell={ell_values}) ---")

    fig, axes = pl.subplots(len(ell_values), 1, figsize=(8, 10), sharex=True)
    if len(ell_values) == 1: axes = [axes]

    for i, ell in enumerate(ell_values):
        ax = axes[i]
        npz_filename, exists = get_best_npz_filename(result, kp_fixed, kd_fixed, ell=ell)

        if exists:
            data = np.load(npz_filename)
            t = data['t']
            xks_data = data['xks']
        else:
            print(f"  -> Running Sim: ell={ell}")
            sim_params = common_simulation_parameters()
            sim_params["k_p"] = kp_fixed
            sim_params["k_d"] = kd_fixed
            sim_params["ell"] = ell
            desc = f"cache/ell_{ell}_{get_get_nuk.__name__}"

            output = get_value(sim_params, get_get_nuk, desc)
            if isinstance(output, str) or len(output) < 5: continue
            d_min_T, d_min_S, k_final, xks_history, time_history = output
            if isinstance(xks_history, str): continue

            t = time_history[:k_final + 1]
            xks_data = xks_history[:, :k_final + 1]
            np.savez(npz_filename, t=t, xks=xks_data)

        p0 = ensure_flat(xks_data[0, :])
        ax.plot(t, p0, label="$p_0$", linestyle='--')

        for v in range(1, n_vehicles + 1):
            p_idx = 5 + 6 * (v - 1)
            if p_idx < xks_data.shape[0]:
                pi = ensure_flat(xks_data[p_idx, :])
                ax.plot(t, pi, label=f"$p_{{{v}}}$")

        # LaTeX upright units
        ax.set_ylabel(r"Position $[\mathrm{m}]$")
        ax.grid(True, linestyle=":")
        if i == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1.02))

    # LaTeX upright units
    axes[-1].set_xlabel(r"Time $t$ [s]")
    pl.tight_layout()

    ell_str = "_".join(map(str, ell_values))
    save_path = f"plots/plot_{result}_ell_combined_ells_{ell_str}.pdf"

    pl.savefig(save_path, bbox_inches="tight")
    pl.close()
    print(f"--- Varying Ell Combined Plot Saved to {save_path} ---")

def solve_t_star(tau_d, t_brake, v0_brake, a0_brake, gamma, eta):
    target_v = gamma / eta

    def func(t_star):
        if t_star < t_brake: return 100.0  # penalty
        term_exp = np.exp((t_brake - t_star) / tau_d)
        val = v0_brake + tau_d * (a0_brake + gamma) * (1.0 - term_exp) - gamma * (t_star - t_brake)
        return val - target_v

    try:
        t_star_sol = scipy.optimize.fsolve(func, t_brake + 1.0)[0]
    except:
        t_star_sol = t_brake
    return t_star_sol


def calculate_exact_states(t_array, gamma, eta):
    """
    Calculates v(t), a(t), and u(t) analytically.
    """
    tau_d = 1.5
    t_brake = 5.0
    v0_init = 30.0
    a0_init = 0.0

    v0_b = v0_init
    a0_b = a0_init

    t_star = solve_t_star(tau_d, t_brake, v0_b, a0_b, gamma, eta)

    a0_t_star = (a0_b + gamma) * np.exp((t_brake - t_star) / tau_d) - gamma
    v0_t_star = gamma / eta

    v_values = []
    a_values = []
    u_values = []

    for t in t_array:
        v_val = 0.0
        a_val = 0.0
        u_val = 0.0

        if t < t_brake:
            u_val = 0.0
            a_val = 0.0
            v_val = v0_init

        elif t < t_star:
            u_val = -gamma

            dt = t - t_brake
            a_val = (a0_b + gamma) * np.exp(-dt / tau_d) - gamma

            term_int = -tau_d * (a0_b + gamma) * (np.exp(-dt / tau_d) - 1.0) - gamma * dt
            v_val = v0_b + term_int

        else:
            dt = t - t_star
            crit = 1.0 / (4.0 * tau_d)

            if abs(eta - crit) < 1e-9:
                lambda_val = -1.0 / (2.0 * tau_d)
                c1 = v0_t_star
                c2 = a0_t_star - lambda_val * c1

                v_val = (c1 + c2 * dt) * np.exp(lambda_val * dt)
                a_val = (c2 + lambda_val * (c1 + c2 * dt)) * np.exp(lambda_val * dt)
                u_val = -eta * v_val

            elif eta < crit:
                discr = np.sqrt(1.0 - 4.0 * eta * tau_d)
                lambda1 = (-1.0 + discr) / (2.0 * tau_d)
                lambda2 = (-1.0 - discr) / (2.0 * tau_d)

                c1 = (a0_t_star - lambda2 * v0_t_star) / (lambda1 - lambda2)
                c2 = (lambda1 * v0_t_star - a0_t_star) / (lambda1 - lambda2)

                v_val = c1 * np.exp(lambda1 * dt) + c2 * np.exp(lambda2 * dt)
                a_val = c1 * lambda1 * np.exp(lambda1 * dt) + c2 * lambda2 * np.exp(lambda2 * dt)
                u_val = -eta * v_val
            else:
                v_val = 0.0
                a_val = 0.0
                u_val = 0.0

        v_values.append(v_val)
        a_values.append(a_val)
        u_values.append(u_val)

    return np.array(v_values), np.array(a_values), np.array(u_values)


def plot_braking_u_curves_combined(result, get_get_nuk):
    etas = [0.1, 0.15]
    gammas = [1.0, 1.2]
    tend = 50.0

    t_eval = np.linspace(0, tend, 1000)

    print(f"--- Start Analytical Braking Curves (etas={etas}, gammas={gammas}) ---")

    fig, axes = pl.subplots(3, 1, figsize=(6, 8), sharex=True)

    linestyles = ['dashdot', 'dotted', 'dashed', (0, (5, 1))]
    style_idx = 0

    for eta in etas:
        for gamma in gammas:
            v_t, a_t, u_t = calculate_exact_states(t_eval, gamma, eta)

            lbl = fr"$\gamma={gamma}, \eta={eta}$"
            ls = linestyles[style_idx]

            axes[0].plot(t_eval, u_t, label=lbl, linestyle=ls)

            axes[1].plot(t_eval, a_t, label=lbl, linestyle=ls)

            axes[2].plot(t_eval, v_t, label=lbl, linestyle=ls)

            style_idx = (style_idx + 1) % 4

    axes[0].set_ylabel(r"$u_0(t)$ $[\mathrm{m/s}^2]$")
    axes[0].grid(True, linestyle=":")
    axes[0].legend(fontsize=10, loc='best')

    axes[1].set_ylabel(r"$a_0(t)$ $[\mathrm{m/s}^2]$")
    axes[1].grid(True, linestyle=":")

    axes[2].set_ylabel(r"$v_0(t)$ $[\mathrm{m/s}]$")
    axes[2].set_xlabel(r"Time $t$ [s]")  # LaTeX upright t
    axes[2].grid(True, linestyle=":")

    pl.tight_layout()

    etas_str = "_".join(map(str, etas))
    gammas_str = "_".join(map(str, gammas))
    save_path = f"plots/plot_{result}_brake_v_a_u_{etas_str}_gammas_{gammas_str}.pdf"

    pl.savefig(save_path, bbox_inches="tight")

    pl.savefig("test.png", bbox_inches="tight")

    pl.close()
    print(f"--- Braking Analytical Plot Saved to {save_path} and test.png ---")


if __name__ == "__main__":
    plot_result("Theorem7", get_get_nuk_thm7)

    plot_varying_ell_combined("Theorem7", get_get_nuk_thm7)

    plot_braking_u_curves_combined("Theorem7", get_get_nuk_thm7)