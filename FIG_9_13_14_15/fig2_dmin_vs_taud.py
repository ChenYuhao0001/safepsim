import numpy as np
import numpy.random as ra
import matplotlib.pyplot as pl
import os
import shutil

from simulationcommon import (
    common_simulation_parameters,
    generate_matrix_Ac,
    generate_matrix_Bc,
    get_discrete_ab,
    construct_vector_x0,
    calculate_t_star,
    calculate_a0_t_star,
    calculate_u0_post_t_star,
    calculate_minimum_distance,
)
from fig_common import get_get_nuk_thm7, setup_latex

setup_latex()

TAUD_VALUES = np.linspace(0.5, 3.0, 10)     # 10 points in [0.5, 3.0]
P_FIXED = 0.8
N_REPEAT = 10
SEEDS = list(range(N_REPEAT))

OUT_DIR = "figures_out"
CACHE_DIR = "cache_taud_IQR"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


_MATRIX_BACKUP = "matrices__backup_before_taud_IQR"


def simulate_random_custom_local(simulation_parameters, get_get_nuk):
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

    time = 0.0
    discrete_step = 0
    t_star, t_star_in_steps, a0_at_t_star = None, None, None

    k = 0
    for k in range(kend - 1):
        if time > tend:
            break
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
                uks[0, k] = calculate_u0_post_t_star(
                    current_time, t_star_val, a0_at_t_star, v0_at_t_star,
                    gamma, eta, tau_d)

        if time > break_time and xks[1, k] < 0:
            xks[1, k], uks[0, k], xks[2, k] = 0, 0, 0

        if discrete_step % Nbar == 0:
            for i in range(1, n):
                is_received = ra.choice(
                    [0, 1],
                    p=[packet_drop_probability, 1.0 - packet_drop_probability])
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


def run_point(tau_d):
    cache = os.path.join(CACHE_DIR, f"taud_{tau_d:.4f}_p_{P_FIXED:.2f}.npz")
    if os.path.exists(cache):
        return np.load(cache)["dmin"]

    # clear ONLY the working matrices dir so this tau_d regenerates its own
    # matrices (the matrix cache filename in simulationcommon does not include
    # tau_d). The user's original matrices/ is backed up/restored in main().
    if os.path.exists("matrices"):
        shutil.rmtree("matrices")

    vals = np.zeros(N_REPEAT)
    for r, seed in enumerate(SEEDS):
        ra.seed(seed)
        sp = common_simulation_parameters()
        sp["prob_drop"] = P_FIXED
        sp["tau_d"] = float(tau_d)
        vals[r] = simulate_random_custom_local(sp, get_get_nuk_thm7)
        print(f"  [tau_d={tau_d:.3f}] rep {r+1}/{N_REPEAT}: d_min={vals[r]:.3f}")
    np.savez(cache, dmin=vals)
    return vals


def main():
    # --- back up any pre-existing matrices/ so we never destroy the user's cache ---
    backed_up = False
    if os.path.exists("matrices"):
        if os.path.exists(_MATRIX_BACKUP):
            shutil.rmtree(_MATRIX_BACKUP)
        shutil.move("matrices", _MATRIX_BACKUP)
        backed_up = True

    try:
        medians = np.zeros(len(TAUD_VALUES))
        q25_values = np.zeros(len(TAUD_VALUES))
        q75_values = np.zeros(len(TAUD_VALUES))

        for j, td in enumerate(TAUD_VALUES):
            vals = run_point(td)
            medians[j] = np.median(vals)
            q25_values[j] = np.percentile(vals, 25)
            q75_values[j] = np.percentile(vals, 75)

        pl.figure(figsize=(6, 3))
        pl.fill_between(
            TAUD_VALUES,
            q25_values,
            q75_values,
            color='#1f77b4',
            alpha=0.2,
            label=r"IQR (25\%-75\%)")

        pl.plot(
            TAUD_VALUES,
            medians,
            color='#1f77b4',
            linewidth=2,
            label=r"Median $d'_{\min}$")

        pl.xlabel(r"$\tau_{\mathrm{d}}\ [\mathrm{s}]$")
        pl.ylabel(r"$d'_{\min}\ [\mathrm{m}]$")
        pl.grid(True, linestyle=":", alpha=0.6)
        pl.legend(loc="best")
        pl.tight_layout()
        out = os.path.join(OUT_DIR, "dmin_vs_taud_iqr.pdf")
        pl.savefig(out)
        pl.close()
        print(f"\nSaved {out}")
    finally:
        if os.path.exists("matrices"):
            shutil.rmtree("matrices")
        if backed_up:
            shutil.move(_MATRIX_BACKUP, "matrices")


if __name__ == "__main__":
    main()
