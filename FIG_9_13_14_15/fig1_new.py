import numpy as np
import numpy.random as ra
import matplotlib.pyplot as pl
import scipy.linalg as la
import math
import os

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

P_VALUES = [0.2, 0.8]
GAMMA_VALUES = [1.0, 1.2, 1.5]
ETA_FIXED = 0.15
N_REPEAT = 10
SEEDS = list(range(N_REPEAT))
STOP_SPEED = 0.01
TEND_EXTENDED = 60.0
DMIN_WINDOW = 25.0


OUT_DIR = "figures_out"
CACHE_DIR = "cache_distance_vs_dmin_fixed_eta"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def simulate_random_with_leader_traj(sim_params, get_get_nuk):
    T = sim_params["T"]
    Nbar = sim_params["Nbar"]
    tau_d = sim_params["tau_d"]
    h = sim_params["h"]
    k_p = sim_params["k_p"]
    k_d = sim_params["k_d"]
    n = sim_params["n"]
    p0 = sim_params["p0"]
    v0 = sim_params["v0"]
    a0 = sim_params["a0"]
    tend = sim_params["tend"]
    kend = Nbar * int(tend / T)

    break_time = sim_params.get("break_time", 5.0)
    break_step = Nbar * int(break_time / T)
    gamma = sim_params["gamma"]
    eta = sim_params["eta"]
    packet_drop_probability = sim_params.get("prob_drop", 0.0)

    matrix_Ac = generate_matrix_Ac(n, tau_d, h, k_p, k_d)
    matrix_Bc = generate_matrix_Bc(n, tau_d, h, k_p, k_d)
    discrete_ab = get_discrete_ab(n, tau_d, h, k_p, k_d, Nbar, T)
    get_nuk_func = get_get_nuk(sim_params, matrix_Ac, matrix_Bc)

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

    p1_at_brake = None
    p1_at_stop = None
    v1_stopped = False
    k_dmin_window = None                 # last column index with t <= DMIN_WINDOW

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
                p1_at_brake = xks[5, k]          # vehicle 1 position at braking onset
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

        # record first time vehicle 1 (platoon head) has effectively stopped
        if (not v1_stopped) and (discrete_step >= break_step) \
                and (xks[6, k] <= STOP_SPEED):
            p1_at_stop = xks[5, k]
            v1_stopped = True

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

        if k_dmin_window is None and time > DMIN_WINDOW:
            k_dmin_window = k + 1

        if v1_stopped:
            break

    k_eval = k_dmin_window if k_dmin_window is not None else (k + 1)
    d_min_T = calculate_minimum_distance(xks[:, :k_eval], n)

    if p1_at_brake is None:
        p1_at_brake = xks[5, 0]
    if p1_at_stop is None:
        p1_at_stop = xks[5, k]

    stopping_distance = float(p1_at_stop - p1_at_brake)
    return d_min_T, stopping_distance


def run_combo(p, gamma):
    cache = os.path.join(
        CACHE_DIR,
        f"p_{p:.2f}_gamma_{gamma:.2f}_eta_{ETA_FIXED:.6f}.npz",
    )
    if os.path.exists(cache):
        data = np.load(cache)
        return data["dmin"], data["stop"]

    dmins = np.zeros(N_REPEAT)
    stops = np.zeros(N_REPEAT)
    for r, seed in enumerate(SEEDS):
        ra.seed(seed)
        sp = common_simulation_parameters()
        sp["prob_drop"] = p
        sp["gamma"] = gamma
        sp["eta"] = ETA_FIXED
        sp["tend"] = TEND_EXTENDED
        dmin, stop = simulate_random_with_leader_traj(sp, get_get_nuk_thm7)
        dmins[r] = dmin
        stops[r] = stop
        print(f"  [p={p:.2f}, gamma={gamma:.2f}, eta={ETA_FIXED:.6f}] "
              f"rep {r+1}/{N_REPEAT}: d_min={dmin:.3f}, stop_dist={stop:.2f}")
    np.savez(cache, dmin=dmins, stop=stops)
    return dmins, stops


def main():
    cmap = pl.get_cmap("tab10")
    combos = [(p, g) for p in P_VALUES for g in GAMMA_VALUES]
    colors = {combo: cmap(i) for i, combo in enumerate(combos)}
    markers = {p: m for p, m in zip(P_VALUES, ["o", "s"])}

    pl.figure(figsize=(6, 3))
    for (p, gamma) in combos:
        dmins, stops = run_combo(p, gamma)
        x_mean, x_std = dmins.mean(), dmins.std()
        y_mean, y_std = stops.mean(), stops.std()
        pl.errorbar(
            x_mean, y_mean, xerr=x_std, yerr=y_std,
            fmt=markers[p], color=colors[(p, gamma)], markersize=7,
            capsize=3, elinewidth=1, markeredgecolor="black", markeredgewidth=0.5,
            label=rf"$p={p},\ \gamma={gamma}$")

    pl.xlabel(r"$d'_{\min}\ [\mathrm{m}]$")
    pl.ylabel(r"Distance traveled by 1st (lead) vehicle $[\mathrm{m}]$")
    pl.grid(True, linestyle=":", alpha=0.6)
    pl.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    pl.tight_layout()
    out = os.path.join(
        OUT_DIR,
        "distance_vs_dmin_fixed_eta.pdf",
    )
    pl.savefig(out)
    pl.close()
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
