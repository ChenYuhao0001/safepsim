import os
import numpy as np
import matplotlib.pyplot as pl
import scipy.linalg as la
import math

import simulationcommon
from simulationcommon import common_simulation_parameters, get_value, generate_matrix_Ac, generate_matrix_Bc, convertc2d

pl.rcParams['text.usetex'] = True
pl.rc('text.latex', preamble="\\usepackage{amsmath}\n \\usepackage{amssymb}")
pl.rcParams['font.size'] = 13
pl.rcParams['axes.labelsize'] = 13
pl.rcParams['xtick.labelsize'] = 13
pl.rcParams['ytick.labelsize'] = 13


def patched_get_discrete_ab(n, tau_d, h, k_p, k_d, Nbar, T):
    cache_dir = "matrices_reaction_experiment"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    filenamea = f"{cache_dir}/Amatrices_Nbar_{Nbar}_T_{T:.3f}_kp_{k_p:.3f}_kd_{k_d:.3f}_h_{h:.3f}.npy"
    filenameb = f"{cache_dir}/Bmatrices_Nbar_{Nbar}_T_{T:.3f}_kp_{k_p:.3f}_kd_{k_d:.3f}_h_{h:.3f}.npy"

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
        for i in range(1, Nbar + 1):
            As.append(Ad @ As[-1])
            Bs.append(Ad @ Bs[-1] + Bd)
        np.save(filenamea, As)
        np.save(filenameb, Bs)
        return [list(pair) for pair in zip(As, Bs)]


simulationcommon.get_discrete_ab = patched_get_discrete_ab
print("[System] Monkey Patch applied: Redirecting cache to 'matrices_reaction_experiment/'.")


def get_get_nuk_thm7(simulation_parameters, matrix_Ac, matrix_Bc):
    T = simulation_parameters["T"]
    Nbar = simulation_parameters["Nbar"]
    alpha = simulation_parameters["alpha"]
    dim_x = matrix_Ac.shape[0]
    dim_u = matrix_Bc.shape[1]
    n = simulation_parameters["n"]
    tilde_A = np.block([[matrix_Ac, matrix_Bc], [np.zeros((dim_u, dim_x)), np.zeros((dim_u, dim_u))]])
    symm_part = (tilde_A + tilde_A.T) / 2
    mu_tilde_A = np.max(np.real(la.eigvals(symm_part)))
    phi_norms = []
    top_block_of_tilde_A = np.hstack([matrix_Ac, matrix_Bc])
    for i in range(1, n + 1):
        qi = np.zeros((dim_x, 1))
        if i == 1:
            p_i_minus_1_idx = 0;
            p_i_idx = 5
        else:
            p_i_minus_1_idx = 5 + 6 * (i - 2);
            p_i_idx = 5 + 6 * (i - 1)
        qi[p_i_minus_1_idx, 0] = 1;
        qi[p_i_idx, 0] = -1
        phi_norms.append(la.norm(qi.T @ top_block_of_tilde_A))
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


def run_experiment():
    output_dir = "experiment_reaction_result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reaction_times = np.arange(0.5, 1.3, 0.025)
    v0_target = 30.0

    configs = [
        {"label": "$k_{\\mathrm{p}}=0.2$, $k_{\\mathrm{d}}=0.6$", "kp": 0.2, "kd": 0.6, "color": "C0", "marker": "o",
         "ls": "solid"},
        {"label": "$k_{\\mathrm{p}}=0.25$, $k_{\\mathrm{d}}=0.6$", "kp": 0.25, "kd": 0.6, "color": "C2", "marker": "s",
         "ls": "dashed"},
        {"label": "$k_{\\mathrm{p}}=0.25$, $k_{\\mathrm{d}}=0.8$", "kp": 0.25, "kd": 0.8, "color": "C1", "marker": "x",
         "ls": "dotted"}
    ]

    results = {}
    print(f"--- Simulation Start (v0={v0_target} m/s) ---")

    base_params = common_simulation_parameters()
    base_params["v0"] = v0_target

    _, _, k_fin, x_hist, _ = get_value(base_params, get_get_nuk_thm7, description=None)

    if isinstance(x_hist, str):
        print("[Error] x_hist is still a string!")
        return

    p0_start = x_hist[0, 0]
    p0_end = x_hist[0, k_fin]
    leader_stop_dist = p0_start - p0_end
    print(f"Information: Virtual Leader Stopping Distance = {leader_stop_dist:.2f} m")

    for cfg in configs:
        kp = cfg["kp"]
        kd = cfg["kd"]
        label = cfg["label"]
        dmin_list = []
        print(f"Running config: {label} ...")

        for h in reaction_times:
            params = common_simulation_parameters()
            params["v0"] = v0_target
            params["h"] = h
            params["k_p"] = kp
            params["k_d"] = kd

            run_desc = f"matrices_reaction_experiment/run_kp{kp}_kd{kd}_h{h:.3f}"
            d_min_T, _, _, _, _ = get_value(params, get_get_nuk_thm7, description=run_desc)
            dmin_list.append(d_min_T)

        results[label] = dmin_list

    pl.figure(figsize=(6, 3))
    for cfg in configs:
        label = cfg["label"]

        a = 0.6 if cfg["color"] == "C2" else 1.0

        pl.plot(reaction_times, results[label],
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle=cfg["ls"],
                linewidth=2,
                label=label,
                markerfacecolor='none',
                markeredgewidth=1.5,
                alpha=a)

    pl.xlabel(r"$h$\,\, [$\mathrm{s}$]")
    pl.ylabel(r'$d^{\prime}_{\min}\,\,$ [$\mathrm{m}$]')
    pl.legend()
    pl.grid(True, linestyle=":")
    pl.tight_layout()

    pdf_path = os.path.join(output_dir, "dmin_vs_reaction_speed.pdf")
    pl.savefig(pdf_path, bbox_inches='tight')
    print(f"Done. Plot saved to {pdf_path}")

    data_path = os.path.join(output_dir, "simulation_data.npz")
    np.savez(data_path,
             reaction_times=reaction_times,
             config1_data=results[configs[0]["label"]],
             config2_data=results[configs[1]["label"]])


if __name__ == "__main__":
    run_experiment()