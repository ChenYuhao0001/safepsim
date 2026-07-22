import numpy as np
import scipy.linalg as la
import math
import matplotlib.pyplot as pl


def setup_latex():
    try:
        pl.rcParams["text.usetex"] = True
        pl.rc("text.latex",
              preamble="\\usepackage{amsmath}\n\\usepackage{amssymb}")
        pl.rcParams["font.family"] = "serif"
    except Exception:
        print("LaTeX not available; using default fonts.")


def get_get_nuk_thm7(simulation_parameters, matrix_Ac, matrix_Bc):
    """Theorem 7 lifted-state adaptive step rule (identical to the paper code)."""
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
