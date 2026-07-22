from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as pl
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from fig_common import get_get_nuk_thm7, setup_latex  # noqa: E402
from simulationcommon import (  # noqa: E402
    calculate_a0_t_star,
    calculate_t_star,
    calculate_u0_post_t_star,
    common_simulation_parameters,
    generate_matrix_Ac,
    generate_matrix_Bc,
    get_discrete_ab,
)


V0_VALUES = np.linspace(25, 35, 10)
FIXED_NET_GAP = 18.3
VEHICLE_LENGTH = 4.7
R_STANDSTILL = 5.0

GAMMA = 1.5
ETA = 1.0 / 6.0
BRAKE_TIME = 0.0
OBSTACLE_DISTANCE = 380.0
P_DROP = 0.8

N_REPEAT = 10
SEEDS = tuple(range(N_REPEAT))
STOP_SPEED = 0.01
EXTENDED_END_TIME = 80.0

# Save and load generated files locally beside this script.
OUT_DIR = SCRIPT_DIR / "figures_out"
CACHE_DIR = (
    OUT_DIR
    / "cache_dmin_vmin0_obstacle380_v22_30"
)
OUT_FILE = (
    OUT_DIR
    / "dmin_and_obstacle_distance_vs_v0.pdf"
)


def construct_fixed_gap_x0(sp: dict) -> np.matrix:
    """Construct x(0) with 18 m net gaps among the actual vehicles."""
    n = sp["n"]
    p0 = sp["p0"]
    v0 = sp["v0"]
    a0 = sp["a0"]
    h = sp["h"]

    x = np.matrix(np.zeros((3 + 6 * n, 1)))
    x[0, 0], x[1, 0], x[2, 0] = p0, v0, a0

    # Vehicle 1 is initialized at CTG equilibrium relative to the virtual
    # reference.  The fixed 18 m net gap applies among actual vehicles.
    p1 = p0 - (VEHICLE_LENGTH + R_STANDSTILL + h * v0)
    physical_spacing = VEHICLE_LENGTH + FIXED_NET_GAP

    for i in range(1, n + 1):
        base = 3 + 6 * (i - 1)
        p_i = p1 if i == 1 else p1 - physical_spacing * (i - 1)
        p_prev = p0 if i == 1 else float(x[5 + 6 * (i - 2), 0])
        v_prev = v0 if i == 1 else float(x[6 + 6 * (i - 2), 0])

        # e_i, dot(e_i), p_i, v_i, a_i, u_i
        x[base, 0] = (
            (p_prev - p_i - VEHICLE_LENGTH)
            - (R_STANDSTILL + h * v0)
        )
        x[base + 1, 0] = v_prev - v0 - h * a0
        x[base + 2, 0] = p_i
        x[base + 3, 0] = v0
        x[base + 4, 0] = a0
        x[base + 5, 0] = 0.0

    return x


def current_minimum_gap(x: np.matrix, n: int) -> float:
    """Return the current minimum net gap among actual vehicles 1--n."""
    gaps = []
    for i in range(2, n + 1):
        p_prev = float(x[5 + 6 * (i - 2), 0])
        p_i = float(x[5 + 6 * (i - 1), 0])
        gaps.append(p_prev - p_i - VEHICLE_LENGTH)
    return min(gaps)


def enforce_nonnegative_speeds(
    x: np.matrix,
    x_before: np.matrix,
    u: np.matrix,
    stopped: np.ndarray,
    sp: dict,
) -> None:
    """Freeze each vehicle when its velocity reaches the stopping threshold."""
    n = sp["n"]
    h = sp["h"]

    # vehicle number, position index, velocity index, acceleration index,
    # stored desired-acceleration index, and input-vector index.
    state_indices = [(0, 0, 1, 2, None, 0)]
    for i in range(1, n + 1):
        base = 3 + 6 * (i - 1)
        state_indices.append(
            (i, base + 2, base + 3, base + 4, base + 5, i)
        )

    for vehicle, p_idx, v_idx, a_idx, stored_u_idx, input_idx in state_indices:
        old_p = float(x_before[p_idx, 0])
        old_v = float(x_before[v_idx, 0])
        new_p = float(x[p_idx, 0])
        new_v = float(x[v_idx, 0])

        if stopped[vehicle]:
            x[p_idx, 0] = old_p
            x[v_idx, 0] = 0.0
            x[a_idx, 0] = 0.0
            u[input_idx, 0] = 0.0
            if stored_u_idx is not None:
                x[stored_u_idx, 0] = 0.0
            continue

        if new_v <= STOP_SPEED:
            # Interpolate the stopping position if a finite step crosses v=0.
            if old_v > 0.0 and new_v < 0.0:
                fraction = old_v / (old_v - new_v)
                x[p_idx, 0] = old_p + fraction * (new_p - old_p)
            else:
                x[p_idx, 0] = new_p

            x[v_idx, 0] = 0.0
            x[a_idx, 0] = 0.0
            u[input_idx, 0] = 0.0
            if stored_u_idx is not None:
                x[stored_u_idx, 0] = 0.0
            stopped[vehicle] = True

    # Recompute e_i and dot(e_i) after applying the stopping constraint.
    for i in range(1, n + 1):
        base = 3 + 6 * (i - 1)
        p_i = float(x[base + 2, 0])
        v_i = float(x[base + 3, 0])
        a_i = float(x[base + 4, 0])

        if i == 1:
            p_prev = float(x[0, 0])
            v_prev = float(x[1, 0])
        else:
            prev_base = 3 + 6 * (i - 2)
            p_prev = float(x[prev_base + 2, 0])
            v_prev = float(x[prev_base + 3, 0])

        x[base, 0] = (
            (p_prev - p_i - VEHICLE_LENGTH)
            - (R_STANDSTILL + h * v_i)
        )
        x[base + 1, 0] = v_prev - v_i - h * a_i


def prepare_numerics(sp: dict):
    """Load the original exact discrete matrices and Theorem-7 step rule."""
    matrix_ac = generate_matrix_Ac(
        sp["n"], sp["tau_d"], sp["h"], sp["k_p"], sp["k_d"]
    )
    matrix_bc = generate_matrix_Bc(
        sp["n"], sp["tau_d"], sp["h"], sp["k_p"], sp["k_d"]
    )

    original_cwd = Path.cwd()
    try:
        os.chdir(OUT_DIR)
        discrete_ab = get_discrete_ab(
            sp["n"],
            sp["tau_d"],
            sp["h"],
            sp["k_p"],
            sp["k_d"],
            sp["Nbar"],
            sp["T"],
        )
    finally:
        os.chdir(original_cwd)

    get_nuk = get_get_nuk_thm7(sp, matrix_ac, matrix_bc)
    return discrete_ab, get_nuk


def simulate_one_seed(sp: dict, discrete_ab, get_nuk, seed: int):
    """Simulate one packet-loss realization over the complete braking event."""
    rng = np.random.RandomState(seed)

    T = sp["T"]
    nbar = sp["Nbar"]
    n = sp["n"]
    tau_d = sp["tau_d"]

    x = construct_fixed_gap_x0(sp)
    u = np.matrix(np.zeros((1 + n, 1)))
    stopped = np.zeros(n + 1, dtype=bool)
    p1_initial = float(x[5, 0])

    last_received = [
        float(x[8 + 6 * (i - 1), 0]) for i in range(1, n)
    ]

    discrete_step = 0
    time = 0.0
    dmin = current_minimum_gap(x, n)
    stopping_distance = np.nan
    stopping_time = np.nan

    t_star = None
    t_star_in_steps = None
    a0_at_t_star = None

    while time <= EXTENDED_END_TIME:
        proposed_nuk = get_nuk(x, u)
        nuk_to_communication = nbar - (discrete_step % nbar)
        actual_nuk = max(1, min(proposed_nuk, nuk_to_communication))
        matrix_a_d, matrix_b_d = discrete_ab[actual_nuk]

        if t_star is None:
            t_star = calculate_t_star(
                float(x[1, 0]),
                float(x[2, 0]),
                BRAKE_TIME,
                GAMMA,
                ETA,
                tau_d,
            )
            a0_at_t_star = calculate_a0_t_star(
                float(x[2, 0]),
                BRAKE_TIME,
                t_star,
                GAMMA,
                tau_d,
            )
            t_star_in_steps = nbar * math.ceil(t_star / T)

        if stopped[0]:
            u[0, 0] = 0.0
        elif discrete_step < t_star_in_steps:
            u[0, 0] = -GAMMA
        else:
            u[0, 0] = calculate_u0_post_t_star(
                (T / nbar) * discrete_step,
                t_star,
                a0_at_t_star,
                GAMMA / ETA,
                GAMMA,
                ETA,
                tau_d,
            )

        if discrete_step % nbar == 0:
            for predecessor in range(1, n):
                received = rng.choice(
                    [0, 1],
                    p=[P_DROP, 1.0 - P_DROP],
                )
                if received == 1:
                    state_row = 8 + 6 * (predecessor - 1)
                    last_received[predecessor - 1] = float(
                        x[state_row, 0]
                    )

        u[1, 0] = u[0, 0]
        for predecessor in range(1, n):
            u[1 + predecessor, 0] = last_received[predecessor - 1]
        for vehicle in range(1, n + 1):
            if stopped[vehicle]:
                u[vehicle, 0] = 0.0

        x_before = x.copy()
        x = matrix_a_d * x + matrix_b_d * u
        enforce_nonnegative_speeds(x, x_before, u, stopped, sp)

        time += actual_nuk * T / nbar
        discrete_step += actual_nuk
        dmin = min(dmin, current_minimum_gap(x, n))

        if np.isnan(stopping_time) and stopped[1]:
            stopping_distance = float(x[5, 0]) - p1_initial
            stopping_time = time

        if np.all(stopped):
            return max(0.0, dmin), stopping_distance, stopping_time

    return max(0.0, dmin), stopping_distance, stopping_time


def cache_file(v0: float) -> Path:
    """Return the original NPZ cache filename for one initial speed."""
    return CACHE_DIR / (
        f"v0_{v0:.4f}_gap_{FIXED_NET_GAP:.1f}_gamma_{GAMMA:.2f}_"
        f"eta_{ETA:.6f}_p_{P_DROP:.2f}_n_{N_REPEAT}_vmin0.npz"
    )


def run_speed(v0: float, discrete_ab, get_nuk):
    """Run or load all packet-loss realizations for one initial speed."""
    filename = cache_file(v0)

    if filename.exists():
        with np.load(filename) as data:
            if all(key in data.files for key in ("dmin", "stop", "stop_time")):
                print(f"Loading cached results: {filename.name}")
                return data["dmin"], data["stop"], data["stop_time"]

    def simulate_seed(seed: int):
        sp = common_simulation_parameters()
        sp["v0"] = float(v0)
        sp["break_time"] = BRAKE_TIME
        sp["gamma"] = GAMMA
        sp["eta"] = ETA
        sp["prob_drop"] = P_DROP
        sp["tend"] = EXTENDED_END_TIME
        return simulate_one_seed(sp, discrete_ab, get_nuk, seed)

    max_workers = min(4, N_REPEAT)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(simulate_seed, SEEDS))

    dmins = np.zeros(N_REPEAT)
    stopping_distances = np.zeros(N_REPEAT)
    stopping_times = np.zeros(N_REPEAT)

    for index, (seed, result) in enumerate(zip(SEEDS, results)):
        dmin, stopping_distance, stopping_time = result
        dmins[index] = dmin
        stopping_distances[index] = stopping_distance
        stopping_times[index] = stopping_time

        print(
            f"v0={v0:.4f} m/s, seed={seed}: "
            f"d_min'={dmin:.4f} m, "
            f"stopping distance={stopping_distance:.4f} m, "
            f"stopping time={stopping_time:.4f} s"
        )

    np.savez(
        filename,
        dmin=dmins,
        stop=stopping_distances,
        stop_time=stopping_times,
    )
    print(f"Saved cache: {filename}")
    return dmins, stopping_distances, stopping_times


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_cases_cached = all(
        cache_file(float(v0)).exists() for v0 in V0_VALUES
    )

    if all_cases_cached:
        discrete_ab = None
        get_nuk = None
        print("All Fig. 3 simulation cases were found in the NPZ cache.")
    else:
        base_sp = common_simulation_parameters()
        base_sp["v0"] = float(V0_VALUES[0])
        base_sp["break_time"] = BRAKE_TIME
        base_sp["gamma"] = GAMMA
        base_sp["eta"] = ETA
        base_sp["prob_drop"] = P_DROP
        base_sp["tend"] = EXTENDED_END_TIME
        discrete_ab, get_nuk = prepare_numerics(base_sp)

    median_dmins = np.zeros(len(V0_VALUES))
    q25_dmins = np.zeros(len(V0_VALUES))
    q75_dmins = np.zeros(len(V0_VALUES))
    mean_stopping_distances = np.zeros(len(V0_VALUES))
    remaining_obstacle_distances = np.zeros(len(V0_VALUES))

    for index, v0 in enumerate(V0_VALUES):
        dmins, stopping_distances, _ = run_speed(
            float(v0), discrete_ab, get_nuk
        )

        median_dmins[index] = np.median(dmins)
        q25_dmins[index] = np.percentile(dmins, 25)
        q75_dmins[index] = np.percentile(dmins, 75)
        mean_stopping_distances[index] = np.nanmean(stopping_distances)

        # Zero is used once the predicted stopping distance reaches or exceeds
        # the 380 m available distance to the obstacle.
        remaining_obstacle_distances[index] = max(
            0.0,
            OBSTACLE_DISTANCE - np.nanmax(stopping_distances),
        )

        print(
            f"v0={v0:.4f} m/s: "
            f"median d_min'={median_dmins[index]:.4f} m, "
            f"mean stopping distance={mean_stopping_distances[index]:.4f} m, "
            f"remaining obstacle distance="
            f"{remaining_obstacle_distances[index]:.4f} m"
        )

    setup_latex()
    pl.rcParams["mathtext.fontset"] = "cm"

    fig, ax_left = pl.subplots(figsize=(6, 3))

    dmin_line, = ax_left.plot(
        V0_VALUES,
        median_dmins,
        color="#d62728",
        linewidth=2,
        markersize=4,
        label=r"Median $d'_{\min}$",
    )
    iqr_band = ax_left.fill_between(
        V0_VALUES,
        q25_dmins,
        q75_dmins,
        color="#d62728",
        alpha=0.2,
        label="IQR (25%-75%)",
    )
    ax_left.set_xlabel(r"$v_0(0)\;[\mathrm{m/s}]$", color="black")
    ax_left.set_ylabel(
        r"$d'_{\min}\;[\mathrm{m}]$",
        color="black",
    )
    ax_left.tick_params(axis="x", colors="black")
    ax_left.tick_params(axis="y", colors="black")
    ax_left.grid(True, linestyle=":", alpha=0.6)

    ax_right = ax_left.twinx()
    obstacle_line, = ax_right.plot(
        V0_VALUES,
        remaining_obstacle_distances,
        color="#1f77b4",
        linewidth=2,
        markersize=4,
        label=r"$d_{\mathrm{obs}}$",
    )
    ax_right.set_ylabel(
        r"$d_{\mathrm{obs}}\;[\mathrm{m}]$",
        color="black",
    )
    ax_right.tick_params(axis="y", colors="black")
    # ax_right.set_ylim(bottom=0)

    ax_left.legend(
        [iqr_band, dmin_line, obstacle_line],
        [
            r"IQR (25\%-75\%)",
            r"Median $d'_{\min}$",
            r"$d_{\mathrm{obs}}$",
        ],
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(OUT_FILE)
    pl.close(fig)
    print(f"\nSaved figure: {OUT_FILE}")


if __name__ == "__main__":
    main()
