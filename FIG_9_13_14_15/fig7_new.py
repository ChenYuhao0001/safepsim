from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as pl
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SUPPLIED_SOURCE_DIR = Path(r"C:\Users\73265\Downloads")


def has_support_files(directory: Path) -> bool:
    return (
        (directory / "simulationcommon.py").exists()
        and (directory / "simulate_generate_distance_plots_3.py").exists()
    )


if has_support_files(SCRIPT_DIR):
    SOURCE_DIR = SCRIPT_DIR
elif has_support_files(SUPPLIED_SOURCE_DIR):
    SOURCE_DIR = SUPPLIED_SOURCE_DIR
else:
    raise FileNotFoundError(
        "Cannot find simulationcommon.py and "
        "simulate_generate_distance_plots_3.py. Put all three Python files "
        "in the same directory."
    )

sys.path.insert(0, str(SOURCE_DIR))
os.chdir(SOURCE_DIR)

from simulationcommon import common_simulation_parameters, get_value  # noqa: E402


ORIGINAL_STEPS_FILE = SOURCE_DIR / "simulate_generate_distance_plots_3.py"
_spec = importlib.util.spec_from_file_location("orig_steps", ORIGINAL_STEPS_FILE)
_orig = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_orig)

get_get_nuk_thm1 = _orig.get_get_nuk_thm1
get_get_nuk_thm2 = _orig.get_get_nuk_thm7


def get_get_nuk_constant(simulation_parameters, matrix_ac, matrix_bc):
    """Constant-step baseline: nu_k = 1, so step size is T/Nbar."""

    def get_nuk(xk, uk):
        return 1

    return get_nuk

KP_FIXED = 0.20
KDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.35,
    1.40,
    1.45,
    1.50,
]

OUT_DIR = SCRIPT_DIR / "figures_out"
CACHE_DIR = SCRIPT_DIR / "cache_fig7_early_stop"

COMBINED_FILE = OUT_DIR / "fig7_steps_with_baseline_combined.pdf"
THEOREMS_ONLY_FILE = OUT_DIR / "fig7_steps_theorems_only.pdf"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def setup_latex() -> None:
    """Use the same LaTeX-style rendering as the supplied figure code."""
    if shutil.which("latex") is not None:
        pl.rcParams["text.usetex"] = True
        pl.rc(
            "text.latex",
            preamble="\\usepackage{amsmath}\n\\usepackage{amssymb}",
        )
        pl.rcParams["font.family"] = "serif"
    else:
        pl.rcParams["text.usetex"] = False
        pl.rcParams["mathtext.fontset"] = "cm"
        pl.rcParams["font.family"] = "serif"
        print("LaTeX not available; using Matplotlib math text.")


def cache_description(method_name: str, sp: dict) -> str:
    """Build a parameter-specific cache key to prevent stale-cache reuse."""
    filename = (
        f"{method_name}_kp_{sp['k_p']:.3f}_kd_{sp['k_d']:.3f}_"
        f"T_{sp['T']:.6f}_Nbar_{sp['Nbar']}_tend_{sp['tend']:.3f}_"
        f"alpha_{sp['alpha']:.8g}_tau_{sp['tau_d']:.3f}_"
        f"h_{sp['h']:.3f}_n_{sp['n']}_p0_{sp['p0']:.3f}_"
        f"v0_{sp['v0']:.3f}_a0_{sp['a0']:.3f}_"
        f"tb_{sp['break_time']:.3f}_gamma_{sp['gamma']:.3f}_"
        f"eta_{sp['eta']:.6f}_ell_{sp['ell']}"
    )
    return str(CACHE_DIR / filename)


def steps_for(method_name: str, get_get_nuk, kd: float) -> tuple[int, float]:
    """Run/load the original simulation and return steps and terminal time."""
    sp = common_simulation_parameters()
    sp["k_p"] = KP_FIXED
    sp["k_d"] = kd

    # Do not use the old short cache key because it does not record alpha and
    # the other simulation settings.
    description = cache_description(method_name, sp)
    _, _, k_final, _, time_vector = get_value(
        sp,
        get_get_nuk,
        description,
    )

    final_time = float(time_vector[k_final])
    return int(k_final), final_time


def collect_results(series):
    """Collect step counts once and reuse them for both output figures."""
    all_results: dict[str, tuple[list[int], list[float]]] = {}

    for label, method_name, step_rule, _, _, _ in series:
        steps = []
        final_times = []

        for kd in KDS:
            k_final, final_time = steps_for(method_name, step_rule, kd)
            steps.append(k_final)
            final_times.append(final_time)

            stop_text = "early stop" if final_time < 24.99 else "full horizon"
            print(
                f"  {label:26s} k_d={kd:.2f}: "
                f"steps={k_final:,}, final_time={final_time:.6f} s "
                f"({stop_text})"
            )

        all_results[method_name] = (steps, final_times)

    return all_results


def plot_combined(series, all_results) -> None:
    """Plot baseline, Theorem 1, and Theorem 2 in one coordinate system."""
    fig, ax = pl.subplots(figsize=(6, 3))

    for label, method_name, _, color, marker, linestyle in series:
        steps, _ = all_results[method_name]
        ax.plot(
            KDS,
            steps,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4,
            linewidth=1.5,
            label=label,
        )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.set_xlabel(r"$k_\mathrm{d}$")
    ax.set_ylabel(r"Total simulation steps $k'_\mathrm{end}$")
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(COMBINED_FILE)
    pl.close(fig)

    print(f"\nSaved {COMBINED_FILE}")


def plot_theorems_only(series, all_results) -> None:
    """Plot only Theorem 1 and Theorem 2 in one coordinate system."""
    fig, ax = pl.subplots(figsize=(6, 3))

    for label, method_name, _, color, marker, linestyle in series[1:]:
        steps, _ = all_results[method_name]
        ax.plot(
            KDS,
            steps,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4,
            linewidth=1.5,
            label=label,
        )

    ax.ticklabel_format(
        axis="y",
        style="sci",
        scilimits=(0, 0),
        useMathText=True,
    )
    ax.set_xlabel(r"$k_\mathrm{d}$")
    ax.set_ylabel(r"Total simulation steps $k'_\mathrm{end}$")
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(THEOREMS_ONLY_FILE)
    pl.close(fig)

    print(f"Saved {THEOREMS_ONLY_FILE}")


def main() -> None:
    setup_latex()

    series = [
        (
            "Constant step (baseline)",
            "baseline",
            get_get_nuk_constant,
            "#d62728",
            "o",
            "-",
        ),
        (
            "Theorem 1",
            "theorem1",
            get_get_nuk_thm1,
            "#1f77b4",
            "s",
            "--",
        ),
        (
            "Theorem 2",
            "theorem2",
            get_get_nuk_thm2,
            "#2ca02c",
            "^",
            "-.",
        ),
    ]

    all_results = collect_results(series)
    plot_combined(series, all_results)
    plot_theorems_only(series, all_results)


if __name__ == "__main__":
    main()
