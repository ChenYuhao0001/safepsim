# SafePSim

SafePSim is a Python simulation framework for safety and fuel-efficiency analysis of vehicle platoons under communication losses and sudden braking.

The framework computes inter-vehicle distances at simulation time instants and uses adaptive simulation intervals to provide guaranteed bounds on distance-approximation errors. It also supports fuel-efficiency analysis with guaranteed error bounds for the calculated average fuel savings.

The simulations consider Predecessor-Following vehicle platoons, Constant Time Gap and Constant Distance Gap spacing policies, consecutive packet losses, and random packet losses.

<!-- IMAGE NEEDED
Suggested image: Illustration of the highway sudden-braking scenario with a merging lane.
Suggested path: images/sudden_braking_scenario.png

After adding the image, uncomment:
![Highway sudden-braking scenario](images/sudden_braking_scenario.png)
-->

# Requirements

SafePSim requires Python 3 and the following libraries:

```text
numpy
scipy
matplotlib
```

Some plotting scripts use LaTeX text rendering. A LaTeX installation may therefore be required to reproduce all plots.

# Installation

Clone this repository:

```bash
git clone https://github.com/ChenYuhao0001/safepsim.git
cd safepsim
```

We recommend using a Python virtual environment:

```bash
python -m venv env
```

Activate the environment on macOS or Linux:

```bash
source env/bin/activate
```

Activate it on Windows:

```powershell
env\Scripts\activate
```

Install the required libraries:

```bash
pip install numpy scipy matplotlib
```

# Basic Usage

SafePSim consists of individual simulation and plotting scripts.

## Safety analysis

Run the main safety simulations based on the adaptive simulation approaches:

```bash
python simulate_generate_distance_plots_3.py
```

These simulations evaluate the minimum of minimum inter-vehicle distances under sudden braking and communication losses and compare the required total simulation steps.

## Random packet-loss analysis

Generate distributions of the minimum inter-vehicle distance:

```bash
python histogram_2.py
```

Evaluate the median and interquartile range for different packet-loss probabilities:

```bash
python histogram_5_SDP_2575.py
```

## Effect of the time gap

Investigate how the time gap \(h\) affects platoon safety under consecutive packet losses:

```bash
python "generate_distance_plots_reaction_transparent facecolor.py"
```

## Vehicle trajectories and braking behavior

Generate vehicle-position trajectories and desired-acceleration, acceleration, and velocity profiles:

```bash
python generate_distance_plots_u.py
```

## Fuel-efficiency analysis

Evaluate the trade-off between the minimum inter-vehicle distance and the average fuel saved for all vehicles:

```bash
python "simulate_generate_distance_plots_d'min vs total fuel saving with error bars and speed up down_3.py"
```

The analysis also calculates guaranteed bounds on the simulation error in the average fuel savings.

<!-- IMAGE NEEDED
Suggested image: Safety and average-fuel-saving trade-off under different control parameters.
Suggested path: images/safety_fuel_tradeoff.png

After adding the image, uncomment:
![Safety and fuel-efficiency trade-off](images/safety_fuel_tradeoff.png)
-->

# Spacing Policies

## Constant Time Gap Policy

The `CTG_PF` directory contains the Predecessor-Following platoon model under the Constant Time Gap spacing policy.

The desired spacing contains a standstill distance and a velocity-dependent time-gap term.

```bash
python "CTG_PF/simulate_generate_distance_plots_d'min_vs_fuelsaving_CTG_PF.py"
```

## Constant Distance Gap Policy

The `CDG_PF` directory contains the corresponding model under the Constant Distance Gap spacing policy.

The spacing error is defined using a fixed inter-vehicle distance.

```bash
python "CDG_PF/simulate_generate_distance_plots_d'min_vs_fuelsaving_CDG_PF.py"
```

If a script inside `CTG_PF` or `CDG_PF` cannot locate a root-level support module, add the repository root to `PYTHONPATH` before running it.

<!-- IMAGE NEEDED
Suggested image: Side-by-side comparison of the Constant Time Gap and Constant Distance Gap results.
Suggested path: images/ctg_cdg_comparison.png

After adding the image, uncomment:
![Comparison of CTG and CDG spacing policies](images/ctg_cdg_comparison.png)
-->

# Additional Analysis Scripts

The scripts in `FIG_9_13_14_15` investigate:

- The effects of packet-loss probability and braking deceleration on the minimum inter-vehicle distance.
- The effect of the characteristic time constant \(\tau_d\) on platoon safety.
- The effect of initial velocity on the minimum inter-vehicle distance and the remaining distance to an obstacle.
- The computational efficiency of the constant-step baseline and adaptive simulation approaches.

Run these scripts from their directory:

```bash
cd FIG_9_13_14_15
python fig1_new.py
python fig2_dmin_vs_taud.py
python fig3_new.py
python fig7_new.py
```

The current `fig7_new.py` expects `simulationcommon.py` and `simulate_generate_distance_plots_3.py` to be available in its configured source directory.

# File Descriptions

## Main Files

| File | Purpose |
|---|---|
| `simulationcommon.py` | Models the linear vehicle-platoon dynamics under sudden braking and communication losses and computes the minimum inter-vehicle distance. |
| `simulationcommon_3.py` | Simulates the continuous velocity-oscillation scenario in which the virtual reference vehicle repeatedly accelerates and decelerates. |
| `simulate_generate_distance_plots_3.py` | Evaluates platoon safety and computational efficiency using the adaptive simulation approaches. |
| `generate_distance_plots_u.py` | Generates vehicle trajectories and braking-related desired-acceleration, acceleration, and velocity profiles. |
| `generate_distance_plots_reaction_transparent facecolor.py` | Investigates how the time gap \(h\) affects platoon safety under consecutive packet losses. |
| `histogram_2.py` | Evaluates the distribution of the minimum inter-vehicle distance under random packet losses. |
| `histogram_5_SDP_2575.py` | Evaluates the median and interquartile range of the minimum inter-vehicle distance for different packet-loss probabilities. |
| `simulate_generate_distance_plots_d'min vs total fuel saving with error bars and speed up down_3.py` | Investigates the safety and fuel-efficiency trade-off and computes guaranteed simulation-error bounds. |

## Constant Distance Gap Model

| File | Purpose |
|---|---|
| `CDG_PF/simulationcommon_3_CDG_PF.py` | Models the platoon dynamics under the Predecessor-Following topology and Constant Distance Gap spacing policy. |
| `CDG_PF/simulate_generate_distance_plots_d'min_vs_fuelsaving_CDG_PF.py` | Evaluates the relationship between platoon safety and average fuel savings under the Constant Distance Gap policy. |

## Constant Time Gap Model

| File | Purpose |
|---|---|
| `CTG_PF/simulationcommon_3_CTG_PF.py` | Models the platoon dynamics under the Predecessor-Following topology and Constant Time Gap spacing policy. |
| `CTG_PF/simulate_generate_distance_plots_d'min_vs_fuelsaving_CTG_PF.py` | Evaluates the relationship between platoon safety and average fuel savings under the Constant Time Gap policy. |

## Supporting Analysis Files

| File | Purpose |
|---|---|
| `FIG_9_13_14_15/fig_common.py` | Provides shared plotting settings and the adaptive time-selection rule. |
| `FIG_9_13_14_15/simulationcommon.py` | Provides the vehicle-platoon model and numerical functions required by the analysis scripts. |
| `FIG_9_13_14_15/fig1_new.py` | Investigates the effects of packet-loss probability and braking deceleration on safety and stopping distance. |
| `FIG_9_13_14_15/fig2_dmin_vs_taud.py` | Investigates how the characteristic time constant \(\tau_d\) affects the minimum inter-vehicle distance. |
| `FIG_9_13_14_15/fig3_new.py` | Investigates how initial velocity affects platoon safety and the remaining distance to an obstacle. |
| `FIG_9_13_14_15/fig7_new.py` | Compares the total simulation steps required by the constant-step baseline and adaptive simulation approaches. |

# Generated Results

Depending on the selected experiment, the scripts generate PDF or PNG plots, NumPy data files, detailed text reports, and error-analysis tables.

Generated files may be stored in directories such as:

```text
figures_out/
plots/
Platoon_Analysis_Result/
experiment_reaction_result/
Theorem7_probability_analysis/
```

The simulations may also create `.npy` or `.npz` cache files. Existing cached results can be reused to avoid repeating computationally expensive simulations.

# Related Paper

This repository accompanies:

> Yuhao Chen and Ahmet Cetinkaya, “A Simulation Framework with Guaranteed Error Bounds for Safety and Fuel-Efficiency Analysis of Vehicle Platoons.”

# License

This project is distributed under the GNU General Public License v3.0. See `LICENSE` for details.
