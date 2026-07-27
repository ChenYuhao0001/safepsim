# SafePSim

SafePSim is a simulation framework for analyzing the safety and fuel efficiency of vehicle platoons under sudden braking and communication losses.

The framework computes inter-vehicle distances across time, evaluates the minimum inter-vehicle distance, and provides guaranteed bounds on the error induced by simulation. It also supports fuel-efficiency analysis under different control parameters and spacing policies.

## Overview

SafePSim considers a highway vehicle platoon in which vehicles communicate through wireless links. The framework evaluates platoon safety under sudden braking, consecutive packet losses, and random packet losses.

📄 [View the highway vehicle platoon scenario](images/new%20illustration%20of%20vehicle%20platoon.pdf)

## Main Features

- Simulation of vehicle platoons under sudden braking
- Predecessor-Following communication topology
- Consecutive and random packet-loss models
- Constant Time Gap and Constant Distance Gap spacing policies
- Computation of the minimum inter-vehicle distance
- Adaptive selection of simulation times
- Guaranteed bounds on simulation-induced error
- Safety and fuel-efficiency analysis

## Safety Analysis

SafePSim evaluates safety using the minimum inter-vehicle distance attained during a simulation. A positive minimum distance indicates collision avoidance, whereas a nonpositive value indicates a collision.

The following result shows how the minimum inter-vehicle distance changes with the packet drop probability under random communication losses.

📄 [View minimum inter-vehicle distance versus packet drop probability](images/Plot_Theorem7_d_min_vs_p.pdf)

## Adaptive Simulation

The framework uses adaptive simulation times to identify potentially critical intervals and compute guaranteed error bounds. This reduces the number of simulation steps required to evaluate safety compared with a constant-step baseline.

📄 [View the comparison of total simulation steps](images/baseline_2thms_steps_combined.pdf)

## Safety and Fuel Efficiency

SafePSim studies the relationship between the minimum inter-vehicle distance and the average fuel saved by the vehicle platoon.

The results are provided for two spacing policies:

- Constant Time Gap policy
- Constant Distance Gap policy

### Constant Time Gap Policy

📄 [View the safety and fuel-efficiency results under CTG](images/Summary_Fuel_vs_Safety_CTG_PF.pdf)

### Constant Distance Gap Policy

📄 [View the safety and fuel-efficiency results under CDG](images/Summary_Fuel_vs_Safety_CDG_PF.pdf)

## Code Description

### Common Simulation Functions

| File | Purpose |
|---|---|
| `simulationcommon.py` | Contains common functions used to simulate vehicle-platoon dynamics and communication losses. |
| `simulationcommon_3.py` | Contains common simulation functions used by the fuel-efficiency analysis scripts. |

### Safety Analysis

| File | Purpose |
|---|---|
| `simulate_generate_distance_plots_3.py` | Simulates inter-vehicle distances and evaluates the minimum inter-vehicle distance. |
| `generate_distance_plots_u.py` | Generates plots associated with vehicle acceleration, velocity, and inter-vehicle distance. |
| `generate_distance_plots_reaction_transparent facecolor.py` | Analyzes inter-vehicle distances while accounting for braking reaction time. |
| `histogram_2.py` | Generates distributions of the minimum inter-vehicle distance from repeated simulations. |
| `histogram_5_SDP_2575.py` | Generates minimum-distance distributions for selected controller parameters and packet-loss settings. |

### Fuel-Efficiency Analysis

| File | Purpose |
|---|---|
| `simulate_generate_distance_plots_d'min vs total fuel saving with error bars and speed up down_3.py` | Evaluates the relationship between minimum inter-vehicle distance and total fuel savings, including simulation error bounds. |

### Simulation Folders

| Folder | Purpose |
|---|---|
| `CTG_PF` | Contains simulations using the Constant Time Gap policy and Predecessor-Following topology. |
| `CDG_PF` | Contains simulations using the Constant Distance Gap policy and Predecessor-Following topology. |
| `FIG_9_13_14_15` | Contains scripts used to generate selected safety-analysis results. |

## Repository Structure

```text
safepsim/
├── README.md
├── images/
│   ├── new illustration of vehicle platoon.pdf
│   ├── Plot_Theorem7_d_min_vs_p.pdf
│   ├── baseline_2thms_steps_combined.pdf
│   ├── Summary_Fuel_vs_Safety_CTG_PF.pdf
│   └── Summary_Fuel_vs_Safety_CDG_PF.pdf
├── CDG_PF/
├── CTG_PF/
├── FIG_9_13_14_15/
├── simulationcommon.py
├── simulationcommon_3.py
├── simulate_generate_distance_plots_3.py
├── generate_distance_plots_u.py
├── generate_distance_plots_reaction_transparent facecolor.py
├── histogram_2.py
├── histogram_5_SDP_2575.py
└── simulate_generate_distance_plots_d'min vs total fuel saving with error bars and speed up down_3.py
```

## Related Paper

This repository contains the simulation code for:

**A Simulation Framework with Guaranteed Error Bounds for Safety and Fuel-Efficiency Analysis of Vehicle Platoons**

Yuhao Chen and Ahmet Cetinkaya
