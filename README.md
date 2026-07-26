## File Descriptions of safe psim

### Main Files

| File | Purpose |
|---|---|
| `simulationcommon.py` | Models the linear vehicle-platoon dynamics under sudden braking and communication losses and computes the minimum inter-vehicle distance. |
| `simulationcommon_3.py` | Simulates the continuous velocity-oscillation scenario in which the virtual reference vehicle repeatedly accelerates and decelerates. |
| `simulate_generate_distance_plots_3.py` | Evaluates platoon safety and computational efficiency using the simulation approaches in Theorems 1 and 2. |
| `generate_distance_plots_u.py` | Generates vehicle trajectories and the desired-acceleration, acceleration, and velocity profiles under sudden braking. |
| `generate_distance_plots_reaction_transparent facecolor.py` | Investigates how the time gap \(h\) affects platoon safety under consecutive packet losses. |
| `histogram_2.py` | Evaluates the distribution of the minimum inter-vehicle distance under random packet losses. |
| `histogram_5_SDP_2575.py` | Evaluates the median and interquartile range of the minimum inter-vehicle distance for different packet-loss probabilities. |
| `simulate_generate_distance_plots_d'min vs total fuel saving with error bars and speed up down_3.py` | Investigates the trade-off between minimum inter-vehicle distance and average fuel savings and computes guaranteed simulation-error bounds. |

### `CDG_PF`

| File | Purpose |
|---|---|
| `simulationcommon_3_CDG_PF.py` | Models the platoon dynamics under the Predecessor-Following topology and Constant Distance Gap spacing policy. |
| `simulate_generate_distance_plots_d'min_vs_fuelsaving_CDG_PF.py` | Evaluates the relationship between platoon safety and average fuel savings under the Constant Distance Gap policy. |

### `CTG_PF`

| File | Purpose |
|---|---|
| `simulationcommon_3_CTG_PF.py` | Models the platoon dynamics under the Predecessor-Following topology and Constant Time Gap spacing policy. |
| `simulate_generate_distance_plots_d'min_vs_fuelsaving_CTG_PF.py` | Evaluates the relationship between platoon safety and average fuel savings under the Constant Time Gap policy. |

### `FIG_9_13_14_15`

| File | Purpose |
|---|---|
| `fig_common.py` | Provides the common plotting settings and adaptive time-selection rule used by the simulation scripts. |
| `simulationcommon.py` | Provides the common vehicle-platoon model and numerical simulation functions required by the plotting scripts. |
| `fig1_new.py` | Investigates how packet-loss probability and braking deceleration affect the minimum inter-vehicle distance and the distance traveled by the lead vehicle. |
| `fig2_dmin_vs_taud.py` | Investigates how the vehicle characteristic time constant \(\tau_d\) affects the minimum inter-vehicle distance under random packet losses. |
| `fig3_new.py` | Investigates how the initial velocity affects platoon safety and the remaining distance between the lead vehicle and an obstacle. |
| `fig7_new.py` | Compares the total simulation steps required by the constant-step baseline and the adaptive simulation approaches. |
