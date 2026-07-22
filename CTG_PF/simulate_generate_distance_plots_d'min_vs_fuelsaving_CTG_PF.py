import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import scipy.linalg as la
import os
import datetime
import types

OUTPUT_ROOT = "Platoon_Analysis_Result"

try:
    import simulationcommon_3_CTG_PF
    from simulationcommon_3_CTG_PF import vehicle_length
except ImportError:
    vehicle_length = 4.7

eta_engine = 0.3
A = 2.3
rho = 1.2
Cd0 = 0.367
Q = 43000.0

try:
    from simulationcommon_3_CTG_PF import common_simulation_parameters, get_value, construct_vector_x0, calculate_error
    from simulate_generate_distance_plots_3 import get_get_nuk_thm7
except ImportError as e:
    print(f"[Error] Failed to import dependencies: {e}")
    print(
        "Please ensure 'simulationcommon_2.py' and 'simulate_generate_distance_plots_2.py' are in the current directory.")
    exit()

lead_a = [0.14725, -0.21819, -0.091455, 0.58344]
lead_b = [0.14192, -0.14349, -0.28946, 0.97713]
lead_sat = 80.0

mid_a = [0.0877, -0.39570, -0.11120, 1.7598]
mid_b = [0.0838, -0.157, -1.5438, 4.1781]
mid_sat = 56.362

trail_a = [0.1498, 1.1719, -3.2785, 2.4494]
trail_b = [0.1380, 2.1565, -5.4882, 3.9392]
trail_sat = 80.639


def get_ratio(d, a, b, sat):
    """ Calculate drag coefficient ratio r = Cd / Cd0 """
    d = np.asarray(d).flatten()
    d = np.maximum(d, 0.01)
    mask = d <= sat
    r = np.ones_like(d)
    if np.any(mask):
        d_in = d[mask]
        num = np.polyval(a, d_in)
        den = np.polyval(b, d_in)
        den[np.abs(den) < 1e-6] = 1e-6
        r[mask] = num / den
    r = np.minimum(r, 1.0)
    return np.clip(r, 0.1, 1.5)


# ==============================================================================
# --- New Functions for Error Bound Calculation (Equation 75/66) ---
# ==============================================================================

def get_poly_derivatives(a, b):
    """ Calculate derivatives of polynomial coefficients """
    # N(d) = a0 d^3 + a1 d^2 + a2 d + a3
    # N'(d) = 3a0 d^2 + 2a1 d + a2
    da = [3 * a[0], 2 * a[1], a[2]]
    db = [3 * b[0], 2 * b[1], b[2]]
    return da, db


def get_ratio_and_derivative(d_val, a, b, sat):
    """ Calculate r(d) and r'(d) for a single distance value """
    # d_val is scalar
    d = max(d_val, 0.01)

    # If out of saturation range, constant r=1, derivative is 0
    if d > sat:
        return 1.0, 0.0

    num = np.polyval(a, d)
    den = np.polyval(b, d)
    if abs(den) < 1e-6: den = 1e-6

    r = num / den

    # Handle clipping boundaries (derivative becomes 0 at clip limits)
    if r < 0.1:
        return 0.1, 0.0
    if r > 1.5:
        return 1.5, 0.0

    # Analytical derivative of rational function: (u/v)' = (u'v - uv') / v^2
    da, db = get_poly_derivatives(a, b)
    d_num = np.polyval(da, d)
    d_den = np.polyval(db, d)

    dr = (d_num * den - num * d_den) / (den ** 2)

    return r, dr


def calculate_error_upper_bound(t_array, pos_list, vel_list, n_vehicles):
    """
    Implementation of Theorem 4, Equation 75 (User referred as Eq 66): Guaranteed Error Bound
    E_k <= dt * K * ( (psi/phi)*S_v + S_d )
    """
    # Parameters from Theorem 4 and context
    alpha = 1.0
    psi = 1.0  # Derived from system matrix properties
    phi = np.sqrt(2)  # Derived from system matrix properties
    K_const = (rho * A * alpha) / (2 * Q * eta_engine)

    total_error = 0.0

    # Iterate over time steps (k=0 to N-2)
    # Using simple rectangular integration sum as per Eq 75 structure
    for k in range(len(t_array) - 1):
        dt = t_array[k + 1] - t_array[k]
        step_error_sum = 0.0

        for i in range(1, n_vehicles + 1):
            # State at t_k
            v_ik = vel_list[i][k]
            p_ik = pos_list[i][k]
            p_front_k = pos_list[i - 1][k]

            # Distance d_i(t_k)
            d_ik = p_front_k - p_ik - vehicle_length

            # Evaluate at boundary d_ik - alpha (Worst case scenario)
            d_eval = d_ik - alpha

            # Select coefficients based on vehicle position
            if i == 1:
                continue
            elif i == 2:
                aa, bb, sat = lead_a, lead_b, lead_sat
            elif i == 3:
                aa, bb, sat = mid_a, mid_b, mid_sat
            else:
                aa, bb, sat = trail_a, trail_b, trail_sat

            # Calculate Cd ratio and its derivative
            r_val, dr_val = get_ratio_and_derivative(d_eval, aa, bb, sat)

            Cd_val = Cd0 * r_val
            dCd_val = Cd0 * dr_val

            # Equation 76: S_v,k = 3 * (v + (psi/phi)*alpha)^2 * |Cd0 - Cd|
            term1 = 3 * (v_ik + (psi / phi) * alpha) ** 2 * abs(Cd0 - Cd_val)

            # Equation 77: S_d,k = v^3 * |d/dd(Cd)|
            term2 = abs(v_ik ** 3 * dCd_val)

            # Sum terms for this vehicle: (psi/phi)*S_v + S_d
            step_error_sum += (psi / phi) * term1 + term2

        # Add to total error: dt * K * Sum(...)
        total_error += dt * K_const * step_error_sum

    return total_error


# ==============================================================================


def detect_parameters_using_functions():
    n_test = 1
    p0_test = 200.0
    v0_test = 0.0
    a0_test = 0.0
    h_test = 0.0

    # 这里调用的是已经被 Patch 过的函数
    x_vec = simulationcommon_3_CTG_PF.construct_vector_x0(n_test, p0_test, v0_test, a0_test, h_test)

    p0_val = x_vec[0, 0]
    p1_val = x_vec[5, 0]

    detected_gap = p0_val - p1_val
    error_val = simulationcommon_3_CTG_PF.calculate_error(1, x_vec, h_test, v0_test)
    detected_r = (p0_val - p1_val) - vehicle_length - error_val

    return detected_r, detected_gap


def run_single_simulation(kp, kd):
    params = common_simulation_parameters()
    params["n"] = 10
    params["tend"] = 300
    params["Nbar"] = 500
    params["k_p"] = kp
    params["k_d"] = kd

    d_min_T, d_min_S, k_final, xks_history, time_history = get_value(params, get_get_nuk_thm7)

    if not hasattr(xks_history, 'todense'):
        xks_history = np.asmatrix(xks_history)

    detected_r, detected_gap = detect_parameters_using_functions()

    print(f"Simulating: kp={kp}, kd={kd}")

    valid_steps = xks_history.shape[1]

    # Handle Time Vector
    if time_history is None or len(time_history) != valid_steps:
        dt_sim = params["T"] / params["Nbar"]
        t = np.arange(valid_steps) * dt_sim
    else:
        real_k = min(k_final + 1, valid_steps)
        t = np.array(time_history[:real_k])
        xks_history = xks_history[:, :real_k]

    # Extract Trajectories
    n_vehicles = params["n"]
    pos_list = []
    vel_list = []

    pos_list.append(np.asarray(xks_history[0, :]).flatten())
    vel_list.append(np.asarray(xks_history[1, :]).flatten())

    for i in range(1, n_vehicles + 1):
        p_idx = 5 + 6 * (i - 1)
        v_idx = p_idx + 1
        pos_list.append(np.asarray(xks_history[p_idx, :]).flatten())
        vel_list.append(np.asarray(xks_history[v_idx, :]).flatten())

    return {
        "params": params,
        "t": t,
        "pos": pos_list,
        "vel": vel_list,
        "d_min": d_min_T,
        "detected_r": detected_r,
        "detected_gap": detected_gap,
        "steps": k_final
    }


def process_and_save_results(sim_data, parent_folder):
    p = sim_data["params"]
    kp = p["k_p"]
    kd = p["k_d"]
    r_val = sim_data["detected_r"]
    initial_gap = sim_data["detected_gap"]
    h = p["h"]

    folder_name = f"kp{kp}_kd{kd}_InitGap{initial_gap:.1f}_r{r_val:.1f}"
    save_dir = os.path.join(parent_folder, folder_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_vehicles = p["n"]
    t = sim_data["t"]
    pos_list = sim_data["pos"]
    vel_list = sim_data["vel"]

    report_lines = []
    report_lines.append(f"{'Veh ID':<10} | {'Mean Ratio (r)':<15} | {'Fuel Saved (g)':<15}")
    report_lines.append("-" * 50)

    total_fuel_all = 0.0
    savings_list = []

    for i in range(1, n_vehicles + 1):
        p_self = pos_list[i]

        # Select Aerodynamic Coefficients
        if i == 1:
            r = np.ones_like(p_self)
            mean_r = 1.0
        elif i == 2:
            p_front = pos_list[i - 1]
            gap_front = p_front - p_self - vehicle_length
            r = get_ratio(gap_front, lead_a, lead_b, lead_sat)
        elif i == 3:
            p_front = pos_list[i - 1]
            gap_front = p_front - p_self - vehicle_length
            r = get_ratio(gap_front, mid_a, mid_b, mid_sat)
        else:
            p_front = pos_list[i - 1]
            gap_front = p_front - p_self - vehicle_length
            r = get_ratio(gap_front, trail_a, trail_b, trail_sat)

        mean_r = np.mean(r)

        # Calculate Fuel
        v = vel_list[i]
        # Normal model:
        p_watts = 0.5 * rho * A * Cd0 * (1 - r) * (v ** 3)
        # acc = np.gradient(v, t)
        dt_array = np.diff(t)
        if len(dt_array) > 0:
            p_watts_min = np.minimum(p_watts[:-1], p_watts[1:])
            energy_J = np.sum(p_watts_min * dt_array)
        else:
            energy_J = 0.0
        # energy_J = np.trapz(p_watts, x=t)
        fuel_g = energy_J / (eta_engine * Q)
        total_fuel_all += fuel_g
        savings_list.append(fuel_g)

        r_str = np.array2string(r, precision=8, separator=' ', suppress_small=True, edgeitems=3, threshold=10)
        report_lines.append(f"Vehicle {i} r array: {r_str}")
        report_lines.append(f"Vehicle {i:<10} | {mean_r:.4f}          | {fuel_g:.4f}")

    v10_mean = np.mean(vel_list[10])

    # --- Calculate Error Upper Bound (Eq 75) ---
    error_bound = calculate_error_upper_bound(t, pos_list, vel_list, n_vehicles)
    report_lines.append("-" * 50)
    report_lines.append(f"Total Fuel Saved (All Veh): {total_fuel_all:.4f} g")
    report_lines.append(f"Theoretical Error Bound (Eq 75): {error_bound:.4f} g")
    report_lines.append(
        f"Error Percentage: {(error_bound / total_fuel_all) * 100:.2f}%" if total_fuel_all > 1e-6 else "Error Percentage: N/A")
    report_lines.append("-" * 50)
    # -------------------------------------------

    report_lines.append(f"--- All plots generated in {save_dir} ---")
    report_lines.append(f"Vehicle 10 Mean Velocity: {v10_mean:.16f}")

    txt_path = os.path.join(save_dir, "Detailed_Data.txt")
    with open(txt_path, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    # Plot PT
    plt.figure(figsize=(10, 6))
    plt.plot(t, pos_list[0], '--', label='$p_0$ (Virtual Leader)')
    for i in range(1, n_vehicles + 1):
        plt.plot(t, pos_list[i], label=f'$p_{{{i}}}$')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title(f'Vehicle Positions (PT) - kp={kp}, kd={kd}')
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Plot_PT.png"), dpi=200)
    plt.close()

    # --- 新增代码: Plot VT (Velocity vs Time) ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, vel_list[0], '--', label='$v_0$ (Virtual Leader)')
    for i in range(1, n_vehicles + 1):
        plt.plot(t, vel_list[i], label=f'$v_{{{i}}}$')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.title(f'Vehicle Velocities (VT) - kp={kp}, kd={kd}')
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Plot_VT.png"), dpi=200)
    plt.close()
    # ------------------------------------------

    # Plot Fuel Savings Bar
    plt.figure(figsize=(6, 3))
    ids = np.arange(1, n_vehicles + 1)
    plt.bar(ids, savings_list, color='#4CAF50', edgecolor='black', width=1.0, alpha=0.8, linewidth=0.7)
    plt.xlabel(r'Vehicle Index $i$')
    plt.ylabel(r'Fuel Saved $M_{\mathrm{save}}$ $[\mathrm{g}]$')
    plt.title(f'Fuel Savings per Vehicle\n(kp={kp}, kd={kd})')
    plt.xticks(ids)
    plt.xlim(0.4, n_vehicles + 0.6)
    plt.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Fuel_Savings_Bar.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # print(f"  -> Results saved to subfolder: {folder_name}")

    simulation_duration = t[-1] if len(t) > 0 else 1.0
    average_fuel_saving = total_fuel_all / simulation_duration

    # Modified Return: Added error_bound and simulation_duration for table generation
    return average_fuel_saving, sim_data["d_min"], r_val, initial_gap, error_bound, simulation_duration


# ==============================================================================
# Main Execution
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    print("--- Starting Analysis (Optimized Visualization) ---")

    kp_values = [0.005, 0.0075, 0.010, 0.0125]  #
    kd_values = [0.15, 0.20, 0.25]

    summary_results = []
    table_data = []

    detected_r_final = 0
    detected_gap_final = 0
    batch_folder_path = ""

    for idx, (kp, kd) in enumerate([(k, d) for k in kp_values for d in kd_values]):
        try:
            sim_data = run_single_simulation(kp, kd)

            if idx == 0:
                detected_r_final = sim_data["detected_r"]
                detected_gap_final = sim_data["detected_gap"]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_folder_name = f"Batch_Analysis_{timestamp}_CTG_PF"
                batch_folder_path = os.path.join(OUTPUT_ROOT, batch_folder_name)
                if not os.path.exists(batch_folder_path):
                    os.makedirs(batch_folder_path)
                print(f"\n[Info] Batch Folder Created: {batch_folder_path}\n")

            # Unpack the new return values: error_bound and duration
            fuel, dmin, _, _, error_total_g, duration = process_and_save_results(sim_data, batch_folder_path)

            # Convert Total Error (g) -> Avg Error Rate (g/s) to match plot units
            error_rate_gps = error_total_g / duration

            print(f"  -> dmin(s): {dmin:.4f}, steps(k): {sim_data['steps']}, xerror: {error_rate_gps:.5f}")

            summary_results.append((kp, kd, fuel, dmin, error_rate_gps))

            # --- Data for Table ---
            # Append to table data (kp, kd, Fuel Rate, Error Rate) - NO PERCENTAGE
            table_data.append([
                f"${kp}$",
                f"${kd}$",
                f"${fuel:.4f}$",
                f"${error_rate_gps:.5f}$"
            ])

        except Exception as e:
            print(f"  -> Simulation Failed (kp={kp}, kd={kd}): {e}")

    if batch_folder_path:
        print("\nGenerating Summary Plot and Tables...")

        plt.figure(figsize=(6, 4))
        plt.gca().set_axisbelow(True)
        markers = {0.005: 'v', 0.0075: '^', 0.010: 'D', 0.0125: 's', 0.25: 'o'}
        colors = {0.15: 'blue', 0.20: 'cyan', 0.25: 'C0', 1.25: 'C1', 1.5: 'C2'}

        for r in summary_results:
            kp, kd, fuel, dmin, fuel_error = r  # CHANGE 2: Unpack fuel_error

            # CHANGE 3: Add error bars
            # yerr=1.0 for d'min +/- 1m
            # xerr=fuel_error for Eq 66
            # plt.errorbar(fuel, dmin,
            #              xerr=fuel_error,
            #              yerr=1.0,
            #              fmt='none',  # Don't draw markers (scatter does that)
            #              ecolor='gray',
            #              alpha=0.6,
            #              capsize=3,
            #              zorder=1)  # Behind scatter

            plt.scatter(fuel, dmin, c=colors[kd], marker=markers[kp], s=120, alpha=0.8, edgecolors='black',
                        label=r"$k_{\mathrm{p}}=$" + f"{kp}, " + r"$k_{\mathrm{d}}=$" + f"{kd}",
                        zorder=2)

        plt.xlabel(r"Average Fuel Saved (All Vehicles) $\frac{1}{t_{\mathrm{end}}} \sum_{i=2}^{n}"
                   r" M_{\mathrm{save},i}$ [g/s]")
        plt.ylabel(r"$d'_{\min}$ [m]")
        plt.grid(True, linestyle='--', alpha=0.6)
        handles, lbls = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), fontsize='small',
                   borderaxespad=0.)
        summary_path = os.path.join(batch_folder_path, "Summary_Fuel_vs_Safety_CTG_PF.pdf")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved to: {summary_path}")
        if table_data:
            col_labels = [
                r"$k_{\mathrm{p}}$",
                r"$k_{\mathrm{d}}$",
                r"Average Fuel Savings [g/s]",
                r"Error Bound [g/s]"
            ]

            fig, ax = plt.subplots(figsize=(8, len(table_data) * 0.5 + 1.5))
            ax.axis('off')

            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='center',
                colColours=['#f2f2f2'] * len(col_labels),
                colWidths=[0.15, 0.15, 0.35, 0.35]
            )

            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.8)
            for (row, col), cell in table.get_celld().items():
                if row == 0: cell.set_text_props(weight='bold')

            table_pdf_path = os.path.join(batch_folder_path, "Error_Analysis_Table_CTG_PF.pdf")
            plt.savefig(table_pdf_path, bbox_inches='tight')
            plt.close()
            print(f"PDF Table saved to: {table_pdf_path}")

    print("Done.")