#!/usr/bin/env python3
"""
Unified Model: Flow/Power, Effective Ion Sputtering, Nucleation, and Voronoi Tessellation

This script incorporates effective stopping power computed from Zn and O cross–section data and uses an adaptive domain
area determined by the tamed inverse relationship:

    A_sim(P,Q) = (N0/(N(P,Q)+ε))^α * A_target

where A_target is our target simulation area when N(P,Q)=N0 (here, 1e-10 m²). Additionally, the incident ion energy 
depends on power via:
    
    E(P) = E0 + κP

Furthermore, the evolution of the grain size distribution is simulated using a stochastic mean-field 
approximation of the Herring–Mullins mechanism:
    dA_i/dt = γ κ_i + √(2D_A) η_i(t),
with curvature approximated as κ_i ≈ 2π/(perimeter of cell).

The code outputs:
  • Standard unified model plots (nucleation density, Voronoi tessellation with inset, Gamma fit, contour heatmap).
  • A 2×2 grid snapshot of the Voronoi tessellation at selected times with insets.
  • Trajectories of selected individual grain areas over time.
All outputs are saved in the "./voronoi" directory.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import gamma as gamma_dist
from scipy.stats import gaussian_kde
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# -------------------------------
# Global Plot Settings
# -------------------------------
try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    plt.style.use('default')

mpl_params = {
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.usetex': True,
    'font.size': 14,
    'font.family': 'serif',
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.figsize': (8, 6)
}
plt.rcParams.update(mpl_params)

# Create output directory
output_dir = "./voronoi"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def save_fig(filename):
    plt.savefig(os.path.join(output_dir, f"{filename}.jpeg"), format='jpeg', bbox_inches='tight')

# -------------------------------
# Physical Constants and Parameters
# -------------------------------
# Gas/Plasma parameters
T_gas0 = 300.0         # Baseline gas temperature (K)
xi = 2                 # Heating coefficient (K/W)
P0 = 0.1               # Reference pressure (Pa)
Q0 = 20.0              # Reference flow (sccm)
delta = 1.1            # Empirical exponent
k_B = 1.3807e-23       # Boltzmann constant (J/K)

# Sputtering yield parameters
Lambda = 0.05          # Dimensionless constant
E_th = 25.0            # Threshold energy (eV)
E_b = 3.5              # Surface binding energy (eV)

# Surface diffusion parameters
D0 = 1e-7              # Pre-exponential factor (m²/s)
E_d = 0.5              # Activation energy (eV)
k_eV = 8.617333262145e-5  # Boltzmann constant in eV/K
T_s = 300              # Substrate temperature (K)
D_s = D0 * np.exp(-E_d/(k_eV*T_s))

# Reference conditions (for scaling mean free path)
P_ref = 100.0          # Reference power (W)
Q_ref = 50.0           # Reference flow (sccm)

# -------------------------------
# Cross Section & Stopping Power Constants
# -------------------------------
M_Ar = 6.6335209e-26  
M_Zn = 65.38 * 1.66054e-27
M_O  = 16.00 * 1.66054e-27

def stopping_factor(M_Ar, M_X):
    return 2 * M_Ar * M_X / (M_Ar + M_X)**2

def stopping_power(E, Q0_val, M_Ar, M_X):
    return stopping_factor(M_Ar, M_X) * E * Q0_val

def effective_stopping_power_from_data(E, df_Zn, df_O):
    Q0_Zn = np.interp(E, df_Zn["Energy (eV)"].values, df_Zn["Q(00)"].values)
    Q0_O  = np.interp(E, df_O["Energy (eV)"].values, df_O["Q(00)"].values)
    S_Zn = stopping_power(E, Q0_Zn, M_Ar, M_Zn)
    S_O  = stopping_power(E, Q0_O, M_Ar, M_O)
    return 0.5 * (S_Zn + S_O)

# -------------------------------
# CSV Data Loading for Cross Sections
# -------------------------------
def load_moment_data(csv_file):
    hartree_to_eV = 27.211386
    bohr2_to_m2 = (5.29177210903e-11)**2
    df = pd.read_csv(csv_file)
    df["Energy (eV)"] = df["Energy (Hartee)"] * hartree_to_eV
    for col in df.columns:
        if col.startswith("Q("):
            df[col] = df[col] * bohr2_to_m2
    return df

# -------------------------------
# Gas/Plasma Functions
# -------------------------------
def T_gas(P):
    b = 0.8
    return T_gas0 + xi * (P ** b)

def P_gas(Q):
    return P0 * (Q / Q0)**delta

def n_Ar(P, Q):
    return Q * P_gas(Q) / (k_B * T_gas(P))

def lambda_mfp(P, Q):
    return k_B * (T_gas0 + xi * (P ** 0.8)) / (Q * P0 * (Q / Q0)**delta)

lambda_ref = lambda_mfp(P_ref, Q_ref)

# -------------------------------
# Revised Sputtering Yield and Deposition Flux Using Cross–Section Data
# -------------------------------
def Y_eff_cross(E, P, Q, df_Zn, df_O):
    if E <= E_th:
        return 0.0
    S_eff = effective_stopping_power_from_data(E, df_Zn, df_O)
    return Lambda * (E - E_th) / E_b * S_eff * (lambda_ref / lambda_mfp(P, Q))

def deposition_flux_cross(E, P, Q, df_Zn, df_O):
    return Y_eff_cross(E, P, Q, df_Zn, df_O) * (2/3)

def nucleation_density_cross(E, P, Q, df_Zn, df_O):
    return deposition_flux_cross(E, P, Q, df_Zn, df_O) / D_s

# -------------------------------
# Ion Energy Model (Dynamic)
# -------------------------------
def ion_energy_from_power(P, E0=75.0, kappa=5):
    return E0 + kappa * P

# -------------------------------
# Adaptive Domain Area based on Nucleation Density (Tamed Inverse)
# -------------------------------
# Target simulation area when N = N0 is A_target.
A_target = .95e-10       # Target simulation area (m²)
N0_area = 1e6          # Reference nucleation density (nuclei/m²) at which A_sim = A_target
alpha = 0.075          # Taming exponent (tunable)
eps = 1e-10

def adaptive_area_from_nucleation(N_val, A_target=A_target, N0=N0_area, alpha=alpha, eps=eps):
    return (N0 / (N_val + eps))**alpha * A_target

def simulate_voronoi_tessellation_adaptive(N_val, N_points=1000):
    A_sim = adaptive_area_from_nucleation(N_val)
    side_length = np.sqrt(A_sim)
    points = np.random.rand(N_points, 2) * side_length
    vor = Voronoi(points)
    return vor, points, side_length

def compute_voronoi_cell_areas(vor, side_length):
    areas = []
    for region_index in vor.point_region:
        vertices = vor.regions[region_index]
        if -1 in vertices or len(vertices) == 0:
            continue
        polygon = vor.vertices[vertices]
        if np.all((polygon >= 0) & (polygon <= side_length)):
            x = polygon[:, 0]
            y = polygon[:, 1]
            area_cell = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            areas.append(area_cell)
    return np.array(areas)

# -------------------------------
# Voronoi Plot with Inset (for all Voronoi plots)
# -------------------------------
def plot_voronoi_and_gamma_fit(N_intensity, N_points=1000, zoom=0.2):
    vor, points, side_length = simulate_voronoi_tessellation_adaptive(N_intensity, N_points)
    areas = compute_voronoi_cell_areas(vor, side_length)
    if len(areas) == 0:
        print("No bounded Voronoi cells found. Check your parameters.")
        return
    shape, loc, scale = gamma_dist.fit(areas, floc=0)
    x_vals = np.linspace(min(areas), max(areas), 100)
    gamma_pdf = gamma_dist.pdf(x_vals, shape, loc=loc, scale=scale)
    
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', point_size=2)
    ax.plot(points[:, 0], points[:, 1], 'bo', markersize=1)
    ax.set_xlim(0, side_length)
    ax.set_ylim(0, side_length)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # ax.set_title("Voronoi Tessellation (Adaptive Area)")
    
    # Add an inset zoom to the main Voronoi plot
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
    voronoi_plot_2d(vor, ax=axins, show_vertices=False, line_colors='black', point_size=2, point_colors='blue')
    center = side_length / 2.0
    zoom_side = side_length * zoom
    axins.set_xlim(center - zoom_side/2, center + zoom_side/2)
    axins.set_ylim(center - zoom_side/2, center + zoom_side/2)
    axins.set_xticks([])
    axins.set_yticks([])
    
    save_fig("Voronoi_Tessellation")
    plt.close(fig)
    
    plt.figure()
    plt.hist(areas, bins=150, density=True, alpha=0.6, label="Voronoi Cell Areas")
    plt.plot(x_vals, gamma_pdf, 'r-', label=f"Gamma Fit: shape={shape:.2f}, scale={scale:.2e}")
    plt.xlabel("Cell Area (m²)")
    plt.ylabel("Probability Density")
    # plt.title("Gamma Distribution Fit to Voronoi Cell Areas")
    plt.legend()
    save_fig("Voronoi_Cell_Area_Gamma_Fit")
    plt.close()

# -------------------------------
# Plot Nucleation Density vs. Ion Energy
# -------------------------------
def plot_nucleation_density_vs_energy(P, Q, E_min=10, E_max=200, num=100, df_Zn=None, df_O=None):
    E_vals = np.linspace(E_min, E_max, num)
    N_vals = []
    for E in E_vals:
        if df_Zn is not None and df_O is not None:
            N_vals.append(nucleation_density_cross(E, P, Q, df_Zn, df_O))
        else:
            N_vals.append(0.0)
    plt.figure()
    plt.plot(E_vals, N_vals, linestyle='-')
    plt.xlabel("Ion Energy (eV)")
    plt.ylabel("Nucleation Density N (nuclei/m²)")
    # plt.title(f"Nucleation Density vs. Ion Energy (P={P} W, Q={Q} sccm)")
    save_fig("Nucleation_Density_vs_Energy")
    plt.close()

# -------------------------------
# Stochastic Herring–Mullins Grain Evolution Simulation
# -------------------------------
gamma_line = 1e-26    # Extremely low line tension coefficient [m²/s]
D_A = 8e-29           # Tiny diffusion coefficient [m⁴/s]
dt = 0.1              # Time step [s]
total_time = 100.0    # Total simulation time [s]
time_steps = int(total_time / dt)

def compute_area_and_perimeter(vor, side_length):
    areas, perimeters = [], []
    for region_index in vor.point_region:
        vertices = vor.regions[region_index]
        if -1 in vertices or len(vertices) < 3:
            continue
        polygon = vor.vertices[vertices]
        if np.any(polygon < 0) or np.any(polygon > side_length):
            continue
        x, y = polygon[:, 0], polygon[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        perim = np.sum(np.sqrt(np.diff(np.append(x, x[0]))**2 + np.diff(np.append(y, y[0]))**2))
        areas.append(area)
        perimeters.append(perim)
    return np.array(areas), np.array(perimeters)

def curvature(perimeters):
    return 2 * np.pi / perimeters

def stochastic_herring_mullins(vor, side_length):
    areas, perimeters = compute_area_and_perimeter(vor, side_length)
    N_grains = len(areas)
    area_history = np.zeros((time_steps, N_grains))
    mean_diameter = np.zeros(time_steps)
    
    area_history[0, :] = areas.copy()
    mean_diameter[0] = 2 * np.sqrt(np.mean(areas) / np.pi)
    
    for t in range(1, time_steps):
        kappa = curvature(perimeters)
        noise = np.random.normal(0, np.sqrt(2 * D_A * dt), N_grains)
        dA = gamma_line * kappa * dt + noise
        areas += dA
        areas = np.clip(areas, 1e-20, None)
        diameters = 2 * np.sqrt(areas / np.pi)
        perimeters = np.pi * diameters  # Approximate as circle perimeter.
        mean_diameter[t] = 2 * np.sqrt(np.mean(areas) / np.pi)
        area_history[t, :] = areas.copy()
    return area_history, mean_diameter

def plot_grain_size_distribution(area_history, times_to_plot, dt):
    plt.figure(figsize=(12,8))
    for t in times_to_plot:
        idx = int(t/dt)
        if idx >= area_history.shape[0]:
            idx = area_history.shape[0] - 1
        areas = area_history[idx, :]
        diameters = 2 * np.sqrt(areas / np.pi)
        kde = gaussian_kde(diameters)
        x_grid = np.linspace(min(diameters), max(diameters), 100)
        plt.plot(x_grid, kde(x_grid), label=f"t = {t:.1f} s")
    plt.xlabel("Grain Diameter [m]")
    plt.ylabel("Probability Density")
    # plt.title("Evolution of Grain Size Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "GrainSizeDistribution.jpeg"))
    plt.close()

def plot_mean_grain_diameter(mean_diameter, dt):
    time = np.arange(len(mean_diameter)) * dt
    plt.figure(figsize=(10,6))
    plt.plot(time, mean_diameter)
    plt.xlabel("Time [s]")
    plt.ylabel("Mean Grain Diameter [m]")
    # plt.title("Mean Grain Diameter Evolution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "MeanGrainDiameter.jpeg"))
    plt.close()

# -------------------------------
# Additional Plots: Voronoi Snapshots with Inset and Grain Trajectories
# -------------------------------
def plot_voronoi_snapshots_with_inset(vor, side_length, snapshot_indices, dt, zoom=0.1):
    """
    Plot a 2x2 grid of Voronoi tessellation snapshots at given time indices (snapshot_indices).
    Each subplot has an inset showing a zoomed view of the center region.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    
    zoom_side = side_length * zoom
    center = side_length / 2.0
    inset_xlim = (center - zoom_side/2, center + zoom_side/2)
    inset_ylim = (center - zoom_side/2, center + zoom_side/2)
    
    for ax, idx in zip(axs, snapshot_indices):
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', point_size=2)
        ax.set_xlim(0, side_length)
        ax.set_ylim(0, side_length)
        # ax.set_title(f"Snapshot t = {idx*dt:.1f} s")
        axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
        voronoi_plot_2d(vor, ax=axins, show_vertices=False, line_colors='black', point_size=2)
        axins.set_xlim(inset_xlim)
        axins.set_ylim(inset_ylim)
        axins.set_xticks([])
        axins.set_yticks([])
    plt.savefig(os.path.join(output_dir, "Voronoi_Snapshots_With_Inset.jpeg"), format='jpeg', bbox_inches='tight')
    plt.close()

def plot_grain_trajectories(area_history, dt, grain_indices):
    time = np.arange(area_history.shape[0]) * dt
    plt.figure(figsize=(10,6))
    for i in grain_indices:
        plt.plot(time, area_history[:, i], label=f"Grain {i}")
    plt.xlabel("Time [s]")
    plt.ylabel("Grain Area [m²]")
    # plt.title("Trajectories of Selected Grain Areas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Grain_Trajectories.jpeg"))
    plt.close()

# -------------------------------
# Unified Model Heatmap Simulation
# -------------------------------
def simulate_heatmap_avg_grain_area(power_range, flow_range, E_fixed, N_points=1000, df_Zn=None, df_O=None):
    avg_area_matrix = np.zeros((len(power_range), len(flow_range)))
    for i, P in enumerate(power_range):
        for j, Q in enumerate(flow_range):
            E_dynamic = ion_energy_from_power(P)
            N_val = nucleation_density_cross(np.array([E_dynamic]), P, Q, df_Zn, df_O)[0]
            vor, points, side_length = simulate_voronoi_tessellation_adaptive(N_val, N_points)
            areas = compute_voronoi_cell_areas(vor, side_length)
            if len(areas) > 0:
                avg_area_matrix[i, j] = np.mean(areas)
            else:
                avg_area_matrix[i, j] = np.nan
    return avg_area_matrix

def plot_heatmap(avg_area_matrix, power_range, flow_range):
    plt.figure()
    extent = [flow_range[0], flow_range[-1], power_range[0], power_range[-1]]
    im = plt.imshow(avg_area_matrix, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.xlabel("Flow Rate (sccm)")
    plt.ylabel("Power (W)")
    # plt.title("Average Grain Area vs. Power and Flow Rate")
    plt.colorbar(im, label="Avg. Cell Area (m²)")
    save_fig("Heatmap_Avg_Grain_Area")
    plt.close()

def plot_contour(avg_area_matrix, power_range, flow_range):
    plt.figure()
    Q_grid, P_grid = np.meshgrid(flow_range, power_range)
    cp = plt.contourf(Q_grid, P_grid, avg_area_matrix, cmap='viridis', levels=20)
    plt.xlabel("Flow Rate (sccm)")
    plt.ylabel("Power (W)")
    # plt.title("Contour Plot of Average Grain Area vs. Power and Flow Rate")
    plt.colorbar(cp, label="Avg. Cell Area (m²)")
    save_fig("Contour_Avg_Grain_Area")
    plt.close()

# -------------------------------
# Main Unified Model Workflow
# -------------------------------
if __name__ == "__main__":
    # Load cross-section data for Zn and O from CSV files in ./crosssection
    df_Zn = load_moment_data("./crosssection/Zn_cross_section_moments.csv")
    df_O  = load_moment_data("./crosssection/O_cross_section_moments.csv")
    
    # Choose a representative ion energy for plotting nucleation density vs. energy.
    E_ion = 150.0
    
    # 1. Plot nucleation density vs. ion energy for fixed conditions at reference power/flow.
    plot_nucleation_density_vs_energy(P_ref, Q_ref, df_Zn=df_Zn, df_O=df_O)
    
    # 2. Simulate a Voronoi tessellation and Gamma fit for the nucleation density computed at E_ion (reference conditions).
    N_intensity_ref = nucleation_density_cross(np.array([E_ion]), P_ref, Q_ref, df_Zn, df_O)[0]
    print(f"Simulated nucleation density at E = {E_ion} eV, P = {P_ref} W, Q = {Q_ref} sccm: {N_intensity_ref:.2e} nuclei/m²")
    plot_voronoi_and_gamma_fit(N_intensity_ref, N_points=1000, zoom=0.1)
    
    # 3. Create a contour plot of the average grain (Voronoi cell) area as a function of power and flow rate.
    power_vals = np.linspace(10, 150, 20)   # in Watts
    flow_vals = np.linspace(10, 150, 20)      # in sccm
    avg_area_matrix = simulate_heatmap_avg_grain_area(power_vals, flow_vals,
                                                      E_fixed=E_ion, N_points=1000,
                                                      df_Zn=df_Zn, df_O=df_O)
    plot_contour(avg_area_matrix, power_vals, flow_vals)
    
    print("Unified model processing complete. The following plots have been saved in './voronoi':")
    print(" - Nucleation_Density_vs_Energy")
    print(" - Voronoi_Tessellation (with inset)")
    print(" - Voronoi_Cell_Area_Gamma_Fit")
    print(" - Contour_Avg_Grain_Area")
    
    # -------------------------------
    # Run Herring–Mullins Grain Evolution Simulation
    # -------------------------------
    vor_init, pts_init, side_length_init = simulate_voronoi_tessellation_adaptive(N_intensity_ref, N_points=1000)
    area_history, mean_diameter = stochastic_herring_mullins(vor_init, side_length_init)
    
    # Plot snapshots in a 2x2 grid with inset at selected time indices
    snapshot_indices = [0, int(time_steps/3), int(2*time_steps/3), time_steps-1]
    plot_voronoi_snapshots_with_inset(vor_init, side_length_init, snapshot_indices, dt, zoom=0.1)
    
    # Plot trajectories of selected grains (e.g., first 5 grains)
    selected_grains = np.arange(5)
    plot_grain_trajectories(area_history, dt, selected_grains)
    
    # Plot the evolution of the mean grain diameter over time
    plot_mean_grain_diameter(mean_diameter, dt)
    
    # Plot the grain size distribution evolution at selected times
    times_to_plot = [0, total_time*0.25, total_time*0.5, total_time - dt]
    plot_grain_size_distribution(area_history, times_to_plot, dt)
    
    print("Herring–Mullins simulation complete. Additional snapshots and grain trajectory plots are saved in './voronoi'.")
