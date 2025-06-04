#!/usr/bin/env python3
"""
Unified Model: Flow/Power, Effective Ion Sputtering, Nucleation, and Voronoi Tessellation

This script now incorporates effective stopping power computed from Zn and O cross–section data,
and uses an adaptive domain area determined by the tamed inverse relationship:

    A_sim(P,Q) = (N0/(N(P,Q)+ε))^α * A_target

where A_target is our target simulation area when N(P,Q)=N0 (here, 1e-5 m²). Additionally,
the incident ion energy is now made dependent on power via:
    
    E(P) = E0 + κP

1. Gas–plasma definitions (with sublinear T_gas).
2. Sputtering yield from Zn & O cross sections.
3. Deposition flux and nucleation density.
4. Voronoi simulation with a fixed number of points distributed in a domain whose area is given by the adaptive formula.
5. All outputs are saved in "./voronoi".
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import gamma as gamma_dist
import pandas as pd

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
Q0 = 20.0              # Reference flow rate (sccm)
delta = 1.1            # Empirical exponent

k_B = 1.3807e-23       # Boltzmann constant (J/K)

# Electron temperature parameters (not used further in stopping power)
T_e0 = 2.0             # Baseline electron temperature (eV)
beta = 0.2
gamma_exp = -0.1       

# Sputtering yield parameters
Lambda = 0.05          # Dimensionless constant
E_th = 25.0            # Threshold energy (eV)
E_b = 3.5              # Surface binding energy (eV)

# Surface diffusion parameters
D0 = 1e-7              # Pre-exponential factor (m^2/s)
E_d = 0.5              # Activation energy (eV)
k_eV = 8.617333262145e-5  # Boltzmann constant in eV/K
T_s = 300              # Substrate temperature (K)
D_s = D0 * np.exp(-E_d/(k_eV*T_s))  # Surface diffusion coefficient

# Reference conditions (for scaling mean free path)
P_ref = 100.0          # Reference power (W)
Q_ref = 50.0           # Reference flow (sccm)

# -------------------------------
# Cross Section & Stopping Power Constants
# -------------------------------
# Masses (kg)
M_Ar = 6.6335209e-26  
M_Zn = 65.38 * 1.66054e-27
M_O  = 16.00 * 1.66054e-27

def stopping_factor(M_Ar, M_X):
    """K = 2 M_Ar M_X/(M_Ar+M_X)^2."""
    return 2 * M_Ar * M_X / (M_Ar + M_X)**2

def stopping_power(E, Q0_val, M_Ar, M_X):
    """
    Nuclear stopping power S_X(E) = K * E * Q0(E)
    E in eV, Q0 in m^2.
    """
    K = stopping_factor(M_Ar, M_X)
    return K * E * Q0_val

def effective_stopping_power_from_data(E, df_Zn, df_O):
    """
    Compute effective stopping power S_eff(E) for a Zn:O 1:1 target.
    Interpolate Q0 for Zn and O from the cross–section CSV data.
    """
    Q0_Zn = np.interp(E, df_Zn["Energy (eV)"].values, df_Zn["Q(00)"].values)
    Q0_O  = np.interp(E, df_O["Energy (eV)"].values, df_O["Q(00)"].values)
    S_Zn = stopping_power(E, Q0_Zn, M_Ar, M_Zn)
    S_O  = stopping_power(E, Q0_O, M_Ar, M_O)
    return 0.5 * (S_Zn + S_O)

# -------------------------------
# CSV Data Loading for Cross Sections
# -------------------------------
def load_moment_data(csv_file):
    """
    Load cross–section moment data from CSV.
    Assumes columns: "Energy (Hartee)" and "Q(00)", converts Energy to eV and Q(00) to m^2.
    """
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
# --- Improved Gas Temperature Model using a sublinear power-law ---
def T_gas(P):
    """
    Gas temperature as a sublinear power–law function of input power.
    T_gas = T_gas0 + ξ * P^0.8.
    """
    b = 0.8
    return T_gas0 + xi * (P ** b)

def P_gas(Q):
    """Gas pressure (Pa) from flow rate (sccm)."""
    return P0 * (Q / Q0)**delta

def n_Ar(P, Q):
    """Neutral Argon density (m^-3)."""
    return Q * P_gas(Q) / (k_B * T_gas(P))

def lambda_mfp(P, Q):
    """
    Mean free path (m) computed as:
      λ_mfp = k_B (T_gas0 + ξ * P^0.8) / [Q P0 (Q/Q0)^δ].
    """
    return k_B * (T_gas0 + xi * (P ** 0.8)) / (Q * P0 * (Q / Q0)**delta)

# Compute reference mean free path at (P_ref, Q_ref)
lambda_ref = lambda_mfp(P_ref, Q_ref)

# -------------------------------
# Revised Sputtering Yield and Deposition Flux Using Cross–Section Data
# -------------------------------
def Y_eff_cross(E, P, Q, df_Zn, df_O):
    """
    Effective sputtering yield using effective stopping power from cross–section data.
    Scaled by the mean free path ratio.
    
    Y(E;P,Q) = Λ (E - E_th)/E_b * S_eff(E) * (λ_ref/λ_mfp(P,Q))
    """
    if E <= E_th:
        return 0.0
    S_eff = effective_stopping_power_from_data(E, df_Zn, df_O)
    return Lambda * (E - E_th) / E_b * S_eff * (lambda_ref / lambda_mfp(P, Q))

def deposition_flux_cross(E, P, Q, df_Zn, df_O):
    """
    Deposition flux computed from the effective sputtering yield with an angular projection factor of 2/3.
    """
    return Y_eff_cross(E, P, Q, df_Zn, df_O) * (2/3)

def nucleation_density_cross(E, P, Q, df_Zn, df_O):
    """
    Nucleation density is deposition flux divided by the surface diffusion coefficient.
    """
    return deposition_flux_cross(E, P, Q, df_Zn, df_O) / D_s

# -------------------------------
# Ion Energy Model
# -------------------------------
def ion_energy_from_power(P, E0=75.0, kappa=5):
    """
    Estimate the incident ion energy as a function of power:
        E(P) = E0 + κ * P
    """
    return E0 + kappa * P

# -------------------------------
# Adaptive Domain Area based on Nucleation Density using a Hill-type Function
# -------------------------------
# Our target domain area when N = N0 is A_target
A_target = 0.95e-5*1e-5    # Target domain area (m^2)
N0_area = 1e6      # Reference nucleation density (nuclei/m^2) at which A_sim = A_target
alpha = 0.075      # Taming exponent (tunable)
eps = 1e-10        # Small constant to prevent division by zero

def adaptive_area_from_nucleation(N_val, A_target=A_target, N0=N0_area, alpha=alpha, eps=eps):
    """
    Compute the simulation domain area as:
        A_sim = (N0/(N_val + eps))^α * A_target.
    When N_val = N0, A_sim = A_target.
    """
    return (N0 / (N_val + eps))**alpha * A_target

def simulate_voronoi_tessellation_adaptive(N_val, N_points=1000):
    """
    Fix the number of points to N_points and set the simulation domain area using the adaptive area formula.
    The simulation domain is a square with side length = sqrt(A_sim).
    """
    A_sim = adaptive_area_from_nucleation(N_val)
    side_length = np.sqrt(A_sim)
    points = np.random.rand(N_points, 2) * side_length
    vor = Voronoi(points)
    return vor, points, side_length

def compute_voronoi_cell_areas(vor, side_length):
    """
    Compute areas of bounded Voronoi cells using the shoelace formula.
    """
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

def plot_voronoi_and_gamma_fit(N_intensity, N_points=1000):
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
    ax.plot(points[:, 0], points[:, 1], 'ro', markersize=2)
    ax.set_xlim(0, side_length)
    ax.set_ylim(0, side_length)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    # ax.set_title("Voronoi Tessellation of Nucleation Sites (Adaptive Area)")
    save_fig("Voronoi_Tessellation")
    plt.close(fig)
    
    plt.figure()
    plt.hist(areas, bins=150, density=True, alpha=0.6, label="Voronoi Cell Areas")
    plt.plot(x_vals, gamma_pdf, 'r-', label=f"Gamma Fit: shape={shape:.2f}, scale={scale:.2e}")
    plt.xlabel("Cell Area (m$^2$)")
    plt.ylabel("Probability Density")
    #("Gamma Distribution Fit to Voronoi Cell Areas")
    plt.legend()
    save_fig("Voronoi_Cell_Area_Gamma_Fit")
    plt.close()

def plot_nucleation_density_vs_energy(P, Q, E_min=10, E_max=200, num=100, df_Zn=None, df_O=None):
    """
    Plot nucleation density vs. ion energy for fixed power and flow.
    Uses the cross–section model if df_Zn and df_O are provided.
    """
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
    plt.ylabel("Nucleation Density N (nuclei/m$^2$)")
    #(f"Nucleation Density vs. Ion Energy (P={P} W, Q={Q} sccm)")
    save_fig("Nucleation_Density_vs_Energy")
    plt.close()

def simulate_heatmap_avg_grain_area(power_range, flow_range, E_fixed, N_points=1000, df_Zn=None, df_O=None):
    """
    For a grid of input powers and flow rates, simulate one Voronoi tessellation using
    the nucleation density computed at an ion energy determined dynamically from power.
    The simulation uses the adaptive area function.
    Returns a matrix of average cell areas.
    """
    avg_area_matrix = np.zeros((len(power_range), len(flow_range)))
    for i, P in enumerate(power_range):
        for j, Q in enumerate(flow_range):
            # Compute dynamic ion energy based on power:
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
    #("Average Grain Area vs. Power and Flow Rate")
    plt.colorbar(im, label="Avg. Cell Area (m$^2$)")
    save_fig("Heatmap_Avg_Grain_Area")
    plt.close()

def plot_contour(avg_area_matrix, power_range, flow_range):
    plt.figure()
    Q_grid, P_grid = np.meshgrid(flow_range, power_range)
    cp = plt.contourf(Q_grid, P_grid, avg_area_matrix, cmap='viridis', levels=20)
    plt.xlabel("Flow Rate (sccm)")
    plt.ylabel("Power (W)")
    #("Average Grain Area vs. Power and Flow Rate")
    plt.colorbar(cp, label="Avg. Cell Area (m$^2$)")
    save_fig("Contour_Avg_Grain_Area")
    plt.close()

# -------------------------------
# Main Workflow
# -------------------------------
if __name__ == "__main__":
    # Load cross-section data for Zn and O from CSV files in ./crosssection
    df_Zn = load_moment_data("./crosssection/Zn_cross_section_moments.csv")
    df_O  = load_moment_data("./crosssection/O_cross_section_moments.csv")
    
    # Choose a representative ion energy (for plotting nucleation density vs. energy we still use a range).
    E_ion = 150.0  # This is used only for plotting that curve.
    
    # 1. Plot nucleation density vs. ion energy for fixed conditions at reference power/flow.
    plot_nucleation_density_vs_energy(P_ref, Q_ref, df_Zn=df_Zn, df_O=df_O)
    
    # 2. Simulate a Voronoi tessellation and Gamma fit for the nucleation density computed at E_ion,
    #    at the reference conditions.
    N_intensity_ref = nucleation_density_cross(np.array([E_ion]), P_ref, Q_ref, df_Zn, df_O)[0]
    print(f"Simulated nucleation density at E = {E_ion} eV, P = {P_ref} W, Q = {Q_ref} sccm: {N_intensity_ref:.2e} nuclei/m^2")
    plot_voronoi_and_gamma_fit(N_intensity_ref, N_points=1000)
    
    # 3. Create a contour plot of the average grain (Voronoi cell) area as a function of power and flow rate.
    power_vals = np.linspace(10, 150, 20)   # in Watts
    flow_vals = np.linspace(10, 150, 20)      # in sccm
    avg_area_matrix = simulate_heatmap_avg_grain_area(power_vals, flow_vals,
                                                      E_fixed=E_ion, N_points=1000,
                                                      df_Zn=df_Zn, df_O=df_O)
    plot_contour(avg_area_matrix, power_vals, flow_vals)
    
    print("Processing complete. The following plots have been saved in './voronoi':")
    print(" - Nucleation_Density_vs_Energy")
    print(" - Voronoi_Tessellation")
    print(" - Voronoi_Cell_Area_Gamma_Fit")
    print(" - Contour_Avg_Grain_Area")
