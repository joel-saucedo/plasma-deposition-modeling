import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from numpy import trapz
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===============================
# Global Plot Settings
# ===============================
try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    plt.style.use('default')

mpl.rcParams['figure.dpi'] = 300          
mpl.rcParams['savefig.dpi'] = 300           
mpl.rcParams['text.usetex'] = True          
mpl.rcParams['font.size'] = 14              
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.figsize'] = (8, 6)

if not os.path.exists('./results'):
    os.makedirs('./results')

def save_fig(filename):
    plt.savefig(f"./results/{filename}.jpeg", format='jpeg', bbox_inches='tight')

# ===============================
# Physical Constants
# ===============================
m_e = 9.10938356e-31          # kg
k_J = 1.3807e-23              # J/K
k_eV = 8.617333262145e-5      # eV/K
h = 6.626e-34                 # J·s
eV_to_J = 1.602176634e-19     # J/eV

# ===============================
# Updated Gas Parameters (Magnetron Sputtering)
# ===============================
# Baseline gas temperature (room temperature)
T_gas0 = 300.0              # K
# Heating coefficient (e.g., 0.5 K/W)
xi = 2                    # K/W
# Reference pressure and flow rate
P0 = 0.1                    # Pa (low-pressure sputtering)
Q0 = 20.0                   # sccm
delta = 1.0                 # Empirical exponent

# ===============================
# Function Definitions (same as before)
# ===============================
def maxwell_boltzmann_eedf(e, T_e):
    return (1.0 / T_e) * np.exp(-e / T_e)

def compute_total_cross_section(diff_csv_file, energy_target, tol=1e-6):
    data = pd.read_csv(diff_csv_file)
    mask = np.isclose(data["Energy (eV)"].values, energy_target, atol=tol)
    angles_deg = data["Angle (deg)"].values[mask]
    diff_cross = data["Differential cross section (m2/sr)"].values[mask]
    angles_rad = np.deg2rad(angles_deg)
    integral = trapz(diff_cross * np.sin(angles_rad), angles_rad)
    return 2 * np.pi * integral

def compute_mean_free_path(P, Q, diff_csv_file, energy_target, P0, Q0, delta, T_gas0, xi):
    T_gas = T_gas0 + xi * P ** 0.8
    P_gas = P0 * (Q / Q0)**delta
    n_Ar = (Q * P_gas) / (k_J * T_gas)
    sigma_tot = compute_total_cross_section(diff_csv_file, energy_target)
    return 1 / (n_Ar * sigma_tot)

# For electron temperature we use:
# T_e = T_e0 * P^β * Q^γ, with T_e0 = 2 eV.
def compute_electron_temperature(P, Q, T_e0, beta, gamma):
    return T_e0 * (P ** beta) * (Q ** gamma)

# (Other functions like compute_excitation_rate, etc., remain as defined previously.)
# -------------------------------
def compute_excitation_rate(n_e, n_Ar, cross_section_file, T_e, P, Q, lambda_P=0.005, lambda_Q=0.002):
    data = pd.read_csv(cross_section_file)
    energy = data["Energy (eV)"].values
    cross_section = data["Cross Section (m^2)"].values
    sigma_interp = interp1d(energy, cross_section, bounds_error=False, fill_value=0.0)
    def integrand(e):
        e_J = e * eV_to_J
        v_e = np.sqrt(2 * e_J / m_e)
        return sigma_interp(e) * v_e * maxwell_boltzmann_eedf(e, T_e)
    integral_value, _ = quad(integrand, 0, np.inf)
    Q_effect = 1 / (1 + lambda_P * P + lambda_Q * Q)
    return n_e * n_Ar * integral_value * Q_effect

def compute_densities(P, Q, C_P, C_Q, alpha, lambda_P_density=0.02):
    n_e = C_P * (P ** alpha) / (1 + lambda_P_density * P)
    n_Ar = C_Q * Q
    return n_e, n_Ar

def compute_optical_emission(R_exc):
    return R_exc

def compute_effective_energy(cross_section_file, threshold, T_e):
    data = pd.read_csv(cross_section_file)
    energy = data["Energy (eV)"].values
    sigma = data["Cross Section (m^2)"].values
    sigma_interp = interp1d(energy, sigma, bounds_error=False, fill_value=0.0)
    def num_integrand(e):
        e_J = e * eV_to_J
        v_e = np.sqrt(2 * e_J / m_e)
        return e * sigma_interp(e) * v_e * np.exp(-e / T_e)
    def den_integrand(e):
        e_J = e * eV_to_J
        v_e = np.sqrt(2 * e_J / m_e)
        return sigma_interp(e) * v_e * np.exp(-e / T_e)
    num, _ = quad(num_integrand, threshold, np.inf)
    den, _ = quad(den_integrand, threshold, np.inf)
    return num / den if den != 0 else 0

def compute_incident_ion_energy(I750, I751, E750, E751, c):
    if (I750 + I751) == 0:
        return 0
    return c * (I750 * E750 + I751 * E751) / (I750 + I751)

def compute_sputtering_yield(Ei, Eth, Eb, Lambda):
    return Lambda * (Ei - Eth) / Eb

# ===============================
# Parameter Ranges for New Plots
# ===============================
# Use denser grids for heatmaps and smooth curves.
power_dense = np.linspace(10, 150, 50)    # 10 to 150 W
flow_dense = np.linspace(10, 150, 50)       # 10 to 150 sccm

# Fixed scaling exponents and baseline electron temperature (in eV)
T_e0 = 2.0
beta = 0.2
gamma = -0.1

# ===============================
# 1. Electron Temperature vs. Power for Different Flow Rates
# ===============================
fixed_flow_rates = [20, 50, 80, 120]  # sccm
plt.figure()
for Q in fixed_flow_rates:
    T_e_vals = compute_electron_temperature(power_dense, Q, T_e0, beta, gamma)
    plt.plot(power_dense, T_e_vals,  label=f"Q = {Q} sccm",  marker='')
plt.xlabel("Input Power (W)")
plt.ylabel(r"Electron Temperature $T_e$ (eV)")
#plt.title("Electron Temperature vs. Power for Different Flow Rates")
plt.legend()
save_fig("Te_vs_Power_Different_Flow")
plt.close()

# ===============================
# 2. Maxwell–Boltzmann Distribution vs. Energy for Different Powers
# ===============================
# Fix flow rate, vary power
fixed_Q_for_MB = 50  # sccm
powers_MB = [20, 60, 100, 140]  # W
energy_range = np.linspace(0, 20, 500)  # eV

plt.figure()
for P in powers_MB:
    T_e = compute_electron_temperature(P, fixed_Q_for_MB, T_e0, beta, gamma)
    f_e_vals = maxwell_boltzmann_eedf(energy_range, T_e)
    plt.plot(energy_range, f_e_vals, label=f"P = {P} W\n($T_e$ = {T_e:.2f} eV)")
plt.xlabel(r"Electron Energy $E$ (eV)")
plt.ylabel(r"$f_e(E)$")
#plt.title("Maxwell–Boltzmann Distribution vs. Energy\n(for Different Powers, Q = 50 sccm)")
plt.legend()
save_fig("MB_vs_Energy_Different_Power")
plt.close()

# ===============================
# 3. Maxwell–Boltzmann Distribution vs. Energy for Different Flow Rates
# ===============================
# Fix power, vary flow rate
fixed_P_for_MB = 100  # W
flow_rates_MB = [20, 50, 80, 120]  # sccm

plt.figure()
for Q in flow_rates_MB:
    T_e = compute_electron_temperature(fixed_P_for_MB, Q, T_e0, beta, gamma)
    f_e_vals = maxwell_boltzmann_eedf(energy_range, T_e)
    plt.plot(energy_range, f_e_vals, label=f"Q = {Q} sccm\n($T_e$ = {T_e:.2f} eV)")
plt.xlabel(r"Electron Energy $E$ (eV)")
plt.ylabel(r"$f_e(E)$")
#plt.title("Maxwell–Boltzmann Distribution vs. Energy\n(for Different Flow Rates, P = 100 W)")
plt.legend()
save_fig("MB_vs_Energy_Different_Flow")
plt.close()

# ===============================
# 4. Heatmap of Maxwell–Boltzmann Distribution vs. Power and Flow Rate
# ===============================
# Choose a representative electron energy, e.g., E = 2 eV.
E_fixed = 2.0  # eV
# Create meshgrid for power and flow rate.
P_grid, Q_grid = np.meshgrid(power_dense, flow_dense, indexing='ij')
T_e_grid = compute_electron_temperature(P_grid, Q_grid, T_e0, beta, gamma)
MB_grid = maxwell_boltzmann_eedf(E_fixed, T_e_grid)

plt.figure()
plt.imshow(MB_grid, extent=[flow_dense[0], flow_dense[-1], power_dense[0], power_dense[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.xlabel("Flow Rate (sccm)")
plt.ylabel("Power (W)")
#plt.title(r"Heatmap of $f_e(E=2\,eV)$ vs. Power and Flow Rate")
plt.colorbar(label=r"$f_e(2\,eV)$")
save_fig("Heatmap_MB_vs_P_and_Q")
plt.close()

# ===============================
# 5. Optical Emission Intensities vs. Power and Flow Rate
# ===============================
# For demonstration, compute I750 and I751 using the excitation rate.
# We use a dense grid for power and flow rate.
power_dense_2D = np.linspace(10, 150, 50)
flow_dense_2D = np.linspace(10, 150, 50)
P2D, Q2D = np.meshgrid(power_dense_2D, flow_dense_2D, indexing='ij')

# Set constant parameters for excitation calculations.
# (These values are arbitrary examples; replace with your experimental values if available.)
C_P = 1e16
C_Q = 1e20
alpha_exp = 0.05
beta_exp = beta
gamma_exp = gamma
lambda_P_density = 0.02
cs_file_750 = './crosssection/E_Ar_E_Ar2P1.csv'
cs_file_751 = './crosssection/E_Ar_E_Ar2P5.csv'

I750_grid = np.zeros_like(P2D)
I751_grid = np.zeros_like(P2D)
for i in range(P2D.shape[0]):
    for j in range(P2D.shape[1]):
        P_val = P2D[i, j]
        Q_val = Q2D[i, j]
        T_e = T_e0 * (P_val ** beta_exp) * (Q_val ** gamma_exp)
        n_e, n_Ar = (C_P * (P_val ** alpha_exp) / (1 + lambda_P_density * P_val), C_Q * Q_val)
        I750_grid[i, j] = compute_optical_emission(compute_excitation_rate(n_e, n_Ar, cs_file_750, T_e, P_val, Q_val))
        I751_grid[i, j] = compute_optical_emission(compute_excitation_rate(n_e, n_Ar, cs_file_751, T_e, P_val, Q_val))

# Line plots vs. power for fixed flow (e.g., Q = 10 sccm)
fixed_flow_line = 10.0
I750_line_power = []
I751_line_power = []
for P_val in power_dense:
    T_e = T_e0 * (P_val ** beta_exp) * (fixed_flow_line ** gamma_exp)
    n_e, n_Ar = (C_P * (P_val ** alpha_exp) / (1 + lambda_P_density * P_val), C_Q * fixed_flow_line)
    I750_line_power.append(compute_optical_emission(compute_excitation_rate(n_e, n_Ar, cs_file_750, T_e, P_val, fixed_flow_line)))
    I751_line_power.append(compute_optical_emission(compute_excitation_rate(n_e, n_Ar, cs_file_751, T_e, P_val, fixed_flow_line)))

plt.figure()
plt.plot(power_dense, I750_line_power,  label=r"$I_{750}$")
plt.plot(power_dense, I751_line_power,  label=r"$I_{751}$")
plt.xlabel("Power (W)")
plt.ylabel("Optical Intensity (arb. units)")
#plt.title("Optical Emission vs. Power (Q = 10 sccm)")
plt.legend()
save_fig("OpticalEmission_vs_Power")
plt.close()

# Line plots vs. flow for fixed power (e.g., P = 10 W)
fixed_power_line = 10.0
I750_line_flow = []
I751_line_flow = []
for Q_val in flow_dense:
    T_e = T_e0 * (fixed_power_line ** beta_exp) * (Q_val ** gamma_exp)
    n_e, n_Ar = (C_P * (fixed_power_line ** alpha_exp) / (1 + lambda_P_density * fixed_power_line), C_Q * Q_val)
    I750_line_flow.append(compute_optical_emission(compute_excitation_rate(n_e, n_Ar, cs_file_750, T_e, fixed_power_line, Q_val)))
    I751_line_flow.append(compute_optical_emission(compute_excitation_rate(n_e, n_Ar, cs_file_751, T_e, fixed_power_line, Q_val)))

plt.figure()
plt.plot(flow_dense, I750_line_flow,  label=r"$I_{750}$")
plt.plot(flow_dense, I751_line_flow,  label=r"$I_{751}$")
plt.xlabel("Flow Rate (sccm)")
plt.ylabel("Optical Intensity (arb. units)")
#plt.title("Optical Emission vs. Flow Rate (P = 10 W)")
plt.legend()
save_fig("OpticalEmission_vs_Flow")
plt.close()

# ===============================
# 6. Heatmap of Electron Velocity vs. Power and Flow Rate
# ===============================
# Electron velocity: v_e = sqrt(2*T_e/m_e), T_e in eV (note: here we assume eV units)
v_e_grid = np.sqrt(2 * (T_e0 * (P2D ** beta_exp) * (Q2D ** gamma_exp)) / m_e)

plt.figure()
plt.imshow(v_e_grid, extent=[flow_dense_2D[0], flow_dense_2D[-1], power_dense_2D[0], power_dense_2D[-1]],
           origin='lower', aspect='auto', cmap='plasma')
plt.xlabel("Flow Rate (sccm)")
plt.ylabel("Power (W)")
#plt.title("Heatmap of Electron Velocity vs. Power and Flow Rate")
plt.colorbar(label=r"$v_e$ (m/s)")
save_fig("Heatmap_Electron_Velocity")
plt.close()

# ===============================
# 7. Heatmap of Incident Ion Energy vs. Power and Flow Rate
# ===============================
# For demonstration, compute a dense grid of incident ion energy.
# We use I750 and I751 as computed in the 2D grid above, and assume a constant scaling factor c_const.
c_const = 1.5
# For effective energies, we assume thresholds as before:
threshold_750 = 13.273
threshold_751 = 13.48
Ei_grid = np.zeros_like(P2D)
for i in range(P2D.shape[0]):
    for j in range(P2D.shape[1]):
        P_val = P2D[i, j]
        Q_val = Q2D[i, j]
        T_e = T_e0 * (P_val ** beta_exp) * (Q_val ** gamma_exp)
        # Compute effective energies from cross-section files
        E_bar_750 = compute_effective_energy(cs_file_750, threshold_750, T_e)
        E_bar_751 = compute_effective_energy(cs_file_751, threshold_751, T_e)
        Ei_grid[i, j] = compute_incident_ion_energy(I750_grid[i, j], I751_grid[i, j], E_bar_750, E_bar_751, c_const)

plt.figure()
plt.imshow(Ei_grid, extent=[flow_dense_2D[0], flow_dense_2D[-1], power_dense_2D[0], power_dense_2D[-1]],
           origin='lower', aspect='auto', cmap='inferno')
plt.xlabel("Flow Rate (sccm)")
plt.ylabel("Power (W)")
#plt.title("Heatmap of Incident Ion Energy vs. Power and Flow Rate")
plt.colorbar(label=r"$E_i$ (eV)")
save_fig("Heatmap_Incident_Ion_Energy")
plt.close()

print("New plots (electron temperature, MB distributions, optical emission, velocity, and incident ion energy) are saved in the './results' directory.")
