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
T_gas0 = 300.0              # Baseline gas temperature (K)
xi = 2                    # Heating coefficient (K/W)
P0 = 0.1                    # Reference pressure (Pa)
Q0 = 20.0                   # Reference flow rate (sccm)
delta = 1.1                 # Empirical exponent

# ===============================
# Function Definitions
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

def compute_mean_free_path(P, Q, diff_csv_file, energy_target):
    T_gas = T_gas0 + xi * P ** 0.8
    P_gas = P0 * (Q / Q0)**delta
    n_Ar = (Q * P_gas) / (k_J * T_gas)
    sigma_tot = compute_total_cross_section(diff_csv_file, energy_target)
    return 1 / (n_Ar * sigma_tot)

def compute_electron_temperature(P, Q, T_e0, beta, gamma):
    return T_e0 * (P ** beta) * (Q ** gamma)

# ===============================
# Parameter Ranges for Plots
# ===============================
power_range = np.linspace(10, 150, 50)
flow_range = np.linspace(10, 150, 50)

T_e0 = 2.0
beta = 0.2
gamma = -0.1

# ===============================
# 1. Mean Free Path vs. Power and Flow Rate
# ===============================
diff_cs_file = './diff_crosssection/differential_cross_section.csv'
energy_target = 0.0013605

mfp_vs_power = [compute_mean_free_path(P, 50, diff_cs_file, energy_target) for P in power_range]
mfp_vs_flow = [compute_mean_free_path(100, Q, diff_cs_file, energy_target) for Q in flow_range]

plt.figure()
plt.plot(power_range, mfp_vs_power)
plt.xlabel("Input Power (W)")
plt.ylabel(r"Mean Free Path $\lambda_{\mathrm{mfp}}$ (m)")
# plt.title("Mean Free Path vs. Power (Q = 50 sccm)")
save_fig("MFP_vs_Power")
plt.close()

plt.figure()
plt.plot(flow_range, mfp_vs_flow)
plt.xlabel("Flow Rate (sccm)")
plt.ylabel(r"Mean Free Path $\lambda_{\mathrm{mfp}}$ (m)")
# plt.title("Mean Free Path vs. Flow Rate (P = 100 W)")
save_fig("MFP_vs_Flow")
plt.close()

# ===============================
# 2. Electron Temperature vs. Power
# ===============================
plt.figure()
for Q in [20, 50, 80, 120]:
    T_e_vals = compute_electron_temperature(power_range, Q, T_e0, beta, gamma)
    plt.plot(power_range, T_e_vals, marker='o', label=f"Q = {Q} sccm")
plt.xlabel("Input Power (W)")
plt.ylabel(r"Electron Temperature $T_e$ (eV)")
# plt.title("Electron Temperature vs. Power for Different Flow Rates")
plt.legend()
save_fig("Te_vs_Power_Different_Flow")
plt.close()

# ===============================
# 3. Maxwell–Boltzmann Distribution Heatmap
# ===============================
P_grid, Q_grid = np.meshgrid(power_range, flow_range, indexing='ij')
T_e_grid = compute_electron_temperature(P_grid, Q_grid, T_e0, beta, gamma)
MB_grid = maxwell_boltzmann_eedf(2.0, T_e_grid)

plt.figure()
plt.imshow(MB_grid, extent=[flow_range[0], flow_range[-1], power_range[0], power_range[-1]],
           origin='lower', aspect='auto', cmap='viridis')
plt.xlabel("Flow Rate (sccm)")
plt.ylabel("Power (W)")
# plt.title(r"Heatmap of $f_e(E=2\,eV)$ vs. Power and Flow Rate")
plt.colorbar(label=r"$f_e(2\,eV)$")
save_fig("Heatmap_MB_vs_P_and_Q")
plt.close()

# ===============================
# 5. Heatmap of Electron Velocity
# ===============================
v_e_grid = np.sqrt(2 * T_e_grid / m_e)

plt.figure()
plt.imshow(v_e_grid, extent=[flow_range[0], flow_range[-1], power_range[0], power_range[-1]],
           origin='lower', aspect='auto', cmap='plasma')
plt.xlabel("Flow Rate (sccm)")
plt.ylabel("Power (W)")
# plt.title("Heatmap of Electron Velocity vs. Power and Flow Rate")
plt.colorbar(label=r"$v_e$ (m/s)")
save_fig("Heatmap_Electron_Velocity")
plt.close()

print("Plots saved in the './results' directory.")
