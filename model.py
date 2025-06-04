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
    """Save the current figure as a JPEG file in ./results with the given filename."""
    plt.savefig(f"./results/{filename}.jpeg", format='jpeg', bbox_inches='tight')

# ===============================
# Physical Constants
# ===============================
m_e = 9.10938356e-31          # Electron mass in kg
k_J = 1.3807e-23              # Boltzmann constant in J/K
k_eV = 8.617333262145e-5      # Boltzmann constant in eV/K
h = 6.626e-34                 # Planck's constant in J·s
eV_to_J = 1.602176634e-19     # Conversion factor from eV to J

# ===============================
# Updated Parameters for Gas Temperature & Pressure
# ===============================
# Typical baseline gas temperature (room temperature) in Kelvin
T_gas0 = 300.0  # K
# Heating coefficient: increase in gas temperature per watt (example: 0.5 K/W)
xi = 2  # K/W

# Reference pressure and flow rate (for scaling)
P0 = 0.1    # Pa (typical low-pressure sputtering condition)
Q0 = 20.0   # sccm (reference flow rate)
delta = 1.1 # Empirical exponent

# ===============================
# Function Definitions
# ===============================
def maxwell_boltzmann_eedf(e, T_e):
    """
    Return the Maxwell–Boltzmann electron energy distribution (EEDF) 
    at energy e (in eV) for electron temperature T_e (in eV).
    """
    return (1.0 / T_e) * np.exp(-e / T_e)

def compute_total_cross_section(diff_csv_file, energy_target, tol=1e-6):
    """
    Compute the total cross section, σ_tot(E), from differential cross section data.
    The CSV file is assumed to have columns: "Angle (deg)", "Energy (eV)",
    "Differential cross section (m2/sr)".
    
    σ_tot(E) = 2π ∫₀^π (dσ/dΩ)(E,θ) sinθ dθ.
    Only rows where Energy is within tol of energy_target are used.
    """
    data = pd.read_csv(diff_csv_file)
    mask = np.isclose(data["Energy (eV)"].values, energy_target, atol=tol)
    angles_deg = data["Angle (deg)"].values[mask]
    diff_cross = data["Differential cross section (m2/sr)"].values[mask]
    angles_rad = np.deg2rad(angles_deg)
    integral = trapz(diff_cross * np.sin(angles_rad), angles_rad)
    sigma_tot = 2 * np.pi * integral
    return sigma_tot

def compute_mean_free_path(P, Q, diff_csv_file, energy_target, P0, Q0, delta, T_gas0, xi):
    """
    Compute the mean free path λ_mfp(E;P,Q) defined as:
    
      λ_mfp(E;P,Q) = [k_B (T_gas0 + ξ P)] / [ Q P0 (Q/Q0)^δ σ_tot(E) ],
      
    where the neutral density is given by:
      n_Ar = Q P0 (Q/Q0)^δ / [k_B (T_gas0 + ξ P)].
    """
    # Calculate gas temperature (in Kelvin) from power:
    T_gas = T_gas0 + xi * P ** 0.8
    # Calculate gas pressure from flow rate:
    P_gas = P0 * (Q / Q0)**delta
    # Neutral gas density via the ideal gas law:
    n_Ar = (Q * P_gas) / (k_J * T_gas)
    sigma_tot = compute_total_cross_section(diff_csv_file, energy_target)
    lambda_mfp = 1 / (n_Ar * sigma_tot)
    return lambda_mfp

# ===============================
# Plotting Mean Free Path vs. Power and Flow Rate
# ===============================
# Define a differential cross section file and target energy (example value)
diff_cs_file = './diff_crosssection/differential_cross_section.csv'
energy_target = 0.0013605  # eV (sample value from the file)

# Define ranges for power and flow rate
power_values = np.linspace(10, 150, 15)   # from 10 to 150 W
flow_values = np.linspace(10, 120, 12)      # from 10 to 120 sccm

# Plot mean free path vs. Power for a fixed flow rate (e.g., Q = 50 sccm)
fixed_Q = 50.0
mfp_vs_power = [compute_mean_free_path(P, fixed_Q, diff_cs_file, energy_target, P0, Q0, delta, T_gas0, xi) for P in power_values]

plt.figure()
plt.plot(power_values, mfp_vs_power, marker='o')
plt.xlabel("Input Power (W)")
plt.ylabel(r"Mean Free Path $\lambda_{\mathrm{mfp}}$ (m)")
# plt.title("Mean Free Path vs. Power (Q = 50 sccm)")
save_fig("MFP_vs_Power")
plt.close()

# Plot mean free path vs. Flow Rate for a fixed power (e.g., P = 100 W)
fixed_P = 100.0
mfp_vs_flow = [compute_mean_free_path(fixed_P, Q, diff_cs_file, energy_target, P0, Q0, delta, T_gas0, xi) for Q in flow_values]

plt.figure()
plt.plot(flow_values, mfp_vs_flow, marker='o')
plt.xlabel("Flow Rate (sccm)")
plt.ylabel(r"Mean Free Path $\lambda_{\mathrm{mfp}}$ (m)")
# plt.title("Mean Free Path vs. Flow Rate (P = 100 W)")
save_fig("MFP_vs_Flow")
plt.close()

# ===============================
# Plotting Differential Cross Sections vs. Angle
# ===============================
data = pd.read_csv(diff_cs_file)
# Filter data for energies between 10 and 15 eV
data_filtered = data[(data["Energy (eV)"] >= 10) & (data["Energy (eV)"] <= 15)]
unique_energies = np.sort(data_filtered["Energy (eV)"].unique())

plt.figure()
for energy in unique_energies:
    subset = data_filtered[data_filtered["Energy (eV)"] == energy].sort_values("Angle (deg)")
    plt.plot(subset["Angle (deg)"].values, subset["Differential cross section (m2/sr)"].values, linestyle='-', label=f"{energy:.2f} eV")
plt.xlabel("Scattering Angle (deg)")
plt.ylabel("Differential Cross Section (m$^2$/sr)")
# plt.title("Differential Cross Sections vs. Angle (10--15 eV)")
plt.legend(title="Energy")
plt.tight_layout()
save_fig("DiffCrossSection_vs_Angle")
plt.close()

# ===============================
# Plotting Maxwell–Boltzmann Electron Energy Distribution
# ===============================
def plot_EEDF(T_e_values):
    energies = np.linspace(0, 20, 500)
    plt.figure()
    for T in T_e_values:
        f_e_vals = maxwell_boltzmann_eedf(energies, T)
        plt.plot(energies, f_e_vals, label=r"$T_e = {}$ eV".format(T))
    plt.xlabel(r"Electron Energy $\varepsilon$ (eV)")
    plt.ylabel(r"$f_e(\varepsilon)$")
    # plt.title("Maxwell–Boltzmann Electron Energy Distribution")
    plt.legend()
    save_fig("EEDF_vs_Te")
    plt.close()

plot_EEDF([1, 2, 3, 4])

print("Plots for mean free path vs. power, vs. flow rate, differential cross sections, and the EEDF are saved in the './results' directory.")
