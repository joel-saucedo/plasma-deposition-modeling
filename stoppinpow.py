import os
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# -------------------------------
# Global Plot Settings
# -------------------------------
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
if not os.path.exists('./crosssection'):
    os.makedirs('./crosssection')

def save_fig(filename):
    plt.savefig(f"./results/{filename}.jpeg", format='jpeg', bbox_inches='tight')

# -------------------------------
# Conversion Factors
# -------------------------------
hartree_to_eV = 27.211386      # 1 Hartree = 27.211386 eV
bohr2_to_m2 = (5.29177210903e-11)**2  # 1 Bohr^2 in m^2

# -------------------------------
# Function to Load Moment Data from CSV
# -------------------------------
def load_moment_data(csv_file):
    """
    Loads a CSV file with columns: Energy (Hartee), Q(00), Q(01), Q(02), Q(03), Q(04)
    and returns a DataFrame with Energy in eV and the moments in m^2.
    """
    df = pd.read_csv(csv_file)
    df["Energy (eV)"] = df["Energy (Hartee)"] * hartree_to_eV
    for col in df.columns:
        if col.startswith("Q("):
            df[col] = df[col] * bohr2_to_m2
    return df

# -------------------------------
# Plot Q0 vs. Energy for Zn and O (with interpolation)
# -------------------------------
def plot_Q0(df_Zn, df_O):
    # Use Zn's energy grid as reference
    E_vals_Zn = df_Zn["Energy (eV)"].values
    Q0_Zn = df_Zn["Q(00)"].values
    
    # Interpolate O's Q(00) values onto Zn's energy grid
    E_vals_O = df_O["Energy (eV)"].values
    Q0_O_interp = np.interp(E_vals_Zn, E_vals_O, df_O["Q(00)"].values)
    
    plt.figure()
    plt.plot(E_vals_Zn, Q0_Zn,  linestyle='-', label="Zn Q$_0$")
    plt.plot(E_vals_Zn, Q0_O_interp, marker='s', linestyle='-', label="O Q$_0$ (interpolated)")
    plt.xscale('log')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Total Cross Section, Q$_0$(E) (m$^2$)")
    # plt.title("Total Cross Section Q$_0$(E) vs. Energy for Zn and O")
    plt.legend()
    save_fig("Q0_vs_Energy_Zn_vs_O")
    plt.close()

# -------------------------------
# Nuclear Stopping Power and Sputtering Yield Calculations
# -------------------------------
# Physical constants (SI units)
M_Ar = 6.6335209e-26          # mass of Ar (kg)
M_Zn = 65.38 * 1.66054e-27     # mass of Zn (kg)
M_O  = 16.00 * 1.66054e-27     # mass of O (kg)

def stopping_factor(M_Ar, M_X):
    """Calculate stopping factor: K = 2 M_Ar M_X/(M_Ar+M_X)^2."""
    return 2 * M_Ar * M_X / (M_Ar + M_X)**2

def stopping_power(E, Q0, M_Ar, M_X):
    """
    Calculate nuclear stopping power:
      S_n(E) = (2 M_Ar M_X/(M_Ar+M_X)^2) * E * Q0(E),
    with E in eV and Q0 in m^2.
    """
    K = stopping_factor(M_Ar, M_X)
    return K * E * Q0

def effective_stopping_power(E, Q0_Zn, Q0_O):
    """
    For a Zn:O 1:1 target, the effective stopping power is the average of the species-specific values.
    """
    S_Zn = stopping_power(E, Q0_Zn, M_Ar, M_Zn)
    S_O  = stopping_power(E, Q0_O,  M_Ar, M_O)
    return 0.5 * (S_Zn + S_O)

# Sputtering yield model parameters
Lambda = 0.05         # dimensionless constant
E_th = 25.0           # sputtering threshold energy (eV)
E_b = 3.5             # surface binding energy (eV)

def sputtering_yield(E, S_eff):
    """
    Simplified sputtering yield model assuming effective incident energy E_i ≈ E:
      Y(E) = Λ (E - E_th)/E_b * S_eff(E).
    """
    return Lambda * (E - E_th) / E_b * S_eff

def deposition_flux(Y):
    """
    Calculate deposition flux from sputtering yield.
    For n = 1 (cosine distribution), the angular projection factor is 2/3.
    """
    return Y * (2/3)

# -------------------------------
# Plotting Functions for Stopping Power, Sputtering Yield, and Deposition Flux
# -------------------------------
def plot_stopping_power(df_Zn, df_O):
    E_vals_Zn = df_Zn["Energy (eV)"].values
    Q0_Zn = df_Zn["Q(00)"].values
    E_vals_O = df_O["Energy (eV)"].values
    Q0_O_interp = np.interp(E_vals_Zn, E_vals_O, df_O["Q(00)"].values)
    
    S_Zn = stopping_power(E_vals_Zn, Q0_Zn, M_Ar, M_Zn)
    S_O  = stopping_power(E_vals_Zn, Q0_O_interp, M_Ar, M_O)
    S_eff = 0.5 * (S_Zn + S_O)
    
    plt.figure()
    plt.plot(E_vals_Zn, S_Zn, label="Stopping Power Zn")
    plt.plot(E_vals_Zn, S_O,  label="Stopping Power O")
    plt.plot(E_vals_Zn, S_eff, label="Effective Stopping Power (Zn:O 1:1)", linestyle='--', linewidth=2)
    plt.xlabel("Energy (eV)")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel("Nuclear Stopping Power $S_n(E)$")
    # plt.title("Nuclear Stopping Power vs. Energy")
    plt.legend()
    # plt.xscale('log')
    save_fig("Stopping_Power_vs_Energy")
    plt.close()
    
def plot_sputtering_yield(df_Zn, df_O):
    E_vals_Zn = df_Zn["Energy (eV)"].values
    Q0_Zn = df_Zn["Q(00)"].values
    E_vals_O = df_O["Energy (eV)"].values
    Q0_O_interp = np.interp(E_vals_Zn, E_vals_O, df_O["Q(00)"].values)
    
    S_eff = effective_stopping_power(E_vals_Zn, Q0_Zn, Q0_O_interp)
    Y = sputtering_yield(E_vals_Zn, S_eff)
    
    plt.figure()
    plt.plot(E_vals_Zn, Y, label="Sputtering Yield")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Sputtering Yield $Y(E)$")
    # plt.yscale("log")

    # plt.title("Sputtering Yield vs. Energy")
    # plt.xscale('log')
    plt.legend()
    save_fig("Sputtering_Yield_vs_Energy")
    plt.close()

def plot_deposition_flux(df_Zn, df_O):
    E_vals_Zn = df_Zn["Energy (eV)"].values
    Q0_Zn = df_Zn["Q(00)"].values
    E_vals_O = df_O["Energy (eV)"].values
    Q0_O_interp = np.interp(E_vals_Zn, E_vals_O, df_O["Q(00)"].values)
    
    S_eff = effective_stopping_power(E_vals_Zn, Q0_Zn, Q0_O_interp)
    Y = sputtering_yield(E_vals_Zn, S_eff)
    Phi = deposition_flux(Y)
    
    plt.figure()
    plt.plot(E_vals_Zn, Phi, label="Deposition Flux")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Deposition Flux $\Phi_{\mathrm{sub}}(E)$ ()")
    # plt.title("Deposition Flux vs. Energy (Effective Zn:O 1:1)")
    # plt.xscale('log')
    plt.legend()
    save_fig("Deposition_Flux_vs_Energy")
    plt.close()

# -------------------------------
# Main Workflow
# -------------------------------
if __name__ == "__main__":
    # Load moment data (assumed stored in CSV files in the ./crosssection directory)
    df_Zn = load_moment_data("./crosssection/Zn_cross_section_moments.csv")
    df_O  = load_moment_data("./crosssection/O_cross_section_moments.csv")
    
    # Plot Q0 (total cross section) for Zn and O vs. Energy
    plot_Q0(df_Zn, df_O)
    
    # Plot effective nuclear stopping power vs. Energy
    plot_stopping_power(df_Zn, df_O)
    
    # Plot sputtering yield vs. Energy for effective Zn:O system
    plot_sputtering_yield(df_Zn, df_O)
    
    # Plot deposition flux vs. Energy for effective Zn:O system
    plot_deposition_flux(df_Zn, df_O)
    
    print("Processing complete. Plots for Q$_0$, effective stopping power, sputtering yield, and deposition flux have been saved in './results'.")
















import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma

# -------------------------------
# Global Plot Settings
# -------------------------------
try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    plt.style.use('default')

plt.rcParams['figure.dpi'] = 300          
plt.rcParams['savefig.dpi'] = 300           
plt.rcParams['text.usetex'] = True          
plt.rcParams['font.size'] = 14              
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (8, 6)

if not os.path.exists('./results'):
    os.makedirs('./results')

def save_fig(filename):
    plt.savefig(f"./results/{filename}.jpeg", format='jpeg', bbox_inches='tight')

# -------------------------------
# Sputtering Yield Model Parameters (example values)
# -------------------------------
Lambda = 0.05         # dimensionless constant
E_th = 25.0           # threshold energy for sputtering (eV)
E_b = 3.5             # surface binding energy (eV)

def sputtering_yield(E):
    """
    Simplified sputtering yield:
      Y(E) = Λ (E - E_th)/E_b * E   for E > E_th, zero otherwise.
    Here we assume the effective incident energy E_i ≈ E and that the effective stopping power is ∝ E.
    """
    Y = np.zeros_like(E)
    mask = E > E_th
    Y[mask] = Lambda * (E[mask] - E_th) / E_b * E[mask]
    return Y

# -------------------------------
# Angular Projection Integral I(n)
# -------------------------------
def I_n(n):
    """
    Computes the integral:
      I(n) = ∫_0^(π/2) cos^n(θ)*cos(θ) dθ.
    Alternatively, note that I(n) = ∫_0^(π/2) cos^(n+1)(θ) dθ,
    which has the closed-form solution:
      I(n) = sqrt(pi)*Gamma((n+2)/2)/(2*Gamma((n+3)/2)).
    Here we compute it numerically.
    """
    integrand = lambda theta: np.cos(theta)**(n+1)
    result, _ = quad(integrand, 0, np.pi/2)
    return result

# -------------------------------
# Deposition Flux Calculation
# -------------------------------
def deposition_flux(E, n):
    """
    Given energy E (array) and angular exponent n, compute the deposition flux:
      Φ_sub(E;n) = Y(E) * I(n),
    where Y(E) is the sputtering yield.
    """
    Y = sputtering_yield(E)
    return Y * I_n(n)

# -------------------------------
# Main Plot: Deposition Flux vs. Energy for Different n
# -------------------------------
def plot_deposition_flux_curves():
    # Define an energy grid (eV) covering from, say, 10 eV to 1000 eV
    E_vals = np.linspace(0, 80, 200)  # 10 eV to 1000 eV
    # Define a set of angular exponents n to consider
    n_values = [0, 1, 2, 3, 4]
    
    plt.figure()
    for n in n_values:
        Phi = deposition_flux(E_vals, n)
        plt.plot(E_vals, Phi,  linestyle='-', label=f"n = {n}")
    
    # plt.xscale('log')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Deposition Flux $\Phi_{sub}(E)$")
    # plt.title("Deposition Flux vs. Energy for Different Angular Exponents")
    plt.legend(title="Angular Exponent n")
    save_fig("Deposition_Flux_vs_Energy_Diff_n")
    plt.close()

if __name__ == "__main__":
    plot_deposition_flux_curves()
    print("Deposition flux curves for different angular exponents have been saved in './results'.")
