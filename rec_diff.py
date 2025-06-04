import os
import csv
import numpy as np
import pandas as pd
from numpy import trapz
from scipy.optimize import curve_fit
from scipy.special import eval_legendre
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
# Plotting Moments vs. Energy
# -------------------------------
def plot_moments(df, species):
    plt.figure()
    for col in df.columns:
        if col.startswith("Q("):
            plt.plot(df["Energy (eV)"], df[col], marker='o', label=col)
    plt.xscale('log')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Moment (m$^2$)")
    plt.title(f"{species} Cross-Section Moments vs. Energy")
    plt.legend()
    save_fig(f"{species}_Moments_vs_Energy")
    plt.close()

# -------------------------------
# Reconstructing Differential Cross Section (Positivity-Preserving)
# -------------------------------
def reconstruct_diff_cs(df, energy_value, n_max=4):
    """
    Reconstruct the differential cross section for a given energy value using the Legendre expansion.
    Assumes the CSV contains moments Q(00) (total cross section) through Q(n_max), where Q(00) is the full scattering.
    Normalizes the moments:
      σ_tot(E) = ∑ₙ₌₀^(n_max) Qₙ(E)   and   ˜Qₙ(E) = Qₙ(E)/σ_tot(E).
    Then,
      dσ/dΩ(E,θ) = σ_tot(E) ∑ₙ₌₀^(n_max) (2n+1)/(4π) ˜Qₙ(E) Pₙ(cosθ).
    Enforces non-negativity.
    """
    row = df.iloc[(df["Energy (eV)"] - energy_value).abs().argsort()[:1]]
    # Extract moments Q(00) to Q(n_max)
    moments = [row[f"Q({i:02d})"].values[0] for i in range(0, n_max+1)]
    if len(moments) < n_max+1:
        moments += [moments[-1]] * (n_max+1 - len(moments))
    sigma_tot = sum(moments)
    # Normalize moments
    norm_moments = [m / sigma_tot for m in moments]
    angles_deg = np.linspace(0, 180, 181)
    angles_rad = np.deg2rad(angles_deg)
    diff_cs = np.zeros_like(angles_rad)
    for n in range(0, n_max+1):
        leg_poly = eval_legendre(n, np.cos(angles_rad))
        diff_cs += (2*n+1)/(4*np.pi) * norm_moments[n] * leg_poly
    diff_cs = sigma_tot * np.maximum(diff_cs, 0.0)
    return angles_deg, diff_cs

def plot_reconstructed_diff_cs(df, species, energies_to_plot, n_max=4):
    plt.figure()
    for energy_value in energies_to_plot:
        angles_deg, diff_cs = reconstruct_diff_cs(df, energy_value, n_max)
        plt.plot(angles_deg, diff_cs, label=f"E = {energy_value:.2f} eV")
    plt.xlabel("Scattering Angle (deg)")
    plt.ylabel(r"Differential Cross Section (m$^2$/sr)")
    plt.yscale('log')
    plt.title(f"Reconstructed Differential Cross Section for {species}")
    plt.legend(title="Energy")
    save_fig(f"{species}_Reconstructed_DiffCS")
    plt.close()

def save_reconstructed_diff_cs(df, species, n_max=4):
    angles_deg = np.linspace(0, 180, 181)
    out_rows = []
    header = ["Energy (eV)"] + [f"Angle {angle:.0f} deg" for angle in angles_deg]
    for idx, row in df.iterrows():
        E = row["Energy (eV)"]
        moments = [row[f"Q({i:02d})"] for i in range(0, n_max+1)]
        if len(moments) < n_max+1:
            moments += [moments[-1]] * (n_max+1 - len(moments))
        sigma_tot = sum(moments)
        norm_moments = [m / sigma_tot for m in moments]
        diff_cs = np.zeros_like(angles_deg, dtype=float)
        for n in range(0, n_max+1):
            diff_cs += (2*n+1)/(4*np.pi) * norm_moments[n] * eval_legendre(n, np.cos(np.deg2rad(angles_deg)))
        diff_cs = sigma_tot * np.maximum(diff_cs, 0.0)
        out_rows.append([E] + diff_cs.tolist())
    csv_filename = os.path.join("./diff_crosssection", f"{species}_reconstructed_diff_cs.csv")
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(out_rows)
    print(f"Saved reconstructed differential cross section data for {species} to {csv_filename}")

# -------------------------------
# Main Workflow
# -------------------------------
if __name__ == "__main__":
    # Assume CSV files are saved in ./crosssection with names "Zn_cross_section_moments.csv" and "O_cross_section_moments.csv"
    df_Zn = load_moment_data("./crosssection/Zn_cross_section_moments.csv")
    df_O  = load_moment_data("./crosssection/O_cross_section_moments.csv")
    
    # Plot moments vs. energy for each species
    plot_moments(df_Zn, "Zn")
    plot_moments(df_O, "O")
    
    # Choose a few energy values (in eV) for reconstruction
    energies_to_plot = [min(df_Zn["Energy (eV)"]),
                        np.median(df_Zn["Energy (eV)"]),
                        max(df_Zn["Energy (eV)"])]
    
    # Plot reconstructed differential cross sections for Zn and O
    plot_reconstructed_diff_cs(df_Zn, "Zn", energies_to_plot, n_max=4)
    plot_reconstructed_diff_cs(df_O, "O", energies_to_plot, n_max=4)
    
    # Save reconstructed differential cross section data to CSV for later use
    save_reconstructed_diff_cs(df_Zn, "Zn", n_max=4)
    save_reconstructed_diff_cs(df_O, "O", n_max=4)
    
    print("Processing complete. Moments and reconstructed differential cross-section plots/data have been saved.")
