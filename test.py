import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===============================
# Global Plot Settings
# ===============================
try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    print("scienceplots module not found. Using default matplotlib style.")
    plt.style.use('default')

mpl.rcParams['figure.dpi'] = 300          # Set the display DPI for figures
mpl.rcParams['savefig.dpi'] = 300           # Set the DPI for saved figures
mpl.rcParams['text.usetex'] = True          # Enable LaTeX rendering
mpl.rcParams['font.size'] = 14              # Increase base font size for readability
mpl.rcParams['font.family'] = 'serif'       # Use a serif font for a classic look
mpl.rcParams['axes.labelsize'] = 16         # Label font size for axes
mpl.rcParams['xtick.labelsize'] = 14        # Font size for x-tick labels
mpl.rcParams['ytick.labelsize'] = 14        # Font size for y-tick labels
mpl.rcParams['legend.fontsize'] = 14        # Font size for legends
mpl.rcParams['figure.figsize'] = (8, 6)     # Set a standard figure size

def save_fig(filename):
    """Save the current figure as a JPEG file with the given filename."""
    plt.savefig(f"{filename}.jpeg", format='jpeg', bbox_inches='tight')

# ===============================
# Physical Constants
# ===============================
m_e = 9.10938356e-31           # Electron mass in kg
k_J = 1.3807e-23               # Boltzmann constant in J/K
k_eV = 8.617333262145e-5       # Boltzmann constant in eV/K
h = 6.626e-34                  # Planck's constant in J·s
eV_to_J = 1.602176634e-19      # Conversion factor from eV to J

# ===============================
# Function Definitions
# ===============================
def maxwell_boltzmann_eedf(e, T_e):
    """
    Maxwell-Boltzmann electron energy distribution function (EEDF) for energy e (in eV)
    and electron temperature T_e (in eV). Assumes f_e is normalized such that:
    ∫_0^∞ f_e(e) de = 1.
    
    Parameters:
      e : float or np.array
          Electron energy in eV.
      T_e : float
          Electron temperature in eV.
    
    Returns:
      float or np.array: The value of the EEDF at energy e.
    """
    return (1.0 / T_e) * np.exp(-e / T_e)

def compute_excitation_rate(n_e, n_Ar, cross_section_file, T_e):
    """
    Compute the excitation rate coefficient R_exc given electron density (n_e),
    neutral argon density (n_Ar), cross-section data file, and electron temperature T_e.
    
    R_exc = n_e * n_Ar * ∫_0^∞ σ_exc(e) * v_e(e) * f_e(e, T_e) de,
    where v_e(e) = sqrt(2 * e_J / m_e) and e_J = e (in eV) converted to Joules.
    
    Parameters:
      n_e : float
          Electron density (in m^-3).
      n_Ar : float
          Neutral argon density (in m^-3).
      cross_section_file : str
          Path to the CSV file containing the cross-section data with headers
          "Energy (eV)" and "Cross Section (m^2)".
      T_e : float
          Electron temperature in eV.
    
    Returns:
      float: The excitation rate coefficient R_exc.
    """
    data = pd.read_csv(cross_section_file)
    energy = data["Energy (eV)"].values  # in eV
    cross_section = data["Cross Section (m^2)"].values  # in m^2
    sigma_interp = interp1d(energy, cross_section, bounds_error=False, fill_value=0.0)
    
    def integrand(e):
        e_J = e * eV_to_J
        v_e = np.sqrt(2 * e_J / m_e)
        return sigma_interp(e) * v_e * maxwell_boltzmann_eedf(e, T_e)
    
    integral_value, _ = quad(integrand, 0, np.inf)
    R_exc = n_e * n_Ar * integral_value
    return R_exc

def compute_densities(P, Q, C_P, C_Q, α):
    """
    Compute electron density and neutral argon density.
    
    n_e = C_P * P^α and n_Ar = C_Q * Q.
    
    Parameters:
      P : float
          Input power in Watts.
      Q : float
          Gas flow rate (in appropriate units).
      C_P : float
          Proportionality constant for electron density (in m^-3 W^(-α)).
      C_Q : float
          Proportionality constant for neutral argon density (in m^-3 per unit flow rate).
      α : float
          Empirical exponent for electron density scaling.
    
    Returns:
      tuple: (n_e, n_Ar) in m^-3.
    """
    n_e = C_P * (P ** α)
    n_Ar = C_Q * Q
    return n_e, n_Ar

def compute_energy_flux(n_e, T_e):
    """
    Compute the plasma energy flux delivered to the substrate.
    
    Φ_E = n_e * (T_e * eV_to_J), where T_e is in eV.
    
    Parameters:
      n_e : float
          Electron density (in m^-3).
      T_e : float
          Electron temperature (in eV).
    
    Returns:
      float: Plasma energy flux in J/m^3.
    """
    Φ_E = n_e * (T_e * eV_to_J)
    return Φ_E

def compute_optical_emission(R_exc):
    """
    Compute optical emission intensity I_λ proportional to R_exc.
    
    Parameters:
      R_exc : float
          Excitation rate coefficient.
    
    Returns:
      float: Optical emission intensity.
    """
    I_lambda = R_exc
    return I_lambda

def compute_effective_energy(cross_section_file, threshold, T_e):
    """
    Compute effective energy (mean energy) above a threshold:
    
      E_bar = [∫_threshold^∞ ε σ(ε) sqrt(2*ε_J/m_e) exp(-ε/T_e) dε] /
              [∫_threshold^∞ σ(ε) sqrt(2*ε_J/m_e) exp(-ε/T_e) dε],
    where ε is in eV, ε_J = ε * eV_to_J.
    
    Parameters:
      cross_section_file : str
          Path to CSV file with columns "Energy (eV)" and "Cross Section (m^2)".
      threshold : float
          Threshold energy in eV.
      T_e : float
          Electron temperature in eV.
    
    Returns:
      float: Effective energy E_bar in eV.
    """
    data = pd.read_csv(cross_section_file)
    energy = data["Energy (eV)"].values  # in eV
    sigma = data["Cross Section (m^2)"].values  # in m^2
    sigma_interp = interp1d(energy, sigma, bounds_error=False, fill_value=0.0)
    
    def numerator_integrand(e):
        e_J = e * eV_to_J
        v_e = np.sqrt(2 * e_J / m_e)
        return e * sigma_interp(e) * v_e * np.exp(-e / T_e)
    
    def denominator_integrand(e):
        e_J = e * eV_to_J
        v_e = np.sqrt(2 * e_J / m_e)
        return sigma_interp(e) * v_e * np.exp(-e / T_e)
    
    num, _ = quad(numerator_integrand, threshold, np.inf)
    den, _ = quad(denominator_integrand, threshold, np.inf)
    E_bar = num / den if den != 0 else 0
    return E_bar

def compute_incident_ion_energy(I_750, I_751, E_750, E_751, c):
    """
    Compute effective incident ion energy as weighted average:
    
      E_i = c * (I_750 * E_750 + I_751 * E_751) / (I_750 + I_751).
    
    Parameters:
      I_750 : float
          Optical intensity for 750 nm.
      I_751 : float
          Optical intensity for 751 nm.
      E_750 : float
          Effective energy for 750 nm (in eV).
      E_751 : float
          Effective energy for 751 nm (in eV).
      c : float
          Proportionality constant.
    
    Returns:
      float: Incident ion energy E_i in eV.
    """
    if (I_750 + I_751) == 0:
        return 0
    E_i = c * (I_750 * E_750 + I_751 * E_751) / (I_750 + I_751)
    return E_i

def compute_sputtering_yield(E_i, E_th, E_b, Lambda):
    """
    Compute sputtering yield:
    
      Y(E) = Lambda * (E_i - E_th) / E_b.
    
    Parameters:
      E_i : float
          Incident ion energy (in eV).
      E_th : float
          Threshold energy for sputtering (in eV).
      E_b : float
          Surface binding energy (in eV).
      Lambda : float
          Sputtering yield constant.
    
    Returns:
      float: Sputtering yield.
    """
    Y = Lambda * (E_i - E_th) / E_b
    return Y

def compute_deposition_flux(Y, n):
    """
    Compute deposition flux:
    
      Φ_sub = Y * ∫_0^(pi/2) cos^n(theta)*cos(theta) dtheta.
    
    Parameters:
      Y : float
          Sputtering yield.
      n : float
          Angular distribution exponent.
    
    Returns:
      float: Deposition flux Φ_sub.
    """
    def integrand(theta):
        return (np.cos(theta)**n) * np.cos(theta)
    integral_val, _ = quad(integrand, 0, np.pi/2)
    Phi_sub = Y * integral_val
    return Phi_sub

def compute_nucleation_density(Phi_sub, D_0, E_d, T_s):
    """
    Compute nucleation density N.
    
      D_s = D_0 * exp(-E_d / (k_eV * T_s)), with k_eV in eV/K,
      and then N ~ Phi_sub / D_s.
    
    Parameters:
      Phi_sub : float
          Deposition flux.
      D_0 : float
          Pre-exponential factor for surface diffusion (m^2/s).
      E_d : float
          Diffusion activation energy (in eV).
      T_s : float
          Substrate temperature (in K).
    
    Returns:
      float: Nucleation density N.
    """
    k_eV = 8.617333262145e-5  # eV/K
    D_s = D_0 * np.exp(-E_d / (k_eV * T_s))
    N = Phi_sub / D_s
    return N

def compute_grain_size(N):
    """
    Compute mean grain size and grain boundary length.
    
      d_bar ~ 1/sqrt(N) and L_g ~ 1/d_bar.
    
    Parameters:
      N : float
          Nucleation density.
    
    Returns:
      tuple: (mean grain size d_bar, grain boundary length L_g)
    """
    d_bar = 1 / np.sqrt(N) if N > 0 else np.inf
    L_g = 1 / d_bar if d_bar > 0 else np.inf
    return d_bar, L_g

def generate_voronoi(N):
    """
    Generate a Voronoi tessellation based on nucleation sites.
    
    Generates a set of points in a unit square corresponding to the nucleation density.
    Limits the number of points between 10 and 500 for visualization. Saves the plot as a JPEG image.
    
    Parameters:
      N : float
          Nucleation density (interpreted as number of points per unit area).
    
    Returns:
      Voronoi: The Voronoi tessellation object.
    """
    num_points = int(np.clip(N, 10, 500))
    points = np.random.rand(num_points, 2)
    vor = Voronoi(points)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2, line_alpha=0.6, point_size=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Voronoi Tessellation of Nucleation Sites")
    plt.tight_layout()
    save_fig("voronoi_tessellation")
    plt.close(fig)
    return vor

# ===============================
# Main Execution (Example Usage)
# ===============================
if __name__ == "__main__":
    # Example input parameters
    P_example = 100.0        # Power in Watts
    Q_example = 1.0          # Flow rate (in chosen units)
    C_P_example = 1e16       # m^-3 W^(-α)
    C_Q_example = 1e20       # m^-3 per unit flow rate
    α_example = 0.5          # Empirical exponent for n_e scaling
    T_e_example = 2.0        # Electron temperature in eV
    T_e0_example = 2.0       # Baseline temperature for scaling (if needed)
    
    # Compute densities and energy flux
    n_e_val, n_Ar_val = compute_densities(P_example, Q_example, C_P_example, C_Q_example, α_example)
    Φ_E_val = compute_energy_flux(n_e_val, T_e_example)
    
    print(f"Electron density: {n_e_val:.3e} m^-3")
    print(f"Neutral Ar density: {n_Ar_val:.3e} m^-3")
    print(f"Plasma energy flux: {Φ_E_val:.3e} J/m^3")
    
    # Compute excitation rate for a given transition (e.g., 750 nm)
    cross_section_file_750 = './crosssection/E_Ar_E_Ar2P5.csv'
    R_exc_750 = compute_excitation_rate(n_e_val, n_Ar_val, cross_section_file_750, T_e_example)
    I_750 = compute_optical_emission(R_exc_750)
    print(f"Optical emission intensity (750 nm): {I_750:.3e}")
    
    # Compute effective energies for both transitions
    E_bar_750 = compute_effective_energy('./crosssection/E_Ar_E_Ar2P5.csv', 13.273, T_e_example)
    E_bar_751 = compute_effective_energy('./crosssection/E_Ar_E_Ar2P1.csv', 13.48, T_e_example)
    print(f"Effective energy for 750 nm: {E_bar_750:.3e} eV")
    print(f"Effective energy for 751 nm: {E_bar_751:.3e} eV")
    
    # Assume some optical intensities for the 751 nm transition (example values)
    cross_section_file_751 = './crosssection/E_Ar_E_Ar2P1.csv'
    R_exc_751 = compute_excitation_rate(n_e_val, n_Ar_val, cross_section_file_751, T_e_example)
    I_751 = compute_optical_emission(R_exc_751)
    print(f"Optical emission intensity (751 nm): {I_751:.3e}")
    
    # Compute incident ion energy from the two transitions
    c_example = 1.5
    E_i_val = compute_incident_ion_energy(I_750, I_751, E_bar_750, E_bar_751, c_example)
    print(f"Incident ion energy: {E_i_val:.3e} eV")
    
    # Compute sputtering yield
    E_th_example = 25.0  # eV
    E_b_example = 3.5    # eV
    Lambda_example = 0.05
    Y_val = compute_sputtering_yield(E_i_val, E_th_example, E_b_example, Lambda_example)
    print(f"Sputtering yield: {Y_val:.3e}")
    
    # Compute deposition flux
    n_exp = 2  # Angular distribution exponent
    Phi_sub_val = compute_deposition_flux(Y_val, n_exp)
    print(f"Deposition flux: {Phi_sub_val:.3e}")
    
    # Compute nucleation density
    D_0_example = 1e-7   # m^2/s
    E_d_example = 0.5    # eV
    T_s_example = 300    # K
    N_val = compute_nucleation_density(Phi_sub_val, D_0_example, E_d_example, T_s_example)
    print(f"Nucleation density: {N_val:.3e}")
    
    # Compute grain size and grain boundary length
    d_bar, L_g = compute_grain_size(N_val)
    print(f"Mean grain size: {d_bar:.3e} m")
    print(f"Grain boundary length per unit area: {L_g:.3e} m^-1")
    
    # Generate and save Voronoi tessellation plot based on nucleation density
    vor = generate_voronoi(N_val)
    print("Voronoi tessellation generated and saved as 'voronoi_tessellation.jpeg'.")



def simulate_power_flow_variability(t, P0, Q0, sigma_P, sigma_Q, tau=None):
    """
    Simulate time-dependent fluctuations in power and gas flow rate.
    
    The power P(t) and gas flow rate Q(t) are modeled as:
        P(t) = P0 + δP(t),   Q(t) = Q0 + δQ(t),
    where δP(t) and δQ(t) are Gaussian random fluctuations with standard deviations sigma_P and sigma_Q, respectively.
    
    Optionally, an exponential low-pass filter with correlation time tau (in seconds) is applied to introduce temporal correlation.
    
    Parameters:
      t : np.array
          1D array of time points (in seconds).
      P0 : float
          Mean power (in Watts).
      Q0 : float
          Mean gas flow rate (in appropriate units).
      sigma_P : float
          Standard deviation of power fluctuations.
      sigma_Q : float
          Standard deviation of flow rate fluctuations.
      tau : float, optional
          Correlation time (in seconds) for the fluctuations. If provided, the time series is low-pass filtered.
    
    Returns:
      tuple: (P_t, Q_t)
          P_t : np.array
              Time series of power values.
          Q_t : np.array
              Time series of gas flow rate values.
    """
    # Generate white noise fluctuations
    P_noise = np.random.normal(0, sigma_P, size=t.shape)
    Q_noise = np.random.normal(0, sigma_Q, size=t.shape)
    
    # Create initial time series with uncorrelated noise
    P_t = P0 + P_noise
    Q_t = Q0 + Q_noise
    
    if tau is not None:
        # Apply an exponential moving average to introduce temporal correlation.
        dt = t[1] - t[0]  # Assumes uniform time spacing.
        alpha_filter = dt / (tau + dt)
        P_filtered = np.zeros_like(P_t)
        Q_filtered = np.zeros_like(Q_t)
        P_filtered[0] = P_t[0]
        Q_filtered[0] = Q_t[0]
        for i in range(1, len(t)):
            P_filtered[i] = alpha_filter * P_t[i] + (1 - alpha_filter) * P_filtered[i-1]
            Q_filtered[i] = alpha_filter * Q_t[i] + (1 - alpha_filter) * Q_filtered[i-1]
        P_t, Q_t = P_filtered, Q_filtered
    
    return P_t, Q_t

# Example usage:
if __name__ == "__main__":
    # Define a time array from 0 to 100 seconds with 0.1 s intervals
    t = np.arange(0, 100, 0.1)
    
    # Mean values for power and flow rate
    P0 = 100.0   # Watts
    Q0 = 1.0     # Arbitrary flow rate units
    
    # Standard deviations for fluctuations
    sigma_P = 5.0   # Watts
    sigma_Q = 0.1   # Flow rate units
    
    # Correlation time (e.g., 5 seconds)
    tau = 5.0
    
    # Simulate the time-dependent power and flow rate
    P_t, Q_t = simulate_power_flow_variability(t, P0, Q0, sigma_P, sigma_Q, tau)
    
    # Plot the simulated time series
    plt.figure()
    plt.plot(t, P_t, label="Power (W)")
    plt.plot(t, Q_t, label="Flow Rate (units)")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Time-Dependent Power and Flow Rate Variability")
    plt.legend()
    save_fig("power_flow_variability")
    plt.show()
