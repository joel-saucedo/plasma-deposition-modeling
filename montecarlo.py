import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib as mpl
import matplotlib.pyplot as plt
from numba import jit, prange

# ===============================
# Global Plot Settings
# ===============================
try:
    import scienceplots
    plt.style.use(['science', 'grid'])
except ImportError:
    print("scienceplots module not found. Using default matplotlib style.")
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

# Create results directory if it doesn't exist
if not os.path.exists('./montecarlo'):
    os.makedirs('./montecarlo')

def save_fig(filename):
    """Save the current figure as a JPEG file in ./montecarlo with the given filename."""
    plt.savefig(f"./montecarlo/{filename}.jpeg", format='jpeg', bbox_inches='tight')

# ===============================
# Physical Constants
# ===============================
m_e = 9.10938356e-31           # Electron mass in kg
k_J = 1.3807e-23               # Boltzmann constant in J/K
k_eV = 8.617333262145e-5       # Boltzmann constant in eV/K
h = 6.626e-34                  # Planck's constant in J·s
eV_to_J = 1.602176634e-19      # Conversion factor from eV to J

# ===============================
# Functions for Plasma & Deposition
# ===============================
def maxwell_boltzmann_eedf(e, T_e):
    """Return the Maxwell-Boltzmann EEDF at energy e (in eV) for temperature T_e (in eV)."""
    return (1.0 / T_e) * np.exp(-e / T_e)

def compute_excitation_rate(n_e, n_Ar, cross_section_file, T_e, P, Q, lambda_P=0.005, lambda_Q=0.002):
    """
    Compute the excitation rate coefficient R_exc with saturation effects.
    R_exc = n_e * n_Ar * ∫_0^∞ σ_exc(e)*v_e(e)*f_e(e,T_e) de × Q_effect,
    with Q_effect = 1 / (1 + lambda_P * P + lambda_Q * Q).
    """
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

def compute_densities(P, Q, C_P, C_Q, α, lambda_P_density=0.02):
    """
    Compute electron density and neutral argon density with saturation for electron density.
    n_e = C_P * P^α / (1 + lambda_P_density * P),
    n_Ar = C_Q * Q.
    """
    n_e = C_P * (P ** α) / (1 + lambda_P_density * P)
    n_Ar = C_Q * Q
    return n_e, n_Ar

def compute_energy_flux(n_e, T_e):
    """Compute plasma energy flux: Φ_E = n_e * (T_e * eV_to_J)."""
    return n_e * (T_e * eV_to_J)

def compute_optical_emission(R_exc):
    """Optical emission intensity is assumed proportional to R_exc."""
    return R_exc

def compute_effective_energy(cross_section_file, threshold, T_e):
    """
    Compute effective energy E_bar above a threshold.
    E_bar = (∫_threshold^∞ ε σ(ε)v_e(e)exp(-ε/T_e) de) / (∫_threshold^∞ σ(ε)v_e(e)exp(-ε/T_e) de).
    """
    data = pd.read_csv(cross_section_file)
    energy = data["Energy (eV)"].values
    sigma = data["Cross Section (m^2)"].values
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
    return num / den if den != 0 else 0

def compute_incident_ion_energy(I_750, I_751, E_750, E_751, c):
    """Compute incident ion energy E_i as a weighted average."""
    if (I_750 + I_751) == 0:
        return 0
    return c * (I_750 * E_750 + I_751 * E_751) / (I_750 + I_751)

def compute_sputtering_yield(E_i, E_th, E_b, Lambda):
    """Compute sputtering yield: Y(E) = Lambda * (E_i - E_th) / E_b."""
    return Lambda * (E_i - E_th) / E_b

def compute_deposition_flux(Y, n):
    """Compute deposition flux: Φ_sub = Y * ∫_0^(π/2) cos^n(θ)*cos(θ) dθ."""
    def integrand(theta):
        return (np.cos(theta)**n) * np.cos(theta)
    val, _ = quad(integrand, 0, np.pi/2)
    return Y * val

def compute_nucleation_density(Phi_sub, D_0, E_d, T_s):
    """Compute nucleation density: N ~ Φ_sub / D_s, with D_s = D_0 exp(-E_d/(k_eV*T_s))."""
    D_s = D_0 * np.exp(-E_d / (k_eV * T_s))
    return Phi_sub / D_s

def compute_grain_size(N):
    """Compute mean grain size d_bar ~ 1/sqrt(N) and grain boundary length L_g ~ 1/d_bar."""
    d_bar = 1 / np.sqrt(N) if N > 0 else np.inf
    L_g = 1 / d_bar if d_bar > 0 else np.inf
    return d_bar, L_g

def generate_voronoi(N):
    """Generate and save a Voronoi tessellation based on nucleation sites."""
    num_points = int(np.clip(N, 10, 500))
    points = np.random.rand(num_points, 2)
    vor = Voronoi(points)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False,
                    line_colors='blue', line_width=2, line_alpha=0.6, point_size=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Voronoi Tessellation of Nucleation Sites")
    plt.tight_layout()
    save_fig("voronoi_tessellation")
    plt.close(fig)
    return vor

# ===============================
# Hybrid Time-Dependent Surface Diffusion & Nucleation (Spatial Binning)
# ===============================
@jit(nopython=True)
def assign_cells(positions, substrate_size, cell_size, max_per_cell):
    """
    Assign each adatom to a cell in a grid to reduce collision checks.
    """
    n_cells = int(substrate_size / cell_size)
    num_cells = n_cells * n_cells
    cell_counts = np.zeros(num_cells, dtype=np.int32)
    cell_list = -1 * np.ones((num_cells, max_per_cell), dtype=np.int32)
    N = positions.shape[0]
    for i in range(N):
        x = positions[i, 0]
        y = positions[i, 1]
        cx = int(x / cell_size)
        cy = int(y / cell_size)
        if cx >= n_cells:
            cx = n_cells - 1
        if cy >= n_cells:
            cy = n_cells - 1
        idx = cy * n_cells + cx
        cnt = cell_counts[idx]
        if cnt < max_per_cell:
            cell_list[idx, cnt] = i
            cell_counts[idx] = cnt + 1
    return cell_counts, cell_list, n_cells

@jit(nopython=True, parallel=True)
def simulate_surface_diffusion_hybrid(num_steps, dt, flux, D_s, r_c,
                                      substrate_size, cell_size,
                                      max_mobile, max_nuclei, max_per_cell):
    """
    Hybrid simulation (time-dependent) using spatial binning + diffusion + nucleation.
    """
    mobile_positions = np.empty((max_mobile, 2))
    nuclei_positions = np.empty((max_nuclei, 2))
    mobile_count = 0
    nuclei_count = 0
    sigma = np.sqrt(2 * D_s * dt)

    for step in range(num_steps):
        # Deposition step
        area = substrate_size * substrate_size
        expected_new = flux * dt * area
        n_new = int(expected_new + 0.5)
        for k in range(n_new):
            if mobile_count < max_mobile:
                mobile_positions[mobile_count, 0] = np.random.uniform(0, substrate_size)
                mobile_positions[mobile_count, 1] = np.random.uniform(0, substrate_size)
                mobile_count += 1

        # Diffusion step
        for i in prange(mobile_count):
            dx = sigma * np.random.randn()
            dy = sigma * np.random.randn()
            mobile_positions[i, 0] += dx
            mobile_positions[i, 1] += dy
            # Periodic boundaries
            if mobile_positions[i, 0] < 0:
                mobile_positions[i, 0] += substrate_size
            elif mobile_positions[i, 0] >= substrate_size:
                mobile_positions[i, 0] -= substrate_size
            if mobile_positions[i, 1] < 0:
                mobile_positions[i, 1] += substrate_size
            elif mobile_positions[i, 1] >= substrate_size:
                mobile_positions[i, 1] -= substrate_size

        # Spatial binning
        cell_counts, cell_list, n_cells = assign_cells(mobile_positions[:mobile_count, :],
                                                       substrate_size,
                                                       cell_size, max_per_cell)

        # Collision / nucleation checks
        for cell in range(n_cells * n_cells):
            count = cell_counts[cell]
            if count == 0:
                continue
            row = cell // n_cells
            col = cell % n_cells
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    nrow = row + dr
                    ncol = col + dc
                    if nrow < 0 or nrow >= n_cells or ncol < 0 or ncol >= n_cells:
                        continue
                    neighbor = nrow * n_cells + ncol
                    ncount = cell_counts[neighbor]
                    if ncount == 0:
                        continue
                    for idx1 in range(count):
                        i = cell_list[cell, idx1]
                        if i < 0 or i >= mobile_count:
                            continue
                        for idx2 in range(ncount):
                            j = cell_list[neighbor, idx2]
                            if j <= i or j < 0 or j >= mobile_count:
                                continue
                            dx = mobile_positions[i, 0] - mobile_positions[j, 0]
                            dy = mobile_positions[i, 1] - mobile_positions[j, 1]
                            # Periodic distance
                            if dx > substrate_size / 2:
                                dx -= substrate_size
                            elif dx < -substrate_size / 2:
                                dx += substrate_size
                            if dy > substrate_size / 2:
                                dy -= substrate_size
                            elif dy < -substrate_size / 2:
                                dy += substrate_size
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r_c:
                                # Merge -> nucleus
                                new_x = (mobile_positions[i, 0] + mobile_positions[j, 0]) / 2.0
                                new_y = (mobile_positions[i, 1] + mobile_positions[j, 1]) / 2.0
                                if nuclei_count < max_nuclei:
                                    nuclei_positions[nuclei_count, 0] = new_x
                                    nuclei_positions[nuclei_count, 1] = new_y
                                    nuclei_count += 1
                                # Mark them invalid
                                mobile_positions[i, 0] = -1.0
                                mobile_positions[j, 0] = -1.0

        # Compact to remove merged mobile adatoms
        new_mobile = 0
        for i in range(mobile_count):
            if mobile_positions[i, 0] >= 0:
                mobile_positions[new_mobile, :] = mobile_positions[i, :]
                new_mobile += 1
        mobile_count = new_mobile

    return (mobile_positions[:mobile_count, :],
            nuclei_positions[:nuclei_count, :],
            mobile_count, nuclei_count)

# ===============================
# Parameter Sweep (No JIT to avoid Numba typing conflicts)
# ===============================
power_range = np.linspace(10, 110, 11)
flow_range = np.linspace(10, 120, 12)

C_P = 1e16
C_Q = 1e20
α_exp = 0.05
T_e0 = 2.0
β_exp = 0.2
gamma_exp = -0.1
c_const = 1.5
E_th = 25.0
E_b = 3.5
Lambda_val = 0.05
D_0 = 1e-7
E_d = 0.5
T_s = 300
n_exp = 2

cs_file_751 = './crosssection/E_Ar_E_Ar2P5.csv'
cs_file_750 = './crosssection/E_Ar_E_Ar2P1.csv'
threshold_750 = 13.273
threshold_751 = 13.48

I750_arr = np.zeros((len(power_range), len(flow_range)))
I751_arr = np.zeros((len(power_range), len(flow_range)))
Y_arr = np.zeros((len(power_range), len(flow_range)))
d_bar_arr = np.zeros((len(power_range), len(flow_range)))
L_g_arr = np.zeros((len(power_range), len(flow_range)))
Phi_sub_arr = np.zeros((len(power_range), len(flow_range)))

def parameter_sweep(power_vals, flow_vals):
    I750_local = np.zeros((len(power_vals), len(flow_vals)))
    I751_local = np.zeros((len(power_vals), len(flow_vals)))
    Y_local = np.zeros((len(power_vals), len(flow_vals)))
    d_bar_local = np.zeros((len(power_vals), len(flow_vals)))
    L_g_local = np.zeros((len(power_vals), len(flow_vals)))
    Phi_sub_local = np.zeros((len(power_vals), len(flow_vals)))
    
    for i, P in enumerate(power_vals):
        for j, Q in enumerate(flow_vals):
            # Electron temperature
            T_e = T_e0 * (P ** β_exp) * (Q ** gamma_exp)
            # Densities
            n_e, n_Ar = compute_densities(P, Q, C_P, C_Q, α_exp)
            # Optical intensities
            R_exc_750 = compute_excitation_rate(n_e, n_Ar, cs_file_750, T_e, P, Q)
            R_exc_751 = compute_excitation_rate(n_e, n_Ar, cs_file_751, T_e, P, Q)
            I750 = compute_optical_emission(R_exc_750)
            I751 = compute_optical_emission(R_exc_751)
            # Effective energies & sputtering
            E_bar_750 = compute_effective_energy(cs_file_750, threshold_750, T_e)
            E_bar_751 = compute_effective_energy(cs_file_751, threshold_751, T_e)
            E_i = compute_incident_ion_energy(I750, I751, E_bar_750, E_bar_751, c_const)
            Y_val = compute_sputtering_yield(E_i, E_th, E_b, Lambda_val)
            Phi_sub = compute_deposition_flux(Y_val, n_exp)
            # Nucleation density & grain sizes
            N_val = compute_nucleation_density(Phi_sub, D_0, E_d, T_s)
            d_bar, L_g = compute_grain_size(N_val)
            
            I750_local[i, j] = I750
            I751_local[i, j] = I751
            Y_local[i, j] = Y_val
            Phi_sub_local[i, j] = Phi_sub
            d_bar_local[i, j] = d_bar
            L_g_local[i, j] = L_g
    return I750_local, I751_local, Y_local, Phi_sub_local, d_bar_local, L_g_local

I750_arr, I751_arr, Y_arr, Phi_sub_arr, d_bar_arr, L_g_arr = parameter_sweep(power_range, flow_range)

# ===============================
# Run Hybrid Time-Dependent Simulation
# ===============================
substrate_size = 1e-3
num_steps = 1000
dt = 0.01
flux = 1e14
D_s_value = D_0 * np.exp(-E_d / (k_eV * T_s))
r_c = 1e-6
cell_size = 10 * r_c
max_mobile = 5000
max_nuclei = 500
max_per_cell = 50

mobile_pos, nuclei_pos, mobile_count, nuclei_count = simulate_surface_diffusion_hybrid(
    num_steps, dt, flux, D_s_value, r_c, substrate_size,
    cell_size, max_mobile, max_nuclei, max_per_cell
)

# Save final nucleation data
nuclei_df = pd.DataFrame(nuclei_pos, columns=["x", "y"])
nuclei_df.to_csv("./montecarlo/nuclei_positions_hybrid.csv", index=False)

# If we have nuclei, do a Voronoi plot
if nuclei_count > 0:
    vor = Voronoi(nuclei_pos)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='green',
                    line_width=2, line_alpha=0.8, point_size=10)
    ax.set_xlim(0, substrate_size)
    ax.set_ylim(0, substrate_size)
    ax.set_title("Voronoi Tessellation of Nuclei (Hybrid Simulation)")
    plt.tight_layout()
    plt.savefig("./montecarlo/voronoi_tessellation_hybrid.jpeg", format='jpeg',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

# Plot final distributions of mobile adatoms and nuclei
fig, ax = plt.subplots()
ax.scatter(mobile_pos[:, 0], mobile_pos[:, 1], s=10, c='red', label="Mobile Adatoms")
if nuclei_count > 0:
    ax.scatter(nuclei_pos[:, 0], nuclei_pos[:, 1], s=30, c='blue', label="Nuclei")
ax.set_xlim(0, substrate_size)
ax.set_ylim(0, substrate_size)
ax.set_title("Final Adatom and Nuclei Positions (Hybrid)")
ax.legend()
plt.tight_layout()
plt.savefig("./montecarlo/final_positions_hybrid.jpeg", format='jpeg',
            dpi=300, bbox_inches='tight')
plt.close(fig)

print("Hybrid time-dependent surface diffusion simulation complete. Results saved in './montecarlo'.")





