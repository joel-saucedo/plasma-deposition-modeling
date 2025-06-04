import os
import csv
import requests
import xml.etree.ElementTree as ET
import re

def sanitize_filename(name):
    """
    Cleans up the reaction name for use as a CSV filename by:
    - Replacing special characters and spaces with underscores
    - Removing consecutive underscores
    - Trimming leading and trailing underscores
    """
    name = re.sub(r"[^\w\s]", "", name)  # Remove special characters
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores
    name = re.sub(r"_+", "_", name)  # Remove multiple consecutive underscores
    return name.strip("_")  # Remove leading/trailing underscores

def scrape_cross_section_data():
    """
    Scrapes the excitation cross section data from LXCat and saves each excitation process's data as a CSV file in the './crosssection' directory.
    
    The function performs the following steps:
      1. Downloads the XML data from the LXCat URL.
      2. Parses the XML to extract excitation cross section data for each process.
      3. For each process of type "Excitation", extracts the energy (DataX) and cross-section (DataY) values.
      4. Saves the extracted data as a CSV file in the './crosssection' directory.
    
    The CSV file contains two columns: "Energy (eV)" and "Cross Section (m^2)".
    """
    # URL of the XML file from LXCat
    url = "https://us.lxcat.net/cache/67c77bf3c7071/Cross%20section.xml"
    
    # Create the output directory if it doesn't exist
    output_dir = "./crosssection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the XML content
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    
    xml_content = response.text
    
    # Parse the XML
    root = ET.fromstring(xml_content)
    
    # Find all Process elements with attribute type="Excitation"
    # Navigate: root -> Database -> Groups -> Group -> Processes -> Process
    for database in root.findall(".//Database"):
        for group in database.findall(".//Group"):
            group_id = group.get("id", "unknown")
            processes = group.find("Processes")
            if processes is None:
                continue
            for process in processes.findall("Process"):
                # Check if this process is of type "Excitation"
                if process.get("type") != "Excitation":
                    continue

                # Extract reaction text (used to build file name)
                reaction_elem = process.find("Reaction")
                if reaction_elem is not None and reaction_elem.text:
                    reaction_text = reaction_elem.text.strip()
                else:
                    reaction_text = "unknown_reaction"
                
                # Clean and format the filename
                filename_safe = sanitize_filename(reaction_text) + ".csv"
                csv_filename = os.path.join(output_dir, filename_safe)
                
                # Extract energy data from the DataX element
                dataX_elem = process.find("DataX")
                if dataX_elem is None or not dataX_elem.text:
                    print(f"Skipping process '{reaction_text}' due to missing DataX.")
                    continue
                energy_values = dataX_elem.text.strip().split()
                # Convert string values to floats
                try:
                    energy_values = [float(val) for val in energy_values]
                except ValueError:
                    print(f"Could not convert energy values to float for process '{reaction_text}'.")
                    continue

                # Extract cross-section data from the DataY element
                dataY_elem = process.find("DataY")
                if dataY_elem is None or not dataY_elem.text:
                    print(f"Skipping process '{reaction_text}' due to missing DataY.")
                    continue
                xs_values = dataY_elem.text.strip().split()
                try:
                    xs_values = [float(val) for val in xs_values]
                except ValueError:
                    print(f"Could not convert cross-section values to float for process '{reaction_text}'.")
                    continue

                # Ensure the data lengths match
                if len(energy_values) != len(xs_values):
                    print(f"Data length mismatch in process '{reaction_text}': {len(energy_values)} energies vs {len(xs_values)} cross-section values.")
                    continue

                # Save the data to CSV
                with open(csv_filename, mode="w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Energy (eV)", "Cross Section (m^2)"])
                    for energy, xs in zip(energy_values, xs_values):
                        writer.writerow([energy, xs])
                
                print(f"Saved cross section data for process '{reaction_text}' to {csv_filename}")

if __name__ == "__main__":
    scrape_cross_section_data()











import os
import csv
import requests
import re

def sanitize_filename(name):
    """
    Cleans up the reaction name for use as a CSV filename by:
    - Replacing special characters and spaces with underscores
    - Removing consecutive underscores
    - Trimming leading and trailing underscores
    """
    name = re.sub(r"[^\w\s]", "", name)  # Remove special characters
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores
    name = re.sub(r"_+", "_", name)  # Remove multiple consecutive underscores
    return name.strip("_")  # Remove leading/trailing underscores

def scrape_differential_cross_section_data():
    """
    Scrapes the differential cross section data from the provided LXCat TXT file and saves it as a CSV.
    
    Steps:
      1. Downloads the TXT content from the given URL.
      2. Skips header lines until the dashed line that separates the header from the data.
      3. Parses the subsequent rows (each row containing: Angle (deg), Energy (eV), Differential cross section (m2/sr)).
      4. Saves the data into a CSV file in the './diff_crosssection' directory.
    """
    # URL of the differential cross section text file
    url = "https://us.lxcat.net/cache/67c9fedbb3b63/Differential%20cross%20section%20-%201.txt"
    
    # Create the output directory if it doesn't exist
    output_dir = "./diff_crosssection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the text content
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    
    content = response.text
    
    # Split content into lines
    lines = content.splitlines()
    
    # Identify the line that contains the dashed separator, which marks the beginning of the data table
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r"[-]{5,}", line):
            data_start = i + 1
            break
    
    if data_start is None:
        raise Exception("Could not find the data separator line in the file.")
    
    # Prepare CSV file path; you can choose a name based on process or a constant name
    csv_filename = os.path.join(output_dir, "differential_cross_section.csv")
    
    # Open the CSV file for writing
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row
        writer.writerow(["Angle (deg)", "Energy (eV)", "Differential cross section (m2/sr)"])
        
        # Process each data row
        for line in lines[data_start:]:
            # Skip empty lines or lines that do not contain data
            if not line.strip():
                continue
            # Split by whitespace or tab
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"Skipping incomplete line: {line}")
                continue
            try:
                # Convert the three columns to floats
                angle = float(parts[0])
                energy = float(parts[1])
                diff_xs = float(parts[2])
            except ValueError:
                print(f"Error converting line to floats: {line}")
                continue
            writer.writerow([angle, energy, diff_xs])
    
    print(f"Saved differential cross section data to {csv_filename}")

if __name__ == "__main__":
    scrape_differential_cross_section_data()



import os
import csv
import requests
import xml.etree.ElementTree as ET
import re

def sanitize_filename(name):
    """
    Cleans up the reaction name for use as a CSV filename by:
    - Replacing special characters and spaces with underscores
    - Removing consecutive underscores
    - Trimming leading and trailing underscores
    """
    name = re.sub(r"[^\w\s]", "", name)  # Remove special characters
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores
    name = re.sub(r"_+", "_", name)  # Remove multiple consecutive underscores
    return name.strip("_")  # Remove leading/trailing underscores

def scrape_and_combine_cross_section_data():
    """
    Scrapes the isotropic and backscattering cross section data for Ar+ -> Ar from LXCat,
    adds them together to obtain the total cross-section, and saves the result as a CSV file.
    """
    # URL of the XML file from LXCat
    url = "https://us.lxcat.net/cache/67d35a00174eb/Cross%20section.xml"
    
    # Create the output directory if it doesn't exist
    output_dir = "./crosssection"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the XML content
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    
    xml_content = response.text
    
    # Parse the XML
    root = ET.fromstring(xml_content)
    
    # Find all Process elements related to Ar+ -> Ar
    cross_section_data = {}
    for process in root.findall(".//Process"):
        reaction_elem = process.find("Reaction")
        if reaction_elem is None or "Ar^+ + Ar" not in reaction_elem.text:
            continue
        
        process_type = process.get("type", "Unknown")
        dataX_elem = process.find("DataX")
        dataY_elem = process.find("DataY")
        
        if dataX_elem is None or dataY_elem is None:
            continue
        
        energy_values = list(map(float, dataX_elem.text.strip().split()))
        xs_values = list(map(float, dataY_elem.text.strip().split()))
        
        if len(energy_values) != len(xs_values):
            print(f"Skipping {process_type} due to mismatched energy and cross-section data lengths.")
            continue
        
        if process_type not in cross_section_data:
            cross_section_data[process_type] = {"energy": energy_values, "cross_section": xs_values}
    
    # Ensure we have both isotropic and backscattering data
    if "Isotropic" not in cross_section_data or "Backscat" not in cross_section_data:
        raise ValueError("Missing either isotropic or backscattering cross-section data.")
    
    # Retrieve data and sum them element-wise
    energy = cross_section_data["Isotropic"]["energy"]  # Assume same energy grid for both
    xs_isotropic = cross_section_data["Isotropic"]["cross_section"]
    xs_backscat = cross_section_data["Backscat"]["cross_section"]
    xs_total = [iso + back for iso, back in zip(xs_isotropic, xs_backscat)]
    
    # Save the combined cross-section data to a CSV file
    csv_filename = os.path.join(output_dir, "Ar_plus_Ar_total_cross_section.csv")
    with open(csv_filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Energy (eV)", "Total Cross Section (m^2)"])
        for e, xs in zip(energy, xs_total):
            writer.writerow([e, xs])
    
    print(f"Saved total cross section data to {csv_filename}")

if __name__ == "__main__":
    scrape_and_combine_cross_section_data()















import os
import csv
import re
import requests
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import curve_fit

# -------------------------------
# Global Plot Settings
# -------------------------------
try:
    import scienceplots
    import matplotlib.pyplot as plt
    plt.style.use(['science', 'grid'])
except ImportError:
    import matplotlib.pyplot as plt
    plt.style.use('default')

import matplotlib as mpl
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

# Create necessary directories
if not os.path.exists('./results'):
    os.makedirs('./results')
if not os.path.exists('./crosssection'):
    os.makedirs('./crosssection')

def save_fig(filename):
    plt.savefig(f"./results/{filename}.jpeg", format='jpeg', bbox_inches='tight')

def sanitize_filename(name):
    """Clean a string to be used as a filename."""
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")

# Simple linear model for extrapolation: Q(n) = a*n + b
def linear_func(n, a, b):
    return a * n + b

def extrapolate_Q0(n_values, Q_values):
    """Fit a line to the moments Q(n) and extrapolate to n=0."""
    popt, _ = curve_fit(linear_func, n_values, Q_values)
    return linear_func(0, *popt)

def scrape_and_process_moments():
    url = "https://us.lxcat.net/cache/67d362e7ea4a5/Cross%20section.xml"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to retrieve data. Status code: {response.status_code}")
    xml_content = response.text
    root = ET.fromstring(xml_content)

    # Dictionaries to hold moments for Zn and O
    data_Zn = {}
    data_O = {}
    
    # Loop over all Process elements
    for proc in root.findall(".//Process"):
        reaction = proc.findtext("Reaction", default="").strip()
        # Check if reaction corresponds to Zn+ or O- (target species)
        if "Zn^+" in reaction:
            target = data_Zn
        elif "O^-" in reaction:
            target = data_O
        else:
            continue
        
        # Extract moment number from the type attribute (e.g., "Cross section moment Q(01)")
        proc_type = proc.get("type", "")
        m = re.search(r"Q\((\d+)\)", proc_type)
        if not m:
            continue
        moment = int(m.group(1))
        # We work only with moments 1-4.
        if moment < 1 or moment > 4:
            continue
        
        # Retrieve energy and cross section moment data
        dataX = proc.find("DataX")
        dataY = proc.find("DataY")
        if dataX is None or dataY is None:
            continue
        energies = list(map(float, dataX.text.strip().split()))
        moment_values = list(map(float, dataY.text.strip().split()))
        
        # Store energies once and the moment data under its label (e.g., "Q(01)")
        if "energy" not in target:
            target["energy"] = energies
        target[f"Q({moment:02d})"] = moment_values

    # Function to write CSV file for a target species
    def write_csv(target, species):
        energy = target.get("energy", [])
        # Gather moment keys (we assume they are Q(01) to Q(04))
        moment_keys = sorted([k for k in target.keys() if k.startswith("Q(") and k != "energy"])
        # Extrapolate Q(0) at each energy using available moments
        Q0_list = []
        n_indices = np.array([int(key[2:4]) for key in moment_keys])
        for i in range(len(energy)):
            Q_vals = np.array([target[key][i] for key in moment_keys])
            Q0_list.append(extrapolate_Q0(n_indices, Q_vals))
        # Write CSV file with columns: Energy, Q(00), Q(01), Q(02), Q(03), Q(04)
        filename = os.path.join("./crosssection", f"{sanitize_filename(species)}_cross_section_moments.csv")
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            header = ["Energy (Hartee)", "Q(00)"] + moment_keys
            writer.writerow(header)
            for i, E in enumerate(energy):
                row = [E, Q0_list[i]] + [target[key][i] for key in moment_keys]
                writer.writerow(row)
        print(f"Saved {species} moments to {filename}")

    if data_Zn:
        write_csv(data_Zn, "Zn")
    if data_O:
        write_csv(data_O, "O")

if __name__ == "__main__":
    scrape_and_process_moments()










import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Conversion factors:
hartree_to_eV = 27.2114          # 1 Hartree = 27.2114 eV
bohr2_to_m2 = 2.800285e-21       # 1 Bohr^2 = 2.800285e-21 m^2

# Path to the CSV file (change "Zn_cross_section_moments.csv" to "O_cross_section_moments.csv" as needed)
csv_file = "./crosssection/Zn_cross_section_moments.csv"

# Read the CSV file using pandas
data = pd.read_csv(csv_file)

# Convert energy from Hartree to eV (assumes the CSV column is named "Energy (Hartee)")
data["Energy (eV)"] = data["Energy (Hartee)"] * hartree_to_eV

# Define the moment columns (we assume these are labeled as Q(00), Q(01), Q(02), Q(03), Q(04))
moment_cols = ["Q(00)", "Q(01)", "Q(02)", "Q(03)", "Q(04)"]

# Convert each moment from Bohr^2 to m^2
for col in moment_cols:
    data[col] = data[col] * bohr2_to_m2

# Plot the moments vs. energy on a log-scale for the x-axis
plt.figure()
for col in moment_cols:
    plt.plot(data["Energy (eV)"], data[col], marker='o', label=col)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Energy (eV)")
plt.ylabel("Cross Section Moment (m$^2$)")
plt.title("Cross Section Moments for Zn (Converted to SI Units)")
plt.legend(title="Moments")
plt.tight_layout()
plt.savefig("./results/Zn_cross_section_moments_plot.jpeg", format="jpeg", bbox_inches="tight")
plt.close()

print("Plot saved in the './results' directory.")

