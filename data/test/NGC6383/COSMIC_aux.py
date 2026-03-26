import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pymc as pm
import astropy.units as u
from astropy.constants import G
import arviz as az
import pymc.sampling_jax as pm_jax
import warnings
import seaborn as sns
from astropy.coordinates import SkyCoord,angular_separation,Galactic
from astropy.table import QTable, join
from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.visualization import quantity_support
from astropy.visualization.wcsaxes import add_scalebar
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal, norm, ks_2samp, gamma
from astropy.stats import sigma_clip
from scipy.spatial import KDTree
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm
import os
from scipy.spatial import KDTree
from datetime import datetime
import xarray as xr


# Correct proper motion to align with ICRF
def edr3ToICRF(pmra ,pmdec ,ra ,dec ,G):
    if G >=13:
        return pmra , pmdec
    def sind (x):
        return np.sin(np. radians (x))
    def cosd (x):
        return np.cos(np. radians (x))
    table1 =""" 0.0 9.0 18.4 33.8 -11.3
                9.0 9.5 14.0 30.7 -19.4
                9.5 10.0 12.8 31.4 -11.8
                10.0 10.5 13.6 35.7 -10.5
                10.5 11.0 16.2 50.0 2.1
                11.0 11.5 19.4 59.9 0.2
                11.5 11.75 21.8 64.2 1.0
                11.75 12.0 17.7 65.6 -1.9
                12.0 12.25 21.3 74.8 2.1
                12.25 12.5 25.7 73.6 1.0
                12.5 12.75 27.3 76.6 0.5
                12.75 13.0 34.9 68.9 -2.9 """
    table1 = np.fromstring(table1,sep=' ').reshape((12,5)).T
    Gmin = table1[0]
    Gmax = table1[1]
    # pick the appropriate omegaXYZ for the source ’s magnitude :
    omegaX = table1[2][(Gmin <=G)&(Gmax>G)][0]
    omegaY = table1[3][(Gmin <=G)&(Gmax>G)][0]
    omegaZ = table1[4][(Gmin <=G)&(Gmax>G)][0]
    pmraCorr = -1*sind(dec)*cosd(ra)*omegaX-sind(dec)*sind(ra)*omegaY + cosd(dec)*omegaZ
    pmdecCorr = sind(ra)*omegaX-cosd(ra)*omegaY
    return pmra - pmraCorr/1000. , pmdec - pmdecCorr/1000.
def add_photometric_errors(table):
    """
    Adds photometric errors for Gmag, G_BPmag, G_RPmag, and e_bp_rp to the given QTable,
    incorporating astropy units (u.mag) for proper handling of magnitudes, with errors expressed in milli-magnitudes.

    Parameters:
    - table: QTable, must contain columns 'Gmag', 'G_BPmag', and 'G_RPmag' for G-band magnitudes.

    The function modifies the table in place by adding four new columns for the errors:
    - 'e_Gmag', 'e_G_BPmag', 'e_G_RPmag', 'e_BP_RP', all expressed in magnitudes.
    """
    # Define error functions for g_mag, g_bp, g_rp with corrections for units
def g_mag_error(G):
    G_val = G.to(u.mag).value  # Ensure G is in magnitudes and get its value
    if G_val < 13:
        return 0.3 / 1000 * u.mag  # Convert mmag to mag
    elif G_val < 17:
        return np.interp(G_val, [13, 17], [0.3, 1]) / 1000 * u.mag
    elif G_val <= 20:
        return np.interp(G_val, [17, 20], [1, 6]) / 1000 * u.mag
    else:
        return 6 / 1000 * u.mag

def g_bp_error(G):
    G_val = G.to(u.mag).value
    if G_val < 13:
        return 0.9 / 1000 * u.mag
    elif G_val < 17:
        return np.interp(G_val, [13, 17], [0.9, 12]) / 1000 * u.mag
    elif G_val <= 20:
        return np.interp(G_val, [17, 20], [12, 108]) / 1000 * u.mag
    else:
        return 108 / 1000 * u.mag

def g_rp_error(G):
    G_val = G.to(u.mag).value
    if G_val < 13:
        return 0.6 / 1000 * u.mag
    elif G_val < 17:
        return np.interp(G_val, [13, 17], [0.6, 6]) / 1000 * u.mag
    elif G_val <= 20:
        return np.interp(G_val, [17, 20], [6, 52]) / 1000 * u.mag
    else:
        return 52 / 1000 * u.mag
def ensure_units(value, default_unit):
    """Ensure the given value has specific units, or assign default units."""
    if not hasattr(value, 'unit'):
        return value * default_unit
    return value
def angular_size(linear_size,distance):
    distance = ensure_units(distance,u.kpc)
    angular = (linear_size / distance).to(u.deg, equivalencies=u.dimensionless_angles())
    return angular.to(u.arcmin)
def linear_size(angular_size,distance):
    distance = ensure_units(distance,u.kpc)
    linear = (angular_size * distance).to(u.pc, equivalencies=u.dimensionless_angles())
    return linear.to(u.pc)
def histogram_mode(data,axis=None):
    """
    Compute the mode of a dataset based on its histogram.

    Parameters:
    - data: array-like, the dataset from which to compute the mode.

    Returns:
    - mode_value: The value with the highest count in the dataset.
    """
    # Compute the histogram of the data
    hist, bin_edges = np.histogram(data, bins='auto')
    
    # Find the index of the maximum count in the histogram
    max_index = np.argmax(hist)
    
    # Compute the mode as the average of the bin edges corresponding to the maximum count
    mode_value = 0.5 * (bin_edges[max_index] + bin_edges[max_index + 1])
    
    return mode_value
def half_mass_radius(data, centers, prob_number=70,distance=None):
    """
    Calculate the half-mass radius of a star cluster.

    Parameters:
    data (Table): Table of star data including 'mass' and 'probability'.
    center (tuple): Center of the cluster as a tuple (ra, dec).
    prob_number (float): Probability threshold to filter the data.

    Returns:
    float: The half-mass radius of the cluster in arcminutes.
    """
    # Filter data based on probability threshold
    centers = [SkyCoord(ra=centers[0], dec=centers[1], frame='icrs', unit='deg')]
    prob_threshold = prob_number / 100
    filtered_data = data[data['probability'] >= prob_threshold]
    # Calculate distances to the cluster center
    filtered_data['d_center'] = angular_separation(filtered_data['ra'], filtered_data['dec'], centers[0].ra, centers[0].dec)
    
    # Sort by distance from the center
    sorted_indices = np.argsort(filtered_data['d_center'])
    filtered_data = filtered_data[sorted_indices]

    # Calculate cumulative mass and find the half total mass
    cumulative_mass = np.cumsum(filtered_data['mass'])
    half_total_mass = np.max(cumulative_mass) / 2

    # Find the half-mass radius
    half_mass_index = np.searchsorted(cumulative_mass, half_total_mass)
    half_mass_radius = filtered_data['d_center'][half_mass_index]
    if distance:
        return half_mass_radius.to(u.arcmin),linear_size(half_mass_radius.to(u.arcmin),distance)
    else:
        return half_mass_radius.to(u.arcmin)
def read_isochrones_with_metadata(file_path,columns_type='DR3'):
    metadata = []
    isochrones = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data_start_index = next(i for i, line in enumerate(lines) if line.startswith("# Zini"))
    metadata = lines[:data_start_index]
    separator_indices = [index for index, line in enumerate(lines[data_start_index:]) if line.startswith("#")]
    separator_indices.append(len(lines))
    if columns_type == 'DR3':
        columns = ["Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL", "logTe",
    "logg", "label", "McoreTP", "C_O", "period0", "period1", "period2",
    "period3", "period4", "pmode", "Mloss", "tau1m", "X", "Y", "Xc", "Xn",
    "Xo", "Cexcess", "Z", "mbolmag", "Gmag", "G_BPmag", "G_RPmag"]
    if columns_type == '2MASS':
        columns = ["Zini", "MH", "logAge", "Mini", "int_IMF", "Mass", "logL", "logTe",
    "logg", "label", "McoreTP", "C_O", "period0", "period1", "period2",
    "period3", "period4", "pmode", "Mloss", "tau1m", "X", "Y", "Xc", "Xn",
    "Xo", "Cexcess", "Z", "mbolmag", "Jmag", "Hmag", "Ksmag"]
    for i in range(len(separator_indices) - 1):
        start_index = data_start_index + separator_indices[i] + 1
        end_index = data_start_index + separator_indices[i + 1]
        isochrone_data = lines[start_index:end_index]
        df = pd.DataFrame([x.strip().split() for x in isochrone_data if x.strip()], columns = columns)
        isochrones.append(df)
    return metadata, isochrones

def set_column_types(isochrones, columns):
    dtype_conversion = {column: float for column in columns}
    
    for df in isochrones:
        for column in dtype_conversion:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors='coerce')
    return isochrones

# Previous functions definitions (read_isochrones_with_metadata and set_column_types) are assumed to be defined here

# Stage labels mapping based on your provided categories
stage_labels = {
    0: 'PMS',
    1: 'MS',
    2: 'SGB',
    3: 'RGB',
    4: 'CHEB Stage 1',
    5: 'CHEB Stage 2',
    6: 'CHEB Stage 3',
    7: 'EAGB',
    8: 'TPAGB',
    9: 'post-AGB'
}
def plot_isochrone(isochrones, logAge, Zini, color_row, mag_row, ax, dm=0, A_V=0, R_V=3.1, color='blue', alpha=0.4, linestyle='--', label=True, type='plot', yl=None):
    filtered_isochrone = isochrones.query(f"logAge == {logAge} and Zini == {Zini}").copy()
    
    # Calculate E(B-V) from A_V using the provided R_V value
    ext = A_V / R_V
    
    mass_data = []
    
    if not filtered_isochrone.empty:
        # Apply extinction and distance modulus corrections
        filtered_isochrone['corrected_mag'] = filtered_isochrone[mag_row] + dm + A_V
        filtered_isochrone['corrected_color'] = filtered_isochrone[color_row] + ext

        # Filter by luminosity if yl is specified
        if yl:
            filtered_isochrone = filtered_isochrone[filtered_isochrone['corrected_mag'] >= 0.95*yl[0]]
        # Plotting
        if type == 'plot':
            ax.plot(filtered_isochrone['corrected_color'], filtered_isochrone['corrected_mag'], lw=3, color=color, alpha=alpha, linestyle=linestyle, label=rf'$\log(age/yr)={logAge}$' if label else None)
        elif type == 'scatter':
            ax.scatter(filtered_isochrone['corrected_color'], filtered_isochrone['corrected_mag'], color=color, alpha=alpha, marker=linestyle, label=rf'$\log(age/yr)={logAge}$' if label else None)
    else:
        print(f"No isochrone found for logAge={logAge} and Z={Zini}.")

def plot_color_color(isochrone_1,isochrone_2, logAge, Zini, color_1_row, color_2_row, ax, A_V=0, R_V=3.1, color='blue', alpha=0.4, linestyle='--'):
    filtered_isochrone_1 = isochrone_1.query(f"logAge == {logAge} and Zini == {Zini}").copy()
    filtered_isochrone_2 = isochrone_2.query(f"logAge == {logAge} and Zini == {Zini}").copy()
    # Calculate E(B-V) from A_V using the provided R_V value
    ext = A_V / R_V
    
    if not (filtered_isochrone_1.empty and filtered_isochrone_2.empty):
        # Plot the entire isochrone without grouping by 'label'
        ax.plot(filtered_isochrone_1[color_1_row] + ext, filtered_isochrone_2[color_2_row] + ext, lw=3, c=color, alpha=alpha, linestyle=linestyle,label=rf'logAge={logAge}')
    else:
        print(f"No isochrone found for logAge={logAge} and Z={Zini}.")

def plot_isochrone_label(isochrones, logAge, Zini,color_row,mag_row, ax, dm=0, A_V=0, R_V=3.1,color='blue',alpha=0.4,linestyle='--'):
    filtered_isochrone = isochrones.query(f"logAge == {logAge} and Zini == {Zini}").copy()
    
    # Calculate A_V from E(B-V) using the provided R_V value
    #A_V = R_V * ext  # A_V = R_V * E(B-V)
    ext = A_V/R_V
    # Map the 'label' to a string representation, if stage_labels is defined
    filtered_isochrone.loc[:, 'stage'] = filtered_isochrone['label'].map(stage_labels)
    
    if not filtered_isochrone.empty:
        # Create a scatter plot for each stage
        for label, group in filtered_isochrone.groupby('label'):
            stage = stage_labels.get(label, 'Unknown')  # Get the stage name; default to 'Unknown' if not found
            # Apply distance modulus (dm), total V-band absorption (A_V), and reddening (ext) to colors and magnitudes
            ax.plot(group[color_row] + ext, group[mag_row] + dm + A_V, lw=3,c=color,alpha=alpha,linestyle=linestyle)
    
    else:
        print(f"No isochrone found for logAge={logAge} and Z={Zini}.")

def plot_errors_bar(magnitudes,colors,magnitude_errors, color_errors, ax,loc=None):
    """
    Plot the median magnitude error and mean color error within bins of 2 mag on a CMD.
    
    Parameters:
    - magnitudes: QTable column or Quantity array, magnitudes of the stars/objects with units
    - magnitude_errors: QTable column or Quantity array, corresponding errors of the magnitudes with units
    - color_errors: QTable column or Quantity array, corresponding color errors with units
    - ax: matplotlib.axes.Axes, the axes on which to plot the graph
    - fixed_color_value: Quantity, the fixed color value where to plot the error bars
    """
    magnitudes = magnitudes.value
    magnitude_errors = magnitude_errors.value
    color_errors = color_errors.value
    # Determine the range of magnitudes and create bins every 2 magnitudes
    min_mag = np.floor(np.nanmin(magnitudes))
    max_mag = np.ceil(np.nanmax(magnitudes))
    bins = np.arange(min_mag, max_mag + 2, 2)
    
    median_mag_errors = []
    mean_color_errors = []
    # Calculate the median magnitude error and mean color error for each bin
    for i in range(len(bins) - 1):
        in_bin = (magnitudes >= bins[i]) & (magnitudes < bins[i+1])
        median_result = np.median(magnitude_errors[in_bin])
        mean_result = np.mean(color_errors[in_bin])

        # Check if the result is a masked array and append appropriately
        if hasattr(median_result, 'mask'):
            # Append unmasked value if masked; handle differently if needed
            median_mag_errors.append(median_result.tolist())
        else:
            median_mag_errors.append(median_result)
        if hasattr(mean_result, 'mask'):
            mean_color_errors.append(mean_result.tolist())
        else:
            mean_color_errors.append(mean_result)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if loc is None:
        loc = 0.5*np.nanmin(colors)
    ax.errorbar([loc.value] * len(bin_centers), bin_centers,
                xerr=mean_color_errors, yerr=median_mag_errors, 
                fmt='none',color='k', ecolor='k')
def plot_iso_mass_curves_across_isochrones(isochrones, logAge_range, Zini, ax, dm=0, A_V=0, R_V=3.1, color='black', alpha=1, linestyle='-', mass_values=None, zorder=1):
    # Initialize a dictionary to store color and magnitude for each mass across isochrones
    iso_mass_data = {mass: {'colors': [], 'magnitudes': []} for mass in mass_values}
    
    ext = A_V / R_V  # Calculate E(B-V) from A_V using the provided R_V value
    
    # Get sorted unique logAge values within the specified range and for the given Zini
    unique_logAges = isochrones[(isochrones['Zini'] == Zini) & (isochrones['logAge'].between(*logAge_range))]['logAge'].unique()
    sorted_logAges = np.sort(unique_logAges)
    
    for logAge in sorted_logAges:
        specific_isochrone = isochrones[(isochrones['logAge'] == logAge) & (isochrones['Zini'] == Zini)]
        
        # For each mass, find the first appearance in this specific isochrone by rounding
        for mass in mass_values:
            # Round the 'Mass' column for comparison and find the first match
            rounded_masses = specific_isochrone['Mass'].round(1)
            mass_points = specific_isochrone[(rounded_masses == mass)]
            
            if not mass_points.empty:
                first_point = mass_points.head(1)  # Taking the first appearance if multiple
                corrected_color = first_point['BP_RP'].iloc[0] + ext
                corrected_magnitude = first_point['Gmag'].iloc[0] + dm + A_V
                
                iso_mass_data[mass]['colors'].append(corrected_color)
                iso_mass_data[mass]['magnitudes'].append(corrected_magnitude)
    
    # Plotting lines connecting the first appearances of each mass across isochrones
    for mass, data in iso_mass_data.items():
        if data['colors'] and data['magnitudes']:
            # Plot without sorting as we're assuming chronological order by logAge
            ax.plot(data['colors'], data['magnitudes'], label=f'Mass = {mass} $M_\odot$', color=color, alpha=alpha, linestyle=linestyle, zorder=zorder)
from arviz import kde

def calculate_mode(values, bw="default", circular=False):
    if values.dtype.kind == "f":  # Check if values are floating-point
        if bw == "default":
            bw = "taylor" if circular else "experimental"
        x, density = kde(values, circular=circular, bw=bw)
        point_value = x[np.argmax(density)]
    else:
        point_value = int(mode(values).mode)
    return point_value

def store_trace_results(trace, excluded_parameters=['sigma', 'likelihood', 'likelihood_unobserved'], file_path='fit_parameters.csv', save_trace=False, trace_index=None):
    if trace_index is None:
        trace_index = int(datetime.now().timestamp())

    fit_params_mean = {
        i: np.round(trace.posterior[i].mean(dim=("chain", "draw")).data.tolist(), 3)
        for i in trace.posterior if i not in excluded_parameters
    }
    fit_params_median = {
        i: np.round(trace.posterior[i].median(dim=("chain", "draw")).data.tolist(), 3)
        for i in trace.posterior if i not in excluded_parameters
    }
    fit_params_stds = {
        i: np.round(trace.posterior[i].std(dim=("chain", "draw")).data.tolist(), 3)
        for i in trace.posterior if i not in excluded_parameters
    }

    fit_params_mode = {}
    for i in trace.posterior:
        if i not in excluded_parameters:
            stacked = trace.posterior[i].stack(sample=("chain", "draw"))
            mode_value = calculate_mode(stacked.values)
            fit_params_mode[i] = np.round(mode_value, 3).tolist()

    new_data = pd.DataFrame({
        'Parameter': fit_params_mean.keys(),
        'Mean': fit_params_mean.values(),
        'Median': fit_params_median.values(),
        'Std': fit_params_stds.values(),
        'Mode': fit_params_mode.values(),
        'Date_Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * len(fit_params_mean),
        'Trace_Index': [int(trace_index)] * len(fit_params_mean)
    })

    # Ensure the Trace_Index column is of integer type
    new_data['Trace_Index'] = new_data['Trace_Index'].astype(int)

    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        existing_data['Trace_Index'] = existing_data['Trace_Index'].astype(int)  # Convert existing Trace_Index to int
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data

    updated_data.to_csv(file_path, index=False)
    print("Data stored successfully.")

    if save_trace:
        trace_file_path = file_path.replace('.csv', f'_trace_{trace_index}.nc')
        trace.to_netcdf(trace_file_path)
        print(f"Trace data stored successfully in {trace_file_path}.")
def load_results(file_path='fit_parameters.csv', load_trace=False, only_last=True):
    # Define the path for the CSV file
    results_path = file_path
    
    # Load summarized statistics from CSV
    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        if only_last:
            # Convert 'Date_Time' to datetime type and select the last entry
            results['Date_Time'] = pd.to_datetime(results['Date_Time'])
            max_date = results['Date_Time'].max()
            results = results[results['Date_Time'] == max_date]
            trace_index = results.iloc[0]['Trace_Index']  # Get trace index from the last results
            trace_path = file_path.replace('.csv', f'_trace_{trace_index}.nc')  # Use index to specify trace file
        else:
            # Include handling to load traces for all entries if needed, or specify otherwise
            trace_path = None
        print("Summarized results loaded successfully.")
    else:
        print("No summarized results file found.")
        results = None

    # Optionally load the full trace from NetCDF using the derived trace index
    trace = None
    if load_trace and trace_path:
        if os.path.exists(trace_path):
            trace = az.from_netcdf(trace_path)
            print("Full trace data loaded successfully.")
        else:
            print(trace_path)
            print("No trace file found for the given index.")

    return results, trace
def assign_masses(isochrones,mag_column,color_column,ID, k=5):
    """
    Assigns masses to stars based on the nearest isochrones points using a KDTree.

    Args:
    - isochrones (list of tuples of arrays): List where each tuple represents one isochrone (mag, color1, color2, mass_ini).
    - stars (QTable): Table of stars with columns 'ID', 'color1', and 'magnitude'.
    - k (int): Number of nearest neighbors to consider for std calculation.

    Returns:
    - QTable containing star ID, assigned mass, and std of the mass.
    """
    # Prepare isochrone data for KDTree
    isochrone_points = []
    masses = []

    for iso in isochrones:
        for mag, color1, mass in zip(iso[0], iso[1], iso[3]):
            isochrone_points.append([color1, mag])
            masses.append(mass)
    isochrone_points = np.array(isochrone_points)
    masses = np.array(masses)
    
    # Prepare star data
    star_points = np.vstack([color_column,mag_column]).T
    
    # Create KDTree
    tree = KDTree(isochrone_points)
    
    # Query the tree for each star to find the k nearest isochrones
    distances, indices = tree.query(star_points, k=k)
    
    # Calculate the mean and std of the masses of the nearest isochrones
    mean_masses = np.array([np.mean(masses[idx]) for idx in indices])
    std_masses = np.array([np.std(masses[idx]) for idx in indices])
    
    # Create output QTable
    assigned_masses_table = QTable()
    assigned_masses_table['source_id'] = ID
    assigned_masses_table['mass'] = mean_masses
    assigned_masses_table['mass_std'] = std_masses
    
    return assigned_masses_table
def plot_hist2d(synthcl,param_means, param_stds, ax, n_samples=100,cmin=50,cmap = ListedColormap(plt.get_cmap('gray_r')(np.linspace(0, 1, 69))),alpha=0.8,bins=None,return_masses=False):
    """
    Plot isochrones on the given axes based on parameter means and standard deviations,
    and overlay a 2D histogram using Bayesian Blocks for bin determination.

    Args:
    - param_means (dict): Dictionary of parameter means.
    - param_stds (dict): Dictionary of parameter standard deviations.
    - ax (matplotlib.axes.Axes): The axes object where the isochrones will be plotted.
    - n_samples (int): Number of isochrones to generate.
    """
    all_isochrones = []
    # Generate and plot isochrones
    for _ in range(n_samples):
        # Sample parameters from their respective normal distributions
        sampled_params = {k: norm.rvs(loc=param_means[k], scale=param_stds[k]) for k in param_means.keys()}
        
        # Generate isochrone
        isochrone = synthcl.generate(sampled_params,plot_flag=True)
        if return_masses:
            all_isochrones.append((isochrone[0], isochrone[1],isochrone[2], isochrone[3]))
        else:
            all_isochrones.append((isochrone[0], isochrone[1],isochrone[2]))
    # Create color and magnitude arrays from isochrones
    colors = np.concatenate([iso[1] for iso in all_isochrones])
    mags = np.concatenate([iso[0] for iso in all_isochrones])
    if bins is None:
        _, color_bins = knuth_bin_width(colors[~np.isnan(colors)], return_bins=True)
        _, mag_bins = knuth_bin_width(mags[~np.isnan(mags)], return_bins=True)
        bins = [color_bins,mag_bins]
    _, _, _, quadmesh = ax.hist2d(colors, mags, bins=bins, cmap=cmap, norm=LogNorm(), zorder=-1, alpha=alpha,cmin=cmin)
    if return_masses:
        return all_isochrones
    else:
        return quadmesh
def assign_mass_nearest_isochrone_point_kdtree(isochrones, stars, logAge_range, Zini, color_column, magnitude_column, dm, A_V):
    """
    Modifies the function to return an Astropy Table that can be joined with the original stars table.
    The returned table includes `source_id` and the assigned masses, enabling a direct join.
    """
    # Ensure isochrones is a compatible format (e.g., pandas DataFrame)
    # Filter isochrones by metallicity and age range
    filtered_isochrones = isochrones[(isochrones['Zini'] == Zini) &
                                     (isochrones['logAge'].between(*logAge_range))]
    corrected_magnitude_isochrones = filtered_isochrones[magnitude_column] + dm + A_V

    # Prepare the data for KDTree: isochrone points (color, corrected magnitude)
    isochrone_points = np.vstack([filtered_isochrones[color_column], corrected_magnitude_isochrones]).T
    isochrone_masses = filtered_isochrones['Mass'].values
    
    # Build a KDTree for efficient nearest neighbor search
    tree = KDTree(isochrone_points)
    # Prepare stars data for KDTree query
    stars_features = np.vstack([stars[color_column], stars[magnitude_column]]).T
    
    # Query the KDTree for the nearest isochrone point for each star
    _, indices = tree.query(stars_features)
    
    # Assign the mass of the nearest isochrone point to each star
    assigned_masses = isochrone_masses[indices]*u.Msun
    
    # Create an Astropy Table to return, including source_id and assigned masses
    result_table = QTable([stars['designation'], assigned_masses], names=('designation', 'mass'))
    
    return result_table
def distance_model(data, return_trace=False, progressbar=False,prior_type='uniform'):
    distances = data['r_med_geo']
    parallax = data['parallax']
    prior_mu_r = np.mean([np.mean(parallax).to(u.kpc, equivalencies=u.parallax()).value,(np.mean(1 / parallax)).value,np.mean(data['r_med_geo'].value)])
    with pm.Model() as distance_model:
        # Hyperprior for the mean distance to the star cluster
        if prior_type == 'uniform':
            mu_r = pm.Uniform("mu_r", lower=0.5*prior_mu_r, upper=1.5*prior_mu_r)
        elif prior_type == 'normal':
            mu_r = pm.TruncatedNormal("mu_r", lower=0.5*prior_mu_r, upper=1.5*prior_mu_r,mu=prior_mu_r,sigma=1)
        std_r = pm.HalfNormal("std_r", sigma=np.std(data['r_med_geo']))
        # Model for the distances to each star
        r = pm.Gamma("r", mu=mu_r, sigma=std_r, shape=len(data),observed=distances)
        # Likelihood for observed parallaxes, considering observational errors

    # Iterative sampling process
    max_rhat = 2.0
    target_accept = 0.95
    tune = 4000

    while max_rhat > 1.0:
        with distance_model:
            trace = pm.sample(draws=20000, tune=tune, target_accept=target_accept, progressbar=progressbar,nuts_sampler='numpyro')
            
            # Using ArviZ to compute rhat values
            rhat_vals = az.rhat(trace)
            # Extract the maximum rhat value
            max_rhat = np.round(np.nanmax(rhat_vals.to_array()), 2)
            
            if target_accept <= 0.9999999:
                target_accept += 0.05
            tune += 2000

    results = {
        'mu_r_mean': trace.posterior['mu_r'].mean().item(),
        'std_r_mean': trace.posterior['std_r'].mean().item(),
        'mu_r_std': trace.posterior['mu_r'].std().item(),
        'std_r_std': trace.posterior['std_r'].std().item(),
    }

    if return_trace:
        results['trace_distance'] = trace
    return results
def fit_parallax_model(data, return_trace=False, progressbar=False,prior_distance=None,prior_type='uniform'):
    with pm.Model() as plx_model:
        # Prior for the mean true parallax of the cluster
        if prior_type == 'uniform':
            if prior_distance:
                mu_parallax = pm.Uniform("mu_parallax", lower=0.5*(1/prior_distance),upper=1.5*(1/prior_distance))
            else:
                mu_parallax = pm.Uniform("mu_parallax", lower=0.5*np.mean(data['parallax']),upper=1.5*np.mean(data['parallax']))
        if prior_type == 'normal':
            if prior_distance:
                mu_parallax = pm.TruncatedNormal("mu_parallax", lower=0.5*(1/prior_distance),upper=1.5*(1/prior_distance),sigma=1,mu=(1/prior_distance))
            else:
                mu_parallax = pm.TruncatedNormal("mu_parallax", lower=0.5*np.mean(data['parallax']),upper=1.5*np.mean(data['parallax']),sigma=1,mu=np.mean(data['parallax']))
        # Prior for the standard deviation of the true parallaxes within the cluster
        sigma_parallax = pm.HalfNormal("sigma_parallax", sigma=np.std(data['parallax']))
        
        # Likelihood for observed parallaxes, considering observational errors
        observed_parallax = pm.Normal("observed_parallax", mu=mu_parallax, 
                                      sigma=sigma_parallax, observed=data["parallax"].value)
        # Sampling
    max_rhat = 2.0
    target_accept = 0.8
    tune = 4000
    while max_rhat > 1.0:
        with plx_model:
            trace = pm.sample(draws=20000, tune=tune, target_accept=target_accept, progressbar=progressbar,nuts_sampler='numpyro')
            
            # Using ArviZ to compute rhat values
            rhat_vals = az.rhat(trace)
            # Extract the maximum rhat value
            max_rhat = np.round(np.nanmax(rhat_vals.to_array()), 2)
            
            if target_accept <= 0.9999999:
                target_accept += 0.05
            tune += 2000
    results = {
        'mu_parallax_mean': np.mean(trace.posterior['mu_parallax'].values),
        'sigma_parallax_mean': np.mean(trace.posterior['sigma_parallax'].values),
        'mu_parallax_std': np.std(trace.posterior['mu_parallax'].values),
        'sigma_parallax_std': np.std(trace.posterior['sigma_parallax'].values),
    }
    if return_trace:
        results['trace_parallax'] = trace
    return results
def parallax_determination(data, prob_thresholds=[50, 60, 70, 80], return_trace=False, progressbar=False, savefig=None, paper_single=False, parallax_hist=True,distance_prior=None):
    prob_number = np.array(prob_thresholds) / 100
    num_types = 2 if paper_single else 3  # Adjust the number of types of graphics based on 'paper_single'
    
    # Adjust num_rows if parallax_hist is True
    num_rows = num_types + (1 if parallax_hist else 0)
    # Determine the grid layout
    num_cols = len(prob_number)  # Number of columns is equal to the number of thresholds
    
    # Create the subplots with dynamic layout
    if paper_single:
        figsize = (18,5.5)
    else:
        figsize = (6 * num_cols, 4.5 * num_rows)
    if paper_single:
        fig, axes = plt.subplots(num_cols,num_rows, figsize=figsize, constrained_layout=True)
    else:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, constrained_layout=True)
    # Ensure axes is a 2D array for consistency in indexing
    axes = np.atleast_2d(axes)
    
    # If paper_single, adjust axes for a single plot type
    results_distance = {
         'mu_r_mean': [],
        'std_r_mean': [],
        'mu_r_std': [],
        'std_r_std': [],
    }
    results_parallax = {
        'mu_parallax_mean': [],
        'sigma_parallax_mean': [],
        'mu_parallax_std': [],
        'sigma_parallax_std': [],
    }
    all_traces = []
    min_data = (data['probability'] >= np.nanmin(prob_number))
    data = data[min_data]
    for j, prob_threshold in enumerate(prob_number):
        print(f"{prob_threshold * 100}%")
        # Filter data based on probability threshold
        filtered_data = data[data['probability'] >= prob_threshold]
        error_criteria = (filtered_data['parallax_error']/filtered_data['parallax']) <= 0.1
        useful_data = filtered_data[error_criteria]
        nonuseful_data = filtered_data[~error_criteria]
        print('Distance sampling')
        distance_model_results = distance_model(useful_data, return_trace, progressbar)
        print('Parallax sampling')
        if distance_prior:
            parallax_model_results = fit_parallax_model(useful_data, return_trace, progressbar,prior_distance=distance_prior,prior_type='uniform')
        else:
            parallax_model_results = fit_parallax_model(useful_data, return_trace, progressbar,prior_type='uniform')
        mean_parallax = parallax_model_results['mu_parallax_mean']
        std_mean_parallax = parallax_model_results['mu_parallax_std']
        std_parallax = parallax_model_results['sigma_parallax_mean']
        if parallax_hist:
            ax0 = axes[0, j]
            ax0.axvline(mean_parallax, color='g', linestyle='--', label=r'$\mu_\varpi = {:.2f}\pm{:.2f}\,{}$'.format(mean_parallax,std_mean_parallax,u.mas))
            _, bins = np.histogram(filtered_data['parallax'],bins='auto')
            ax0.hist(useful_data['parallax'], bins=bins, color='red', alpha=0.95, histtype='step', label='U. parallax')
            ax0.hist(filtered_data['parallax'], bins=bins, color='orange', alpha=0.95, histtype='step', label='Parallax')
            ax0.hist(nonuseful_data['parallax'], bins=bins, color='green', alpha=0.95, histtype='step', label='N.U. parallax')
            bin_width = np.diff(bins)[0]
            scaling_factor = len(useful_data) * bin_width
            x = np.linspace(mean_parallax - 3*std_parallax,mean_parallax + 3*std_parallax , 100)
            p = norm.pdf(x, mean_parallax, std_parallax) * scaling_factor  # Scale the PDF by the scaling factor
            ax0.plot(x, p,label='Normal PDF')
            ax0.set_ylim(0, ax0.get_ylim()[1])
            ax0.fill_betweenx(y=[0, 1.5*ax0.get_ylim()[1]], x1=mean_parallax - std_mean_parallax, x2=mean_parallax + std_mean_parallax, color='blue', alpha=0.05,label='Standard desviation')
            ax0.set_xlabel(r'$\varpi$ [{}]'.format(data['parallax'].unit),fontsize=18)
            ax0.set_ylabel('Counts',fontsize=18)
            ax0.legend(framealpha=0.3,loc='upper left',fontsize=12)
            if paper_single:
                ax1 = axes[j,1]
            else:
                ax1 = axes[1, j]
        else:
            ax1 = axes[0, j]
        ax1.scatter(useful_data['parallax'], useful_data['Gmag'], s=10, alpha=0.9,c='red',edgecolor='blue',label='U. for distance est.')
        ax1.scatter(nonuseful_data['parallax'], nonuseful_data['Gmag'], s=10, alpha=0.9,c='red',label='N.U for distance est.')
        ax1.errorbar(filtered_data['parallax'], filtered_data['Gmag'], xerr=filtered_data['parallax_error'], fmt='none', alpha=0.3,capsize=0,elinewidth=0.5,color='gray')
        ax1.axvline(mean_parallax, color='g', linestyle='--', label=r'$\mu_\varpi = {:.2f}\pm{:.2f}\,{}$'.format(mean_parallax,std_parallax,u.mas))
        ax1.set_xlabel(r'$\varpi$ [{}]'.format(data['parallax'].unit),fontsize=18)
        ax1.set_ylabel(r'$G_\mathrm{mag}$',fontsize=18)
        ax1.set_xlim(mean_parallax-15*std_parallax,mean_parallax+15*std_parallax)
        lim_y_ax1 = ax1.get_ylim()
        ax1.fill_betweenx(y=[lim_y_ax1[0], 1.5*lim_y_ax1[1]], x1=mean_parallax - std_parallax, x2=mean_parallax + std_parallax, color='blue', alpha=0.05,label='Standard desviation')
        ax1.set_ylim(lim_y_ax1)
        ax1.invert_yaxis()
        ax1.legend(framealpha=0.3,loc='upper left',fontsize=12)
        # Second type of graphic: Histogram of inverse parallax with sampled distance chains
        if parallax_hist:
            if paper_single:
                ax2 = axes[j,2]
            else:
                ax2 = axes[2, j]
        else:
            ax2 = axes[1, j]
        mean_distance = distance_model_results['mu_r_mean']
        std_distance = distance_model_results['std_r_mean']
        alpha = (mean_distance / std_distance) ** 2
        beta = std_distance ** 2 / mean_distance
        
        # Histogram plotting
        _, bins = np.histogram(filtered_data['r_med_geo'],bins='auto')
        ax2.hist(useful_data['r_med_geo'], bins=bins, color='red', alpha=0.95, histtype='step', label='U. distance')
        ax2.hist(filtered_data['r_med_geo'], bins=bins, color='orange', alpha=0.95, histtype='step', label='Distance')
        ax2.hist(nonuseful_data['r_med_geo'], bins=bins, color='green', alpha=0.95, histtype='step', label='N.U. distance')
        # Calculate bin width and scaling factor
        bin_width = np.diff(bins)[0]
        scaling_factor = len(useful_data['r_med_geo']) * bin_width
        # Plot Gamma PDF
        x = np.linspace(0.8*np.min(bins), np.max(bins), 1000)
        p = gamma.pdf(x, alpha, scale=beta) * scaling_factor
        ax2.plot(x, p, label='Gamma PDF')
        # Calculate mode of the Gamma distribution
        if alpha > 1:
            mode = (alpha - 1) * beta
        else:
            mode = 0  # If alpha <= 1, the mode is at 0
        
        # Plot mode and standard deviation
        ax2.axvline(mode, color='blue', linestyle='--', label=f'Mode: {mode:.2f} kpc')
        ax2.set_ylim(0, ax2.get_ylim()[1])
        ax2.fill_betweenx(y=[0, 1.5*ax2.get_ylim()[1]], x1=mean_distance - std_distance, x2=mean_distance + std_distance, color='blue', alpha=0.05, label='Standard deviation')
        # Set labels and legend
        ax2.set_xlabel('Distance [kpc]', fontsize=18)
        ax2.set_ylabel('Counts', fontsize=18)
        ax2.legend(framealpha=0.3,fontsize=12)
        for key in results_distance.keys():
            results_distance[key].append(distance_model_results[key])
        for key in results_parallax.keys():
            results_parallax[key].append(parallax_model_results[key])
        # Fine-tuning the plots
        for ax in axes.flatten():
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(axis='both', which='both', direction='in',labelsize=14)
        if return_trace:
            all_traces.append(distance_model_results['trace_distance'])
            all_traces.append(parallax_model_results['trace_parallax'])
    if savefig:
        if paper_single:
            plt.savefig(savefig + 'parallax_determination_paper.pdf',bbox_inches='tight')
        else:
            plt.savefig(savefig + 'parallax_determination.pdf',bbox_inches='tight')
    plt.show()
    results = {}
    results.update(results_distance),results.update(results_parallax)
    return results  # or any other results you wish to return
def pm_determination(data, savefig=None, prob_number=[50, 60, 70, 80], progressbar=False,return_trace=False,return_pmdist=False,return_pmprob=False,paper_single=False):
    prob_number = np.array(prob_number) / 100
    if len(prob_number) == 1:
        fig, axes = plt.subplots(1, layout='constrained', figsize=(8,5))
    elif len(prob_number) == 2:
        fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(17.5,7))
    elif len(prob_number) == 3:
        fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(13,13))
    elif len(prob_number) == 4:
        fig, axes = plt.subplots(2, 2, layout='constrained', figsize=(13,13))
    else:
        raise ValueError("Unsupported number of probability thresholds")

    stats_results = []
    distance_pm = []
    pm_prob = []
    if len(prob_number) == 1:
        axes = np.array([axes])
    min_data = (data['probability'] >= np.nanmin(prob_number))
    data = data[min_data]
    # Determine the overall grid range
    total_pm_RA = data['pmra']
    total_pm_DEC = data['pmdec']
    added_space_ra = (np.nanmax(total_pm_RA) - np.nanmin(total_pm_RA)) / 8
    added_space_dec = (np.nanmax(total_pm_DEC) - np.nanmin(total_pm_DEC)) / 8
    grid_RA, grid_DEC = np.meshgrid(
        np.linspace(np.nanmin(total_pm_RA) - added_space_ra, np.nanmax(total_pm_RA) + added_space_ra, 700),
        np.linspace(np.nanmin(total_pm_DEC) - added_space_dec, np.nanmax(total_pm_DEC) + added_space_dec, 700)
    )

    for i, ax in zip(prob_number, axes.flatten()):
        print(f"{i * 100}%")
        # Filter the data based on the probability threshold
        probability_selection = (data['probability'] >= i)
        iterative_data = data[probability_selection]
        pm_RA = iterative_data['pmra']
        pm_DEC = iterative_data['pmdec']
        error_RA = iterative_data['pmra_error']
        error_Dec = iterative_data['pmdec_error']
        probability = iterative_data['probability']
        # Perform Bayesian analysis for the filtered data
        trace_results = FitProperMotion2DGaussian(pm_RA, pm_DEC, progressbar=progressbar,return_trace=return_trace)

        # Extract Bayesian analysis results
        bayesian_results = trace_results['results']
        
        # Extract and store the statistics
        stats = {
            'probability': i,
            'mu_RA_mean': bayesian_results['mu_RA_mean'],
            'mu_Dec_mean': bayesian_results['mu_Dec_mean'],
            'sigma_RA_mean': bayesian_results['sigma_RA_mean'],
            'sigma_Dec_mean': bayesian_results['sigma_Dec_mean'],
            'corr_mean': bayesian_results['corr_mean'],
            'mu_RA_std': bayesian_results['mu_RA_std'],
            'mu_Dec_std': bayesian_results['mu_Dec_std'],
            'sigma_RA_std': bayesian_results['sigma_RA_std'],
            'sigma_Dec_std': bayesian_results['sigma_Dec_std'],
            'corr_std': bayesian_results['corr_std']
        }
        if return_trace:
            stats['trace'] = trace_results.get('trace')
        stats_results.append(stats)
        # Create a 2D Gaussian distribution with mean parameters
        rv = multivariate_normal([bayesian_results['mu_RA_mean'], bayesian_results['mu_Dec_mean']],
                                 [[bayesian_results['sigma_RA_mean']**2, 
                                   bayesian_results['corr_mean'] * bayesian_results['sigma_RA_mean'] * bayesian_results['sigma_Dec_mean']],
                                  [bayesian_results['corr_mean'] * bayesian_results['sigma_RA_mean'] * bayesian_results['sigma_Dec_mean'], 
                                   bayesian_results['sigma_Dec_mean']**2]])
        density = rv.pdf(np.dstack([grid_RA, grid_DEC]))
        max_density_idx = np.argmax(density)
        max_density_coords = (grid_RA.ravel()[max_density_idx], grid_DEC.ravel()[max_density_idx])
            
        # Contour plot and scatter plot
        ax.contour(grid_RA, grid_DEC, density, levels=10, cmap=sns.color_palette("coolwarm", as_cmap=True))
        sc = ax.scatter(pm_RA, pm_DEC, s=25, alpha=0.8, marker='*',c=probability,cmap=sns.color_palette("coolwarm", as_cmap=True))
        cbar = fig.colorbar(sc, pad=0.01)
        cbar.set_label('Probability', fontsize=15)
        cbar.ax.tick_params(labelsize=12)
        ax.scatter(bayesian_results['mu_RA_mean'], bayesian_results['mu_Dec_mean'], marker='1', s=400, color='blue')
        ax.axvline(bayesian_results['mu_RA_mean'], color='darkorange', ls='--')
        ax.axhline(bayesian_results['mu_Dec_mean'], color='darkorange', ls='--')
        # Axis labels and title
        ax.set_xlabel(r'$\mu_\alpha*$ [mas/yr]',fontsize=16)
        ax.set_ylabel(r'$\mu_\delta$ [mas/yr]',fontsize=16)

        # Fine-tuning the plot
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='both', direction='in',labelsize=14)
        ax.set_aspect('equal')
        legend_label = r'$C = ({:.3f},{:.3f})\;{}$'.format(bayesian_results['mu_RA_mean'],bayesian_results['mu_Dec_mean'], (u.mas/u.yr))
        ax.legend([legend_label], loc='best', framealpha=0.8)
        if return_pmdist:
            distance_center_pm = np.sqrt((pm_RA - bayesian_results['mu_RA_mean']*(u.mas/u.yr))**2 + (pm_DEC - bayesian_results['mu_Dec_mean']*(u.mas/u.yr))**2)
            distancepm = {
                'probability': i,
                'distancepm': 1/distance_center_pm}
            distance_pm.append(distancepm)
            if return_pmprob:
                pmprob_ = (1/distance_center_pm)*iterative_data['probability']
                pm_prob.append(pmprob_)
    # Save the figure if a path is provided
    if savefig:
        if paper_single:
            fig.savefig(f"{savefig}proper_motion_2d_paper.pdf",bbox_inches='tight')
        else:
            fig.savefig(f"{savefig}proper_motion_2d.pdf",bbox_inches='tight')
    plt.show()
    if return_pmdist and return_pmprob:
        return stats_results, distance_pm, pm_prob
    elif return_pmdist:
        return stats_results, distance_pm
    else:
        return stats_results
def pm_determination(data, savefig=None, prob_number=[50, 60, 70, 80], progressbar=False,return_trace=False,return_pmdist=False,return_pmprob=False,paper_single=False,ax=None):
    prob_number = np.array(prob_number) / 100
    if len(prob_number) == 1:
        fig, axes = plt.subplots(1, layout='constrained', figsize=(7,5))
    elif len(prob_number) == 2:
        fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(17.5,7))
    elif len(prob_number) == 3:
        fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(13,13))
    elif len(prob_number) == 4:
        fig, axes = plt.subplots(2, 2, layout='constrained', figsize=(13,13))
    else:
        raise ValueError("Unsupported number of probability thresholds")
    if ax is not None:
        axes = ax
    stats_results = []
    distance_pm = []
    pm_prob = []
    if len(prob_number) == 1:
        axes = np.array([axes])
    min_data = (data['probability'] >= np.nanmin(prob_number))
    data = data[min_data]
    # Determine the overall grid range
    total_pm_RA = data['pmra']
    total_pm_DEC = data['pmdec']
    added_space_ra = (np.nanmax(total_pm_RA) - np.nanmin(total_pm_RA)) / 8
    added_space_dec = (np.nanmax(total_pm_DEC) - np.nanmin(total_pm_DEC)) / 8
    grid_RA, grid_DEC = np.meshgrid(
        np.linspace(np.nanmin(total_pm_RA) - added_space_ra, np.nanmax(total_pm_RA) + added_space_ra, 700),
        np.linspace(np.nanmin(total_pm_DEC) - added_space_dec, np.nanmax(total_pm_DEC) + added_space_dec, 700)
    )

    for i, ax in zip(prob_number, axes.flatten()):
        print(f"{i * 100}%")
        # Filter the data based on the probability threshold
        probability_selection = (data['probability'] >= i)
        iterative_data = data[probability_selection]
        pm_RA = iterative_data['pmra']
        pm_DEC = iterative_data['pmdec']
        error_RA = iterative_data['pmra_error']
        error_Dec = iterative_data['pmdec_error']
        probability = iterative_data['probability']
        # Perform Bayesian analysis for the filtered data
        trace_results = FitProperMotion2DGaussian(pm_RA, pm_DEC, progressbar=progressbar,return_trace=return_trace)

        # Extract Bayesian analysis results
        bayesian_results = trace_results['results']
        
        # Extract and store the statistics
        stats = {
            'probability': i,
            'mu_RA_mean': bayesian_results['mu_RA_mean'],
            'mu_Dec_mean': bayesian_results['mu_Dec_mean'],
            'sigma_RA_mean': bayesian_results['sigma_RA_mean'],
            'sigma_Dec_mean': bayesian_results['sigma_Dec_mean'],
            'corr_mean': bayesian_results['corr_mean'],
            'mu_RA_std': bayesian_results['mu_RA_std'],
            'mu_Dec_std': bayesian_results['mu_Dec_std'],
            'sigma_RA_std': bayesian_results['sigma_RA_std'],
            'sigma_Dec_std': bayesian_results['sigma_Dec_std'],
            'corr_std': bayesian_results['corr_std']
        }
        if return_trace:
            stats['trace'] = trace_results.get('trace')
        stats_results.append(stats)
        # Create a 2D Gaussian distribution with mean parameters
        rv = multivariate_normal([bayesian_results['mu_RA_mean'], bayesian_results['mu_Dec_mean']],
                                 [[bayesian_results['sigma_RA_mean']**2, 
                                   bayesian_results['corr_mean'] * bayesian_results['sigma_RA_mean'] * bayesian_results['sigma_Dec_mean']],
                                  [bayesian_results['corr_mean'] * bayesian_results['sigma_RA_mean'] * bayesian_results['sigma_Dec_mean'], 
                                   bayesian_results['sigma_Dec_mean']**2]])
        density = rv.pdf(np.dstack([grid_RA, grid_DEC]))
        max_density_idx = np.argmax(density)
        max_density_coords = (grid_RA.ravel()[max_density_idx], grid_DEC.ravel()[max_density_idx])
            
        # Contour plot and scatter plot
        ax.contour(grid_RA, grid_DEC, density, levels=10, cmap=sns.color_palette("coolwarm", as_cmap=True))
        sc = ax.scatter(pm_RA, pm_DEC, s=25, alpha=0.8, marker='*',c=probability,cmap=sns.color_palette("coolwarm", as_cmap=True))
        fig.colorbar(sc,pad=0.01).set_label('Probability',fontsize=15)
        ax.scatter(bayesian_results['mu_RA_mean'], bayesian_results['mu_Dec_mean'], marker='1', s=400, color='blue')
        ax.axvline(bayesian_results['mu_RA_mean'], color='darkorange', ls='--')
        ax.axhline(bayesian_results['mu_Dec_mean'], color='darkorange', ls='--')
        # Axis labels and title
        ax.set_xlabel(r'$\mu_\alpha*$ [mas/yr]',fontsize=16)
        ax.set_ylabel(r'$\mu_\delta$ [mas/yr]',fontsize=16)

        # Fine-tuning the plot
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_aspect('equal')
        legend_label = r'$C = ({:.2f},{:.2f})\;{}$'.format(bayesian_results['mu_RA_mean'],bayesian_results['mu_Dec_mean'], (u.mas/u.yr))
        ax.legend([legend_label], loc='best', framealpha=0.8)
        if return_pmdist:
            distance_center_pm = np.sqrt((pm_RA - bayesian_results['mu_RA_mean']*(u.mas/u.yr))**2 + (pm_DEC - bayesian_results['mu_Dec_mean']*(u.mas/u.yr))**2)
            distancepm = {
                'probability': i,
                'distancepm': 1/distance_center_pm}
            distance_pm.append(distancepm)
            if return_pmprob:
                pmprob_ = (1/distance_center_pm)*iterative_data['probability']
                pm_prob.append(pmprob_)
    # Save the figure if a path is provided
    if savefig:
        if paper_single:
            fig.savefig(f"{savefig}proper_motion_2d_paper.pdf",bbox_inches='tight')
        else:
            fig.savefig(f"{savefig}proper_motion_2d.pdf",bbox_inches='tight')
    if ax is None:
        plt.show()
    if return_pmdist and return_pmprob:
        return stats_results, distance_pm, pm_prob
    elif return_pmdist:
        return stats_results, distance_pm
    else:
        return stats_results
def velocity_model(data, return_trace=False, progressbar=False):
    # Assuming projected_velocity is in mas/yr, calculate prior mean
    prior_mu_v = np.mean(data['projected_velocity'])

    with pm.Model() as vel_model:
        # Hyperprior for the mean velocity of the star cluster
        mu_v = pm.Normal("mu_v", mu=prior_mu_v, sigma=10)
        
        # Standard deviation for the velocity
        std_v = pm.Uniform("std_v", lower=0, upper=50)
        
        # Likelihood for the observed velocity
        pm.Normal("observed_velocity", mu=mu_v, sigma=std_v, observed=data["projected_velocity"])

    # Iterative sampling process
    max_rhat = 2.0
    target_accept = 0.8
    tune = 4000

    while max_rhat > 1.0:
        with vel_model:
            trace = pm_jax.sample_numpyro_nuts(draws=10000, tune=tune, target_accept=target_accept, progressbar=progressbar)
            
        rhat_vals = az.rhat(trace)
        # Extract the maximum rhat value
        max_rhat = round(np.nanmax(rhat_vals.to_array().values),2)
        if target_accept <= 0.9999999:
            target_accept += 0.05
        tune += 2000

    results = {
        'mu_v_mean': trace.posterior['mu_v'].mean().item(),
        'std_v_mean': trace.posterior['std_v'].mean().item(),
        'mu_v_std': trace.posterior['mu_v'].std().item(),
        'std_v_std': trace.posterior['std_v'].std().item(),
    }

    if return_trace:
        results['trace'] = trace

    return results
def velocity_determination(data, prob_thresholds=[50, 60, 70, 80], return_trace=False, progressbar=False,savefig=None,paper_single=False):
    prob_thresholds = np.array(prob_thresholds) / 100
    num_thresholds = len(prob_thresholds)
    
    # Determine the layout of subplots based on the number of thresholds
    if num_thresholds <= 2:
        nrows, ncols = 1, num_thresholds
    else:
        nrows = 2
        ncols = int(np.ceil(num_thresholds / 2))
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), constrained_layout=True)
    if num_thresholds == 1:
        axs = np.array([axs])  # Ensure axs is iterable even for a single subplot
    
    results = []

    for i, prob_threshold in enumerate(prob_thresholds):
        print(f"Processing for probability threshold: {prob_threshold * 100}%")
        
        # Filter data based on probability threshold
        filtered_data = data[data['probability'] >= prob_threshold]
    
        # Call velocity model with filtered data (Assuming velocity_model is defined elsewhere)
        model_results = velocity_model(filtered_data, return_trace=return_trace, progressbar=progressbar)
    
        # Store results
        results.append({
            'probability': prob_threshold,
            'model_results': model_results
        })
        
        # Select the current axis for plotting
        ax = axs.flatten()[i]

        # Histogram of observed velocities
        _, bins, _ = ax.hist(filtered_data['projected_velocity'], bins='auto', color='orange', alpha=0.7, histtype='step', label='Observed Velocity')

        # Mean velocity line and label
        mean_velocity = model_results['mu_v_mean']
        std_velocity = model_results['std_v_mean']
        mean_std = model_results['mu_v_std']
        ax.axvline(mean_velocity, color='blue', linestyle='--', label=f'Mean Velocity: {mean_velocity:.2f} ± {mean_std:.2f} km/s')
        lim_y_ax = ax.get_ylim()
        ax.fill_betweenx(y=[lim_y_ax[0], 1.5*lim_y_ax[1]], x1=mean_velocity - mean_std, x2=mean_velocity + mean_std, color='blue', alpha=0.05,label='Standard desviation')
        bin_width = np.diff(bins)[0]
        scaling_factor = len(filtered_data) * bin_width
        x = np.linspace(mean_velocity - 3*std_velocity, mean_velocity + 3*std_velocity, 100)
        p = norm.pdf(x, mean_velocity, std_velocity) * scaling_factor
        ax.plot(x, p, label='Velocity Model')
        ax.set_xlabel('Projected Velocity [$\mathrm{mas~yr}^{-1}$]',fontsize=14)
        ax.set_ylabel('Counts',fontsize=14)
        if not paper_single:
            ax.set_title(f'Prob Threshold: {prob_threshold * 100:.0f}%')
        ax.legend()

    # Adjust layout if fewer plots than slots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axs.flatten()[j])

    if savefig:
        if paper_single:
            fig.savefig(f"{savefig}projected_velocity_paper.pdf",bbox_inches='tight')
        else:
            fig.savefig(f"{savefig}projected_velocity.pdf",bbox_inches='tight')
    plt.show()
    return results
def center_determination(data, grid_ra=None, grid_dec=None, weights=None, return_density=False, return_grids=False, return_bestparams=False):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        ra = data['ra']
        dec = data['dec']
        ra_err = data['ra_error']
        dec_err = data['dec_error']
        ra_dec = np.vstack([ra, dec]).T
        min_bandwidth = np.mean([np.mean(ra_err).to(u.deg).value, np.mean(dec_err).to(u.deg).value])
        params = {
            'bandwidth': np.linspace(min_bandwidth, 0.7, 500),
            'kernel': ['gaussian', 'epanechnikov', 'exponential', 'linear']
        }
        grid = GridSearchCV(KernelDensity(), params, cv=5, n_jobs=-1)
        if weights is not None:
            grid.fit(ra_dec, sample_weight=weights)
        else:
            grid.fit(ra_dec)
        kde = grid.best_estimator_
        if grid_ra is None or grid_dec is None:
            added_space_ra = (np.nanmax(ra) - np.nanmin(ra)) / 6
            added_space_dec = (np.nanmax(dec) - np.nanmin(dec)) / 5
            grid_ra, grid_dec = np.meshgrid(
                np.linspace(np.nanmin(ra) - added_space_ra, np.nanmax(ra) + added_space_ra, 700),
                np.linspace(np.nanmin(dec) - added_space_dec, np.nanmax(dec) + added_space_dec, 700)
            )

        grid_points = np.vstack([grid_ra.ravel(), grid_dec.ravel()]).T
        log_density = kde.score_samples(grid_points)
        density = np.exp(log_density).reshape(grid_ra.shape)
        max_density_idx = np.argmax(density)
        max_density_coords = (grid_ra.ravel()[max_density_idx], grid_dec.ravel()[max_density_idx])
        best_bandwidth = grid.best_params_['bandwidth']*u.deg
        # Assuming the bandwidth represents an additional "smoothing" uncertainty, 
        
        # Calculate the mean errors (assuming errors are standard deviations)
        ra_mean_error = np.mean(data['ra_error'])
        dec_mean_error = np.mean(data['dec_error'])
        
        # Incorporate the bandwidth as an additional error term linearly rather than quadratically
        ra_total_error = np.sqrt(ra_mean_error**2 + best_bandwidth**2)
        dec_total_error = np.sqrt(dec_mean_error**2 + best_bandwidth**2)

        print(ra_total_error.to(u.deg), dec_total_error.to(u.deg))
        results = {'center_coords': max_density_coords,
                  'center_coords_error' : (ra_total_error.to(u.deg), dec_total_error.to(u.deg))}
        if return_density:
            results['density'] = density
        if return_grids:
            results['grid_ra'] = grid_ra
            results['grid_dec'] = grid_dec
        if return_bestparams:
            results['best_params'] = grid.best_params_
            
        return results
def graph_center_determination(data, projection=None, savefig=None, prob_number=[50, 60, 70, 80], weighted=False, weight_array=None, only_weight=False,paper_single=False,weight_prob=60,distance_scale=None,ax=None):
    # Ensure prob_number is a list or None
    if prob_number is not None:
        prob_number = np.array(prob_number) / 100.0

    # Determine the number of columns for the subplot based on the inputs
    num_cols = len(prob_number) if prob_number is not None else 0
    if weighted:
        num_cols += 1  # Add an extra column for weighted KDE if needed

    # Adjust num_cols to ensure there's at least one plot
    num_cols = max(1, num_cols)

    subplot_kw = {'projection': projection} if projection else {}
    centers = []
    best_params_list = []

    # Calculate the grid for KDE
    total_ra = data['ra']
    total_dec = data['dec']
    added_space_ra = (np.nanmax(total_ra) - np.nanmin(total_ra)) / 8
    added_space_dec = (np.nanmax(total_dec) - np.nanmin(total_dec)) / 8
    grid_ra, grid_dec = np.meshgrid(
        np.linspace(np.nanmin(total_ra) - added_space_ra, np.nanmax(total_ra) + added_space_ra, 700),
        np.linspace(np.nanmin(total_dec) - added_space_dec, np.nanmax(total_dec) + added_space_dec, 700)
    )

    # Setup the figure and axes for plotting
    fig_kde, ax_kde = plt.subplots(1, num_cols, figsize=(7 * num_cols, 5), subplot_kw=subplot_kw, constrained_layout=True)
    if num_cols == 1:
        ax_kde = [ax_kde]  # Ensure ax_kde is always a list for consistency
    if ax is not None:
        ax_kde = np.array(ax)
    ax_index = 0  # Index to track the current axis for plotting
    # Unweighted KDE calculations for each probability threshold
    if prob_number is not None:
        for i, ax in zip(prob_number, ax_kde[:len(prob_number)]):
            selected_data = data[data['probability'] >= i]
            # Perform center determination and plotting...
            # (Insert your center_determination function and plotting code here)
            print(f"{i * 100}% probability members")
            ax_index += 1
            # Filter the data based on the probability threshold
            selected_data = data[data['probability'] >= i]
            results = center_determination(selected_data, grid_ra, grid_dec, return_density=True, return_bestparams=True)
            max_density_coords = results['center_coords']
            density = results['density']
            best_params = results.get('best_params', {})
            levels = levels = np.sort(np.concatenate([np.linspace(density.min(), density.max()/5, 6),np.linspace(density.max()/5, density.max(), 4)[1:]]))
            lim_coord_min = SkyCoord(ra=grid_ra.min(), dec=grid_dec.min(), frame='icrs', unit='deg')
            lim_coord_max = SkyCoord(ra=grid_ra.max(), dec=grid_dec.max(), frame='icrs', unit='deg')
            if projection is not None:
                ax.scatter(selected_data['ra'], selected_data['dec'], color='yellow', s=30, marker='*', transform=ax.get_transform('world'))
                ax.scatter(max_density_coords[0], max_density_coords[1], marker='1', s=400, color='blue', transform=ax.get_transform('world'))
                ax.contour(grid_ra, grid_dec, density, levels=levels, cmap='viridis',transform=ax.get_transform('world'))
                coord = SkyCoord(ra=max_density_coords[0], dec=max_density_coords[1], frame='icrs', unit='deg')
                pixels = ax.wcs.world_to_pixel(coord)
                ax.axvline(pixels[0], color='y', ls='--')
                ax.axhline(pixels[1], color='y', ls='--')
                ax.coords.grid(True, color='gray', linestyle='dotted')
                ax.coords[0].set_major_formatter('d.ddd')
                ax.coords[1].set_major_formatter('d.ddd')
                pixels_min,pixels_max = ax.wcs.world_to_pixel(lim_coord_min),ax.wcs.world_to_pixel(lim_coord_max)
                ax.set_ylim(pixels_min[0],pixels_max[0])
                ax.set_xlim(pixels_min[1],pixels_max[1])
                if distance_scale != None:
                    gc_distance = distance_scale
                    scalebar_length = 10 * u.pc
                    scalebar_angle = (scalebar_length / gc_distance).to(u.deg, equivalencies=u.dimensionless_angles())
                    add_scalebar(ax, scalebar_angle, label="10 pc", color="white")
            if projection is None:
                ax.scatter(selected_data['ra'], selected_data['dec'], color='yellow', s=30, marker='*')
                ax.scatter(max_density_coords[0], max_density_coords[1], marker='1', s=400, color='blue')
            ax.set_xlabel(r'$\alpha$ [{}]'.format(data['ra'].unit),fontsize=16)
            ax.set_ylabel(r'$\delta$ [{}]'.format(data['dec'].unit),fontsize=16)
            ax.set_title(f'{i * 100}% probability members')
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(axis='both', which='both', direction='in')
            ax.set_aspect('equal')
            ax.set_facecolor('black')
            legend_label = r'$C = ({:.2f},{:.2f})\;{}$'.format(max_density_coords[0].value, max_density_coords[1].value, u.deg)
            ax.legend([legend_label], loc='best', shadow=True, framealpha=0.8)
            centers.append([max_density_coords,results['center_coords_error']])
            best_params_list.append(best_params)

    # Weighted KDE calculation
    if weighted and weight_array is not None:
        ax = ax_kde[ax_index] if ax_index < len(ax_kde) else ax_kde[-1]
        print("Weighted KDE")
        selected_data = data[data['probability'] >= weight_prob/100]
        results = center_determination(selected_data, grid_ra, grid_dec, weights=weight_array, return_density=True, return_bestparams=True)
        max_density_coords = results['center_coords']
        density = results['density']
        best_params = results.get('best_params', {})
        levels = np.sort(np.concatenate([np.linspace(density.min(), density.max()/5, 6),np.linspace(density.max()/5, density.max(), 4)[1:]]))
        lim_coord_min = SkyCoord(ra=grid_ra.min(), dec=grid_dec.min(), frame='icrs', unit='deg')
        lim_coord_max = SkyCoord(ra=grid_ra.max(), dec=grid_dec.max(), frame='icrs', unit='deg')
        if projection is not None:
            sc = ax.scatter(selected_data['ra'], selected_data['dec'], s=30, marker='.', transform=ax.get_transform('world'),c=selected_data['probability'],cmap=sns.color_palette("coolwarm", as_cmap=True),zorder=11,alpha=0.95)
            cbar = fig_kde.colorbar(sc,pad=0.01)
            cbar.set_label('Probability',fontsize=15)
            cbar.ax.tick_params(labelsize=12)
            ax.scatter(max_density_coords[0], max_density_coords[1], marker='1', s=400, color='blue', transform=ax.get_transform('world'),zorder=12,alpha=0.95)
            ax.contour(grid_ra, grid_dec, density, levels=levels, cmap='viridis',transform=ax.get_transform('world'))
            coord = SkyCoord(ra=max_density_coords[0], dec=max_density_coords[1], frame='icrs', unit='deg')
            pixels = ax.wcs.world_to_pixel(coord)
            ax.axvline(pixels[0], color='y', ls='--')
            ax.axhline(pixels[1], color='y', ls='--')
            ax.coords.grid(True, color='gray', linestyle='dotted')
            ax.coords[0].set_major_formatter('d.dd')
            ax.coords[1].set_major_formatter('d.dd')
            pixels_min,pixels_max = ax.wcs.world_to_pixel(lim_coord_min),ax.wcs.world_to_pixel(lim_coord_max)
            ax.set_ylim(pixels_min[0],pixels_max[0])
            ax.set_xlim(pixels_min[1],pixels_max[1])
            if distance_scale != None:
                gc_distance = distance_scale
                scalebar_length = 5 * u.pc
                scalebar_angle = (scalebar_length / gc_distance).to(u.deg, equivalencies=u.dimensionless_angles())
                add_scalebar(ax, scalebar_angle, label="5 pc", color="white")
        if projection is None:
            ax.scatter(selected_data['ra'], selected_data['dec'], color='yellow', s=30, marker='*')
            ax.scatter(max_density_coords[0], max_density_coords[1], marker='1', s=400, color='blue')
        ax.set_xlabel(r'$\alpha$ [{}]'.format(data['ra'].unit),fontsize=16)
        ax.set_ylabel(r'$\delta$ [{}]'.format(data['dec'].unit),fontsize=16)
        if not only_weight:
            ax.set_title(f'Weighted')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='both', direction='in',labelsize=14)
        ax.set_aspect('equal')
        legend_label = r'$C = ({:.4f},{:.4f})\;{}$'.format(max_density_coords[0].value, max_density_coords[1].value, u.deg)
        ax.legend([legend_label], framealpha=0.8,fontsize=11)
        centers.append([max_density_coords,results['center_coords_error']])
        best_params_list.append(best_params)
    # Saving the figure
    if savefig:
        if paper_single:
            plt.savefig(f"{savefig}center_determination_paper.pdf",bbox_inches='tight')
        else:
            plt.savefig(f"{savefig}center_determination.pdf",bbox_inches='tight')
    if ax is None:
        plt.show()
    return centers, best_params_list
def density_annulus_calculator_width(data, center, width, return_radius_gen=False):
    d_center = (angular_separation(data['ra'], data['dec'], center.ra, center.dec)).to(u.arcmin)
    # Generate radii
    radius_gen = np.unique(np.concatenate([np.arange(0, np.nanmax(d_center.value) / 3, 0.5 * width), 
    np.arange(np.nanmax(d_center.value) / 3, 2 * np.nanmax(d_center.value) / 3, width),
    np.arange(2 * np.nanmax(d_center.value) / 3, 1.1 * np.nanmax(d_center.value), 2 * width)
]))
    # Initialize lists to store density and density errors for annulus and circles
    density_annulus = []
    density_errors_annulus = []
    valid_radii = []  # List to store radii corresponding to non-zero densities

    # Calculate density using annulus
    for i in range(len(radius_gen) - 1):
        inner_radius = radius_gen[i]
        outer_radius = radius_gen[i + 1]
        criteria = (d_center >= inner_radius * u.arcmin) & (d_center < outer_radius * u.arcmin)
        counts = len(d_center[criteria])
        annulus_area = np.pi * (outer_radius ** 2 - inner_radius ** 2)
        if counts > 0:  # Only consider annuli with non-zero counts
            density_r_annulus = counts / annulus_area
            density_error_r_annulus = np.sqrt(counts) / annulus_area
            density_annulus.append(density_r_annulus)
            density_errors_annulus.append(density_error_r_annulus)
            valid_radii.append((inner_radius + outer_radius) / 2)  # Midpoint of the annulus

    results = {
        'density_annulus': density_annulus,
        'density_errors_annulus': density_errors_annulus,
        'd_center' : d_center
    }
    if return_radius_gen:
        results['radius_gen'] = valid_radii  # Return only the radii for non-zero densities
    return results
def density_annulus_calculator_equip(data, center, return_radius_gen=False):
    # Calculate the angular separation of each star from the cluster center
    d_center = angular_separation(data['ra'], data['dec'], center.ra, center.dec).to(u.arcmin)
    
    # Determine the number of bins using the equiprobable bin rule
    n = len(data)
    k = int(2 *n**(2/5))  # Number of bins
    
    # Calculate bin edges such that each bin has approximately equal number of stars
    sorted_d_center = np.sort(d_center.value)
    bin_edges = np.interp(np.linspace(0, len(sorted_d_center), k+1),
                          np.arange(len(sorted_d_center)),
                          sorted_d_center)
    # Initialize lists to store density and density errors for each annulus
    density_annulus = []
    density_errors_annulus = []
    valid_radii = []  # To store the average radius of each annulus
    
    # Calculate density for each annulus
    for i in range(len(bin_edges) - 1):
        inner_radius = bin_edges[i]
        outer_radius = bin_edges[i + 1]
        
        criteria = (d_center.value >= inner_radius) & (d_center.value < outer_radius)
        counts = np.sum(criteria)
        
        annulus_area = np.pi * (outer_radius**2 - inner_radius**2)  # Area of the annulus in arcmin^2
        
        density_r_annulus = counts / annulus_area
        density_error_r_annulus = np.sqrt(counts) / annulus_area
        
        density_annulus.append(density_r_annulus)
        density_errors_annulus.append(density_error_r_annulus)
        valid_radii.append((inner_radius + outer_radius) / 2)  # Midpoint radius of the annulus
    
    results = {
        'density_annulus': density_annulus,
        'density_errors_annulus': density_errors_annulus,
        'd_center' : d_center
    }
    
    if return_radius_gen:
        results['radius_gen'] = valid_radii
    
    return results
def calculate_galactic_mass(galactic_radius):
    galactic_radius = ensure_units(galactic_radius,u.pc)
    galactic_mass = 2E8*(galactic_radius/(30*u.pc))**1.2*u.Msun
    return galactic_mass.to(u.Msun)
def tidal_radius_prior(cluster_mass,galactic_radius,galactic_mass=None,type='angular',distance=None,return_galactic_mass=False,data=None):
    if galactic_mass is None:
        galactic_mass = calculate_galactic_mass(galactic_radius)
    if cluster_mass is None:
        cluster_mass = estimate_cluster_mass(data,distance)
    cluster_mass = ensure_units(cluster_mass,u.Msun)
    galactic_mass = ensure_units(galactic_mass,u.Msun)
    galactic_radius = ensure_units(galactic_radius,u.kpc)
    tidal_radius = (cluster_mass/(2*galactic_mass))**(1/3)*galactic_radius
    results = {}
    if type == 'angular':
        results['angular_size'] = angular_size(tidal_radius.to(u.pc),distance)
    if type == 'linear':
        results['linear_size'] = tidal_radius.to(u.pc)
    if type == 'both':
        results['linear_size'] = tidal_radius.to(u.pc)
        results['angular_size'] = angular_size(tidal_radius.to(u.pc),distance)
    if return_galactic_mass:
        results['galactic_mass'] = galactic_mass
    return results
def calculate_absolute_magnitude(apparent_mag, distance):
    # Convert distance from parsec to pc if it's not in the correct unit
    distance = distance.to(u.pc) if hasattr(distance, 'unit') else distance * u.pc
    # Calculate absolute magnitude
    return apparent_mag - 5 * (np.log10(distance.value) - 1)*u.mag

def estimate_luminosity(absolute_mag):
    # Calculate luminosity based on absolute magnitude
    return 10**((4.74 - absolute_mag.value) / 2.5)

def estimate_mass_from_luminosity(luminosity):
    # Invert the mass-luminosity relation
    return luminosity**(1/3.5)*u.Msun

def estimate_cluster_mass(data,distance):
    # Calculate absolute magnitude for each star
    data['G_abs'] = calculate_absolute_magnitude(data['Gmag'], distance)
    
    # Calculate luminosity for each star
    data['luminosity'] = estimate_luminosity(data['G_abs'])
    
    # Estimate mass for each star
    data['mass'] = estimate_mass_from_luminosity(data['luminosity'])

    # Sum the masses to get total cluster mass
    total_mass = np.sum(data['mass'])
    return total_mass

def calculate_galactocentric_distance(ra, dec, distance):
    # Convert RA and Dec to Galactic coordinates
    sky_coord = SkyCoord(ra=ra, dec=dec, frame='icrs')
    galactic_coord = sky_coord.transform_to('galactic')
    distance = ensure_units(distance,u.kpc)
    # Constants
    R_sun = 8.3 * u.kpc
    # Sun's distance to Galactic center

    # Extract Galactic coordinates
    l = galactic_coord.l.rad  # Galactic longitude in radians
    b = galactic_coord.b.rad  # Galactic latitude in radians

    # Calculate galactocentric distance
    d_gc = np.sqrt(R_sun**2 + distance**2 - 2 * R_sun * distance * np.cos(l) * np.cos(b))

    return d_gc
def calculate_hill_radius(distance,center=None,galactocentric_distance=None,galactic_mass=None,cluster_mass=None,data=None,return_galdist=False,return_cluster_mass=False,return_linear_size=False,return_galactic_mass=False):
    distance = ensure_units(distance,u.kpc)
    if galactocentric_distance is None:
        galactocentric_distance = calculate_galactocentric_distance(center[0],center[1], distance)
    if galactic_mass is None:
        galactic_mass = calculate_galactic_mass(galactocentric_distance)
    if cluster_mass is None:
        cluster_mass = estimate_cluster_mass(data,distance)
    hill_radius = (galactocentric_distance * (cluster_mass / (3 * galactic_mass))**(1/3)).to(u.pc)
    hill_angular = (hill_radius / distance).to(u.deg, equivalencies=u.dimensionless_angles())
    results = {}
    results['angular_size'] = hill_angular.to(u.arcmin)
    if return_galdist:
        results['galactocentric_distance'] = galactocentric_distance
    if return_cluster_mass:
        results['cluster_mass'] = cluster_mass
    if return_linear_size:
        results['linear_size'] = hill_radius
    if return_galactic_mass:
        results['galactic_mass'] = galactic_mass
    return results
def grav_bound_radius(M_total, A=15.3*u.kpc**-1*u.km/u.s, A_err=0.4*u.kpc**-1*u.km/u.s, B=-11.9*u.kpc**-1*u.km/u.s, B_err=0.4*u.kpc**-1*u.km/u.s,distance=None):
    """
    Calculate the tidal radius of a star cluster.

    Parameters:
    - M_total: Total mass of the cluster members in solar masses.
    - A: Oort constant A in kpc^-1.
    - A_err: Error in Oort constant A.
    - B: Oort constant B in kpc^-1.
    - B_err: Error in Oort constant B.

    Returns:
    - rt: Tidal radius in parsecs.
    - rt_err: Error in tidal radius.
    """

    # Gravitational constant 

    A_B_squared = (A - B)**2
    A_B_squared_err = 2 * np.abs(A - B) * np.sqrt(A_err**2 + B_err**2)
    # Calculate the tidal radius
    radius = (((G * M_total) / (2 * A_B_squared))**(1/3)).to(u.pc)
    
    # Propagate error for tidal radius
    radius_err = radius * np.sqrt((4 * A_B_squared_err / A_B_squared)**2)
    results = {}
    if distance is None:
        results['linear_radius'] = radius.to(u.pc)
        results['linear_radius_err'] = radius_err.to(u.pc)
    if distance is not None:
        results['angular_radius'] = angular_size(radius,distance)
        results['angular_radius_err'] = angular_size(radius_err,distance)
    return results
def calculate_half_light_radius(Gmag, distance_to_center):
    """
    Calculate the half-light radius for a cluster given the G-band magnitudes
    and distances of stars from the cluster center.
    
    Parameters:
    - Gmag: numpy.ndarray, apparent magnitudes of the stars in the G band.
    - distance_to_center: numpy.ndarray, distances of each star from the cluster center.
    
    Returns:
    - R_h: float, the half-light radius of the cluster (in the same units as distance_to_center).
    """
    if hasattr(Gmag, 'unit'):
        Gmag = Gmag.value  # Assuming Gmag is an Astropy Quantity with unnecessary units
    # Convert Gmag to relative luminosities
    L = 10 ** (-0.4 * (Gmag - np.nanmin(Gmag)))
    
    # Sort stars by their distance from the cluster center
    indices_sorted_by_distance = np.argsort(distance_to_center)
    sorted_distances = distance_to_center[indices_sorted_by_distance]
    sorted_luminosities = L[indices_sorted_by_distance]
    
    # Compute cumulative luminosity
    cumulative_luminosity = np.cumsum(sorted_luminosities)
    
    # Total luminosity is the last value in the cumulative sum
    total_luminosity = cumulative_luminosity[-1]
    half_total_luminosity = total_luminosity / 2
    
    # Find the smallest distance at which the cumulative luminosity exceeds half of the total luminosity
    half_light_radius_index = np.argmax(cumulative_luminosity >= half_total_luminosity)
    R_h = sorted_distances[half_light_radius_index]
    
    return R_h
def RDP_bayesian(density_annulus, radius_gen, return_trace=False,progressbar=False,d_center=None,priors=False,priors_parameters=None,return_priors=None):
    with pm.Model() as king_model:
        sigma = pm.HalfNormal('sigma', sigma=5)
        b = pm.Uniform('b', lower=0, upper=2*np.nanmin(density_annulus))
        k = pm.Uniform('k', lower=b, upper=2*np.nanmax(density_annulus))
        if d_center is None:
            R_c = pm.Uniform('R_c',lower=0,upper=0.8*np.nanmax(radius_gen).value)
            R_t = pm.Uniform('R_t', lower=R_c, upper=1.5*np.nanmax(radius_gen))  # Ensure R_t > R_c
        if (d_center is not None) and (priors == False):
            R_c = pm.Uniform('R_c',lower=0,upper=0.8*np.nanmax(d_center).value)
            R_t = pm.Uniform('R_t', lower=R_c, upper=1.5*np.nanmax(d_center))  # Ensure R_t > R_c
        if (d_center is not None) and (priors == True):
            cluster_mass = priors_parameters['cluster_mass']
            galactic_radius = priors_parameters['galactic_radius']
            distance = priors_parameters['distance']
            tidal_pot_results = tidal_radius_prior(cluster_mass,galactic_radius,galactic_mass=None,type='angular',distance=distance,return_galactic_mass=True)
            tidal_potential = tidal_pot_results['angular_size']
            galactic_mass= tidal_pot_results['galactic_mass']
            hill_radius = calculate_hill_radius(galactocentric_distance=galactic_radius,distance=distance,galactic_mass=galactic_mass,cluster_mass=cluster_mass,data=None)['angular_size']
            gravitational_bound_radius = grav_bound_radius(cluster_mass,distance=distance)['angular_radius']
            max_tidal = np.nanmax([tidal_potential.value,hill_radius.value,gravitational_bound_radius.value])
            R_c = pm.Uniform('R_c',lower=0,upper=0.8*np.nanmax(d_center).value)
            R_t = pm.Uniform('R_t', lower=R_c, upper=1.5*max_tidal)  # Ensure R_t > R_c
        r = pm.ConstantData('radius', radius_gen)
        density_points = pm.ConstantData('density', density_annulus)
        king = pm.Deterministic('king', pm.math.switch(r <= R_t,
                                                       k * ((1 / pm.math.sqrt(1 + (r / R_c) ** 2)) - (1 / pm.math.sqrt(1 + (R_t / R_c) ** 2))) ** 2 + b,
                                                       b))
        obs_density = pm.Normal('obs_density', mu=king, sigma=sigma, observed=density_points)
    rhat = [2, 2]
    target_accept = 0.8
    tune = 4000
    while any(x > 1 for x in rhat):
        with king_model:
            king_trace = pm_jax.sample_numpyro_nuts(draws=100000, tune=tune, target_accept=target_accept, random_seed=np.random.randint(1, 100000),progressbar=progressbar)
            rhat = az.summary(king_trace, var_names=["sigma", "k", "R_c", "R_t", "b"])['r_hat'].iloc[:]
        if target_accept <= 0.9999999:
            target_accept += 0.05
        tune += 2000
    k_mean = king_trace.posterior['k'].median().item()
    b_mean = king_trace.posterior['b'].median().item()
    R_c_mean = king_trace.posterior['R_c'].median().item()
    R_t_mean = king_trace.posterior['R_t'].median().item()
    k_std = king_trace.posterior['k'].std().item()
    b_std = king_trace.posterior['b'].std().item()
    R_c_std = king_trace.posterior['R_c'].std().item()
    R_t_std = king_trace.posterior['R_t'].std().item()
    king_std = king_trace.posterior['sigma'].median().item()
    bg_level = (b_mean + 3*b_std)
    C = np.log(R_t_mean/R_c_mean)
    d_c = 1 + k_mean/bg_level
    r_lim = R_c_mean*np.sqrt(k_mean/(3*b_std)-1)
    results = {
        'k_mean': k_mean,
        'b_mean': b_mean,
        'R_c_mean': R_c_mean*u.arcmin,
        'R_t_mean': R_t_mean*u.arcmin,
        'k_std': k_std,
        'b_std': b_std,
        'bg_level' : bg_level,
        'R_c_std': R_c_std,
        'R_t_std': R_t_std,
        'king_std': king_std,
        'C' : C,
        'd_c' : d_c,
        'r_lim': r_lim
    }
    if return_trace is True:
        results['king_trace'] = king_trace
    if return_priors is True:
        priors_results = {
            'cluster_mass' : cluster_mass,
            'galactic_radius' : galactic_radius,
            'galactic_mass' : galactic_mass,
            'tidal_pot_radius' : tidal_potential,
            'hill_radius' : hill_radius,
            'gravitational_bound_radius' : gravitational_bound_radius,
            'far_member' : np.nanmax(d_center.value),
            'max_tidal' : max_tidal}
        results['priors_results'] = priors_results
    return results
def RDP_bayesian_log_space(density_annulus, radius_gen, return_trace=False, progressbar=False, d_center=None):
    with pm.Model() as king_model:
        # Use a log transformation for parameters that span several orders of magnitude
        log_sigma = pm.Normal('log_sigma', mu=np.log(5), sigma=np.log(1.0000000001))
        sigma = pm.Deterministic('sigma', pm.math.exp(log_sigma))
        log_b = pm.Uniform('log_b', upper=np.log(0.5 * np.nanmax(density_annulus)))
        b = pm.Deterministic('b', pm.math.exp(log_b))
        log_k = pm.Normal('log_k', mu=np.log(b), sigma=np.log(1.000000001))
        k = pm.Deterministic('k', pm.math.exp(log_k))
        
        if d_center is None:
            log_R_c = pm.Uniform('log_R_c',upper=np.log(0.8*np.nanmax(radius_gen).value))
            R_c = pm.Deterministic('R_c', pm.math.exp(log_R_c))

            log_R_t = pm.Uniform('log_R_t', lower=log_R_c, upper=np.log(2*np.nanmax(radius_gen)) ) # Ensure R_t > R_c
            R_t = pm.Deterministic('R_t', pm.math.exp(log_R_t))
        else:
            log_R_c = pm.Uniform('log_R_c',upper=np.log(0.8*np.nanmax(d_center).value))
            R_c = pm.Deterministic('R_c', pm.math.exp(log_R_c))

            log_R_t = pm.Uniform('log_R_t', lower=log_R_c, upper=np.log(2*np.nanmax(d_center).value) ) # Ensure R_t > R_c
            R_t = pm.Deterministic('R_t', pm.math.exp(log_R_t))

        log_r = pm.ConstantData('log_radius', np.log(radius_gen))
        r = pm.Deterministic('radius', pm.math.exp(log_r))
        log_density_points = pm.ConstantData('density', np.log(density_annulus))
        # King model in log space
        log_king = pm.Deterministic('log_king', pm.math.log(pm.math.switch(
            r < R_t,
            k * ((1 / pm.math.sqrt(1 + (r / R_c) ** 2)) - (1 / pm.math.sqrt(1 + (R_t / R_c) ** 2))) ** 2 + b,
            b
        )))
        # Likelihood in log space
        log_obs_density = pm.Normal('obs_log_density', mu=log_king, sigma=log_sigma, observed=log_density_points)
    rhat = [2, 2]
    target_accept = 0.8
    tune = 4000
    while any(x > 1 for x in rhat):
        with king_model:
            king_trace = pm_jax.sample_numpyro_nuts(draws=100000, tune=tune, target_accept=target_accept, random_seed=np.random.randint(1, 100000),progressbar=progressbar)
            rhat = az.summary(king_trace, var_names=["sigma", "k", "R_c", "R_t", "b"])['r_hat'].iloc[:]
        if target_accept <= 0.9999999:
            target_accept += 0.05
        tune += 2000

    k_mean = king_trace.posterior['k'].median().item()
    b_mean = king_trace.posterior['b'].median().item()
    R_c_mean = king_trace.posterior['R_c'].median().item()
    R_t_mean = king_trace.posterior['R_t'].median().item()
    k_std = king_trace.posterior['k'].std().item()
    b_std = king_trace.posterior['b'].std().item()
    R_c_std = king_trace.posterior['R_c'].std().item()
    R_t_std = king_trace.posterior['R_t'].std().item()
    king_std = king_trace.posterior['sigma'].median().item()
    C = np.log(king_trace.posterior['R_t']/king_trace.posterior['R_c']).median().item(),
    d_c = 1 + king_trace.posterior['k'].median().item()/king_trace.posterior['b'].median().item(),
    results = {
        'k_mean': k_mean,
        'b_mean': b_mean,
        'R_c_mean': R_c_mean*u.arcmin,
        'R_t_mean': R_t_mean*u.arcmin,
        'k_std': k_std,
        'b_std': b_std,
        'R_c_std': R_c_std,
        'R_t_std': R_t_std,
        'king_std': king_std,
        'C' : C
    }
    if return_trace is True:
        results['king_trace'] = king_trace
    return results
def king_profile(x, k_mean, b_mean, R_c_mean, R_t_mean):
    king_profile = k_mean * ((1 / np.sqrt(1 + (x / R_c_mean)**2)) - 
                                  (1 / np.sqrt(1 + (R_t_mean / R_c_mean)**2)))**2 + b_mean
    return king_profile
def graph_king(data, centers,prob_number=[0.5,0.6,0.7,0.8], widths=None, savefig=None,return_trace=False,
               progressbar_bayesian=False,return_results=True,log_scale=False,log_space=False,
               density_method='equip',tidal_priors=True,distances=None,return_priors=False,paper_single=False,cluster_mass=None):
    # Assuming centers is a list of tuples or SkyCoord objects with ra & dec
    centers = [SkyCoord(ra=ra_dec[0][0], dec=ra_dec[0][1], frame='icrs') for ra_dec in centers]
    if len(prob_number) == 1:
        fig, axes = plt.subplots(1, layout='constrained', figsize=(7,4.5))
        axes = np.array(axes)
    elif len(prob_number) == 2:
        fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(17.5,7))
    elif len(prob_number) == 3:
        fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(13,13))
    elif len(prob_number) == 4:
        fig, axes = plt.subplots(2, 2, layout='constrained', figsize=(15,11))
    else:
        raise ValueError("Unsupported number of probability thresholds")
    all_bayesian_results = {
        'k_mean': [],
        'b_mean': [],
        'R_c_mean': [],
        'R_t_mean': [],
        'k_std': [],
        'b_std': [],
        'R_c_std': [],
        'R_t_std': [],
        'king_std': [],
        'bg_level': [],
        'C' : [],
        'd_c' : [],
        'r_lim': []
        }
    all_traces = []
    all_priors = []
    if widths is None:
        widths = [None] * len(prob_number)
    if savefig and not savefig.endswith('/'):
        savefig += '/'
    for i, ax, width, center, distance in zip(prob_number, axes.flatten(), widths, centers,distances):
        distance = ensure_units(distance,u.kpc)
        print(str(round(i * 100)) + '%')
        probability_selection = data['probability'] >= i
        iterative_data = data[probability_selection]
        # Calculate the density annulus (assumed function provided)
        if density_method=='kde_wdith':
            density_results = density_annulus_calculator_width(iterative_data, center, width,return_radius_gen=True)
        if density_method=='equip':
            density_results = density_annulus_calculator_equip(iterative_data, center,return_radius_gen=True)
        # Perform Bayesian fitting (assumed function provided)
        if log_space:
            bayesian_results = RDP_bayesian_log_space(density_results['density_annulus'], density_results['radius_gen'],return_trace=return_trace,progressbar=progressbar_bayesian,d_center=density_results['d_center'])
        # Generate the king profile (assumed function provided)
        if tidal_priors and (log_space == False):
            priors_parameters = {}
            if cluster_mass is None:
                priors_parameters['cluster_mass'] = estimate_cluster_mass(iterative_data,distance)
            else:
                priors_parameters['cluster_mass'] = cluster_mass
            priors_parameters['galactic_radius'] = calculate_galactocentric_distance(center.ra, center.dec, distance)
            priors_parameters['distance'] = ensure_units(distance,u.kpc)
            bayesian_results = RDP_bayesian(density_results['density_annulus'], density_results['radius_gen'],return_trace=return_trace,progressbar=progressbar_bayesian,d_center=density_results['d_center'],priors_parameters=priors_parameters,return_priors=return_priors,priors=True)
        x_radius = np.arange(0, bayesian_results['R_t_mean'].value+bayesian_results['R_t_std'], 0.001)
        kp = king_profile(x_radius[x_radius < bayesian_results['R_t_mean'].value], 
                          bayesian_results['k_mean'], bayesian_results['b_mean'], 
                          bayesian_results['R_c_mean'].value, bayesian_results['R_t_mean'].value)
        kp2 = np.full(len(x_radius) - len(kp), bayesian_results['b_mean'])
        kp = np.concatenate((kp,kp2))
        half_light_radius = calculate_half_light_radius(iterative_data['Gmag'],angular_separation(iterative_data['ra'], iterative_data['dec'], center.ra, center.dec).to(u.arcmin))
        # Plotting the king profile and the observed data points
        ax.plot(x_radius, kp, label='King Model', color='fuchsia', alpha=0.7)
        ax.scatter(density_results['radius_gen'], density_results['density_annulus'], s=15, label='Density', color='blue')
        ax.errorbar(density_results['radius_gen'], density_results['density_annulus'], 
                    yerr=density_results['density_errors_annulus'], fmt='none', ecolor='k', alpha=0.5)
        # Adding vertical lines for core and tidal radii, and horizontal line for background level
        ax.axvline(bayesian_results['R_c_mean'], color='red', label=r'$R_c = {:.2f}\,[{}]$'.format(bayesian_results['R_c_mean'].value, u.arcmin), linestyle='--', linewidth=1.3)
        ax.axvline(bayesian_results['R_t_mean'], color='red', label=r'$R_t = {:.2f} \,[{}]$'.format(bayesian_results['R_t_mean'].value, u.arcmin), linestyle='dotted')
        #ax.axvline(half_light_radius, color='green', label=r'$R_{{hl}} = {:.2f} \,[{}]$'.format(half_light_radius.value, u.arcmin), linestyle='-.')
        ax.axhline(bayesian_results['b_mean'], color='green', alpha=0.8, linewidth=1, linestyle='--',label=r'$b$ parameter')
        y_min, y_max = ax.get_ylim()
        # Plotting the standard deviation as a filled area
        ax.fill_between(x_radius, 
                    kp - bayesian_results['king_std'],
                    kp + bayesian_results['king_std'], alpha=0.2, color='blue',label=r'King profile $\sigma$')
        # Plotting error regions for core and tidal radii using fill_betweenx
        ax.fill_betweenx([0.5*y_min, 1.5*y_max], 
                     bayesian_results['R_c_mean'].value - bayesian_results['R_c_std'], 
                     bayesian_results['R_c_mean'].value + bayesian_results['R_c_std'], color='red', alpha=0.1)
        ax.fill_betweenx([0.5*y_min, 1.5*y_max], 
                     bayesian_results['R_t_mean'].value - bayesian_results['R_t_std'], 
                     bayesian_results['R_t_mean'].value + bayesian_results['R_t_std'], color='red', alpha=0.1)
        # Setting labels, limits, and other plot attributes
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis='both', which='both', direction='in',labelsize=14)
        if not log_scale:
            ax.set_ylim(y_max)
            ax.set_xlabel(r'$r$ [{}]'.format(u.arcmin),fontsize=16) 
            ax.set_ylabel(r'$\rho\, \left[stars\,/ \,arcmin^{2}\right]$',fontsize=16)
        if len(prob_number) != 1:
            ax.set_title(str(round(i * 100)) + '% probability members')
        ax.legend(fontsize=11)
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=0.2*np.nanmin(density_results['radius_gen']))
            ax.set_ylim([0.5*np.nanmin(density_results['density_annulus']),1.1*y_max])
            ax.set_xlabel(r'$\log\left(r\right)$', fontsize=16) 
            ax.set_ylabel(r'$\log\left(\rho \right)$', fontsize=16) 
        for key in all_bayesian_results.keys():
            all_bayesian_results[key].append(bayesian_results[key])
        # Store traces if requested
        if return_trace:
            all_traces.append(bayesian_results['king_trace'])
        if return_priors:
            all_priors.append(bayesian_results['priors_results'])
    # Save the figure if a save path is provide
    if savefig:
        if log_space:
            plt.savefig(f'{savefig}king_profile_log.pdf', dpi=800)
        elif log_scale:
            plt.savefig(f'{savefig}king_profile_logscale.pdf', dpi=800)
        else:
            if paper_single:
                plt.savefig(f'{savefig}king_profile_linscale_paper.pdf',bbox_inches='tight')
            else:
                plt.savefig(f'{savefig}king_profile_linscale.pdf',bbox_inches='tight')
    # Return the collected results and/or traces if requested
    return_data = {}
    if return_results:
        return_data['bayesian_results'] = all_bayesian_results
        return_data['bayesian_results']['half_light_radius'] = half_light_radius
    if return_trace:
        return_data['traces'] = all_traces
    if return_priors:
        return_data['priors'] = all_priors
    plt.show()
    return return_data if return_data else None
def plot_cumulative(data,centers,prob_number=[50,60,70,80],R_c=None,R_t=None,savefig=False):
    prob_number = np.array(prob_number) / 100
    centers = [SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg') for ra, dec in centers]
    if R_c is None:
        R_c = [None] * len(prob_number)
    if R_t is None:
        R_t = [None] * len(prob_number)
    if len(prob_number) == 1:
        fig_intmag, ax_intmag = plt.subplots(1, layout='constrained', figsize=(15,15))
    if len(prob_number) == 2:
        fig_intmag, ax_intmag = plt.subplots(1,2, layout='constrained', figsize=(17.5,7))
    if len(prob_number) == 3:
        fig_intmag, ax_intmag = plt.subplots(1,3, layout='constrained', figsize=(15,15))
    if len(prob_number) == 4:
        fig_intmag, ax_intmag = plt.subplots(2,2, layout='constrained', figsize=(15,11))
    if len(prob_number) == 1:
        ax_intmag = np.array(ax_intmag)
    for i, ax, center,r_c,r_t in zip(prob_number, ax_intmag.flatten(),centers,R_c,R_t):
        print(f"{i * 100}%")
        cumulative = []
        selected_data = data[data['probability'] >= i]
        selected_data['d_center'] = (angular_separation(selected_data['ra'], selected_data['dec'], center.ra, center.dec)).to(u.arcmin)
        x_radius = np.linspace(0,np.nanmax(selected_data['d_center'].value),400)
        for r in x_radius:
            sum_i = -2.5 * np.log10(np.sum(10 ** ((selected_data[(selected_data['d_center'] <= r*u.arcmin)]['Gmag']).value / -2.5)) ) 
            cumulative.append(sum_i)
        intmag = ax.twinx()
        if r_c is None:
            ax.scatter(selected_data['d_center'],selected_data['Gmag'],s=10,color='blue')
        else:
            ax.scatter(selected_data['d_center'][selected_data['d_center'] <= r_c],selected_data['Gmag'][selected_data['d_center'] <= r_c],s=10,color='blue',label='$R \leq R_c$')
            ax.scatter(selected_data['d_center'][selected_data['d_center'] > r_c],selected_data['Gmag'][selected_data['d_center'] > r_c],s=10,color='gray',label='$R > R_c$')
        if r_c is not None:
            ax.axvline(r_c, color='green', label=r'$R_c = {:.2f} [{}]$'.format(r_c.value, u.arcmin), linestyle='-.',linewidth=2,alpha=0.8)
        if r_t is not None:
            ax.axvline(r_t, color='red', label=r'$R_t = {:.2f} [{}]$'.format(r_t.value, u.arcmin), linestyle='-.',linewidth=2,alpha=0.8)
        ax.legend(loc='lower right')
        intmag.plot(x_radius,cumulative,color='red',label='Integred Magnitude',linewidth=1)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator()),ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        intmag.xaxis.set_minor_locator(ticker.AutoMinorLocator()),intmag.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set(xlabel='Radius [{}]'.format(u.arcmin),ylabel=r'$G_{{mag}}\,[{}]$'.format(selected_data['Gmag'].unit))
        intmag.set(ylabel=r'$\sum G_{{mag}}\,[{}]$'.format(u.mag))
        ax.tick_params(axis='both', which='both', direction='in'),intmag.tick_params(axis='both', which='both', direction='in');
        intmag.legend()
        intmag.invert_yaxis(),ax.invert_yaxis()
        ax.set_title(str(round(i * 100)) + '% probability members')
        print(np.nanmax(selected_data['d_center']))
    if savefig:
        fig_intmag.savefig(savefig + 'integred_magnitude.pdf',dpi='figure')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
from scipy.stats import ks_2samp

def plot_cumulative_by_brightness(data, centers, prob_number=[50, 60, 70, 80], brightness_ranges=None, savefig=None, R_c=None, R_t=None, normalize=False, ks=True,paper=False):
    prob_number = np.array(prob_number) / 100
    centers = [SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg') for ra, dec in centers]
    num_plots = len(prob_number)
    
    if R_c is None:
        R_c = [None] * len(prob_number)
    if R_t is None:
        R_t = [None] * len(prob_number)
    
    if brightness_ranges is None:
        gmag_sorted = np.sort(data['Gmag'])
        quartiles = np.percentile(gmag_sorted, [0, 25, 50, 75, 100])
        brightness_ranges = [(quartiles[i], quartiles[i+1]) for i in range(len(quartiles)-1)]
    
    figsize = (12,8) if len(prob_number) == 2 else (8,8)
    fig, axs = plt.subplots((num_plots + 1) // 2, 2 if num_plots > 1 else 1, figsize=figsize)
    axs = axs.flatten() if num_plots > 1 else [axs]
    
    all_distances = {brightness_range: [] for brightness_range in brightness_ranges}
    
    for i, ax, center, r_c, r_t in zip(prob_number, axs, centers, R_c, R_t):
        selected_data = data[data['probability'] >= i]
        selected_data['d_center'] = angular_separation(selected_data['ra'], selected_data['dec'], center.ra, center.dec).to(u.arcmin)
        x_radius = np.linspace(0, np.nanmax(selected_data['d_center'].value), 400)
        
        cumulative_counts = {brightness_range: [] for brightness_range in brightness_ranges}
        total_counts = {brightness_range: 0 for brightness_range in brightness_ranges}
        
        for (mag_min, mag_max) in brightness_ranges:
            bright_data = selected_data[(selected_data['Gmag'] > mag_min) & (selected_data['Gmag'] <= mag_max)]
            total_counts[(mag_min, mag_max)] = len(bright_data)
            all_distances[(mag_min, mag_max)].extend(bright_data['d_center'].value)
        
        for r in x_radius:
            for (mag_min, mag_max) in brightness_ranges:
                count = len(selected_data[(selected_data['d_center'] <= r*u.arcmin) & 
                                          (selected_data['Gmag'] > mag_min) & 
                                          (selected_data['Gmag'] <= mag_max)])
                cumulative_counts[(mag_min, mag_max)].append(count)
        
        for (mag_min, mag_max), counts in cumulative_counts.items():
            if total_counts[(mag_min, mag_max)] > 0:
                normalized_counts = np.array(counts) / total_counts[(mag_min, mag_max)] if normalize else counts
                label = rf'$G_{{mag}}$: {mag_min.value:.2f} to {mag_max:.2f}'
                ax.plot(x_radius, normalized_counts, label=label)
        
        if r_c is not None and r_c.value:
            ax.axvline(r_c.value, color='green', label=rf'$R_c = {r_c.value:.2f}$ {r_c.unit}', linestyle='-.', linewidth=2, alpha=0.8,zorder=-1)
        if r_t is not None and r_t.value:
            ax.axvline(r_t.value, color='red', label=rf'$R_t = {r_t.value:.2f}$ {r_t.unit}', linestyle='-.', linewidth=2, alpha=0.8,zorder=-1)
        
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.set_xlabel('Radius [arcmin]',fontsize=16)
        ax.set_ylabel('Normalized cumulative count' if normalize else 'Cumulative count',fontsize=16)
        ax.legend()
        if not paper:
            ax.set_title(f"{int(i * 100)}% probability members")
    
    if ks:
        print("K-S test results:")
        for ((min_i, max_i), dist_i) in all_distances.items():
            for ((min_j, max_j), dist_j) in all_distances.items():
                if (min_i, max_i) < (min_j, max_j):  # Avoid comparing the same range and ensure unique pairs
                    ks_stat, ks_pvalue = ks_2samp(dist_i, dist_j)
                    print(f"Between brightness range {min_i:.2f}, {max_i:.2f} and {min_j:.2f}, {max_j:.2f}: KS-statistic={ks_stat:.2f}, p-value={ks_pvalue:.2f}")

    if savefig and paper:
        plt.savefig(savefig + 'cumulative_by_brightness_paper.pdf', bbox_inches='tight')
    else:
        plt.savefig(savefig + 'cumulative_by_brightness.pdf', bbox_inches='tight')
    plt.show()
def plot_cumulative_by_mass_and_type(data, centers, prob_number=[70], savefig=None, normalize=False, ks=True):
    prob_number = np.array(prob_number) / 100
    centers = [SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg') for ra, dec in centers]

    # Remove NaN values and sort mass data
    mass_sorted = np.sort(data['mass'][~np.isnan(data['mass'])])
    quartiles = np.percentile(mass_sorted, [0, 25, 50, 75, 100])
    mass_ranges = [(quartiles[i], quartiles[i+1]) for i in range(4)]

    # Set up figure
    figsize = (18, 6)
    fig, axs = plt.subplots(1, 3, figsize=figsize, squeeze=False,sharex=True)
    axs = axs.flatten()

    # Filter data based on probability
    data = data[data['probability'] >= prob_number[0]]

    # Calculate angular separation to cluster center
    data['d_center'] = angular_separation(data['ra'], data['dec'], centers[0].ra, centers[0].dec).to(u.arcmin)

    # Define the maximum radius for the x-axis
    max_radius = data['d_center'].max()

    # Split data into single and binary stars
    binary_data = data[data['P_binar'] >= 0.6]
    single_data = data[data['P_binar'] < 0.6]

    # Set up datasets for singles and binaries
    datasets = [('Single Stars', single_data), ('Binary Stars', binary_data)]
    ks_results = {}

    # Process each dataset
    for idx, (title, selected_data) in enumerate(datasets):
        ax = axs[idx]
        all_dists = []

        for (mass_min, mass_max) in mass_ranges:
            mask = (selected_data['mass'] >= mass_min) & (selected_data['mass'] < mass_max)
            dists = selected_data[mask]['d_center']
            # Skip the mass range if no data points are present
            if len(dists) == 0:
                continue
            all_dists.append(dists)
            # Compute cumulative distribution
            cumulative_dist = np.array([np.sum(dists <= r) for r in np.linspace(0, max_radius, 400)], dtype=np.float64)
            if normalize:
                max_cumulative_dist = np.max(cumulative_dist)
                if max_cumulative_dist > 0:  # Protect against division by zero
                    cumulative_dist /= max_cumulative_dist
            label = fr'Mass: {mass_min.value:.2f} - {mass_max.value:.2f} $M_\odot$'
            ax.plot(np.linspace(0, max_radius, 400), cumulative_dist, label=label)
        ax.set_title(title)
        ax.set_xlabel('Radius (arcmin)')
        if idx == 0:
            ax.set_ylabel('Cumulative distribution' + (' (normalized)' if normalize else ''))
        ax.legend()

        # Perform KS tests within each dataset if there's more than one mass range
        if ks and len(all_dists) > 1:
            for i in range(len(all_dists)):
                for j in range(i+1, len(all_dists)):
                    if len(all_dists[i]) > 0 and len(all_dists[j]) > 0:
                        ks_stat, ks_pvalue = ks_2samp(all_dists[i], all_dists[j])
                        ks_results[(title, f'Q{i+1} vs Q{j+1}')] = (ks_stat, ks_pvalue)

    # Perform KS test between single and binary stars if both have data
    if len(single_data['d_center']) > 0 and len(binary_data['d_center']) > 0:
        ks_stat, ks_pvalue = ks_2samp(single_data['d_center'], binary_data['d_center'])
        ks_results[('Single vs Binary Stars', 'Overall')] = (ks_stat, ks_pvalue)

    # Plot for combined single and binary stars
    ax = axs[2]
    single_cum_dist = np.array([np.sum(single_data['d_center'] <= r) for r in np.linspace(0, max_radius, 400)], dtype=np.float64)
    binary_cum_dist = np.array([np.sum(binary_data['d_center'] <= r) for r in np.linspace(0, max_radius, 400)], dtype=np.float64)
    if normalize:
        max_single = np.max(single_cum_dist)
        max_binary = np.max(binary_cum_dist)
        if max_single > 0:
            single_cum_dist /= max_single
        if max_binary > 0:
            binary_cum_dist /= max_binary
    ax.plot(np.linspace(0, max_radius, 400), single_cum_dist, label='Single Stars')
    ax.plot(np.linspace(0, max_radius, 400), binary_cum_dist, label='Binary Stars')
    ax.set_title('Single vs Binary Stars')
    ax.set_xlabel('Radius (arcmin)')
    ax.legend()

    # Setting minor ticks for the x-axis and y-axis
    for idx, ax in enumerate(axs):
        if idx != 0:
            ax.set_yticks([])
        if idx == 0:
            ax.yaxis.set_major_locator(ticker.AutoLocator())  # Auto-locate y-axis ticks
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())  # Auto-locate minor ticks
        ax.xaxis.set_major_locator(ticker.AutoLocator())  # Auto-locate x-axis ticks
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())  # Auto-locate minor tick
    plt.subplots_adjust(wspace=0, hspace=0)
    # Save the figure if a save path is provided
    if savefig:
        plt.savefig(savefig + 'cumulative_by_mass_and_type.pdf', bbox_inches='tight')
    plt.show()

    # Print KS test results
    print("K-S test results:")
    for key, (ks_stat, ks_pvalue) in ks_results.items():
        print(f"{key[0]} - {key[1]}: KS-stat={ks_stat:.2f}, p-value={ks_pvalue:.2f}")
def plot_cumulative_by_mass(data, centers, prob_number=[50, 60, 70, 80], mass_ranges=None, savefig=None, R_c=None, R_t=None, normalize=False, ks=True):
    prob_number = np.array(prob_number) / 100
    centers = [SkyCoord(ra=ra, dec=dec, frame='icrs', unit='deg') for ra, dec in centers]
    num_plots = len(prob_number)
    
    if R_c is None:
        R_c = [None] * len(prob_number)
    if R_t is None:
        R_t = [None] * len(prob_number)
    
    if mass_ranges is None:
        mass_sorted = np.sort(data['mass'])
        quartiles = np.percentile(mass_sorted, [0, 25, 50, 75, 100])
        mass_ranges = [(quartiles[i].value.__float__()*u.Msun, quartiles[i+1].value.__float__()*u.Msun) for i in range(len(quartiles)-1)]
    
    figsize = (12, 8) if len(prob_number) == 2 else (8, 8)
    fig, axs = plt.subplots((num_plots + 1) // 2, 2 if num_plots > 1 else 1, figsize=figsize, squeeze=False)
    axs = axs.flatten()
    
    ks_results = {}
    
    for idx, (i, center, r_c, r_t) in enumerate(zip(prob_number, centers, R_c, R_t)):
        ax = axs[idx]
        selected_data = data[data['probability'] >= i]
        selected_data['d_center'] = angular_separation(selected_data['ra'], selected_data['dec'], center.ra, center.dec).to(u.arcmin)
        x_radius = np.linspace(0, np.nanmax(selected_data['d_center'].value), 400)*u.arcmin
        
        for (mass_min, mass_max) in mass_ranges:
            mask = (selected_data['mass'] >= mass_min) & (selected_data['mass'] < mass_max)
            dists = selected_data[mask]['d_center']
            cumulative_dist = np.array([np.sum(dists <= r) for r in x_radius])
            
            if normalize:
                cumulative_dist = cumulative_dist / np.nanmax(cumulative_dist)
                
            label = f'Mass: {mass_min:.2f}-{mass_max:.2f} Msun'
            ax.plot(x_radius, cumulative_dist, label=label)
        
        if ks and len(mass_ranges) > 1:
            for j in range(len(mass_ranges) - 1):
                for k in range(j + 1, len(mass_ranges)):
                    dist_j = selected_data[(selected_data['mass'] >= mass_ranges[j][0]) & (selected_data['mass'] < mass_ranges[j][1])]['d_center']
                    dist_k = selected_data[(selected_data['mass'] >= mass_ranges[k][0]) & (selected_data['mass'] < mass_ranges[k][1])]['d_center']
                    # Check if we have enough data points to perform K-S test
                    if len(dist_j) > 0 and len(dist_k) > 0:
                        ks_stat, ks_pvalue = ks_2samp(dist_j.value, dist_k.value)
                        ks_results[(mass_ranges[j], mass_ranges[k])] = (ks_stat, ks_pvalue)
            
        ax.set_title(f'Prob >= {i*100}%')
        ax.set_xlabel('Radius (arcmin)')
        ax.set_ylabel('Cumulative distribution' + (' (normalized)' if normalize else ''))
        ax.legend()
    
    if savefig:
        plt.savefig(savefig + 'cumulative_by_mass.pdf', bbox_inches='tight')
    plt.show()
    
    # Print KS test results
    if ks:
        print("K-S test results:")
        for (range_i, range_j), (ks_stat, ks_pvalue) in ks_results.items():
             print(f"Between mass range {range_i} and {range_j}: KS-stat={ks_stat:.2f}, p-value={ks_pvalue:.2f}")
def radial_velocity_model(data, return_trace=False, progressbar=False):
    # Prior for the mean radial velocity
    prior_mu_rv = np.mean(data['radial_velocity'])
    with pm.Model() as rv_model:
        # Hyperprior for the mean radial velocity
        mu_rv = pm.Normal("mu_rv", mu=prior_mu_rv, sigma=10)
        
        # Standard deviation for the radial velocity
        std_rv = pm.Uniform("std_rv", lower=0, upper=40)
        
        # Likelihood for the observed radial velocity
        pm.Normal("observed_rv", mu=mu_rv, sigma=std_rv, observed=data["radial_velocity"])

        # Iterative sampling process
        max_rhat = 2
        target_accept = 0.8
        tune = 4000

        while max_rhat > 1:
            with rv_model:
                trace = pm_jax.sample_numpyro_nuts(draws=10000, tune=tune, target_accept=target_accept, progressbar=progressbar)
                # Using ArviZ to compute rhat values
                rhat_vals = az.rhat(trace)
                # Extract the maximum rhat value
                max_rhat = np.round(np.nanmax(rhat_vals.to_array()), 2)
                
            if target_accept <= 0.9999999:
                target_accept += 0.05
            tune += 2000
        results = {
            'mu_vr_mean': trace.posterior['mu_rv'].mean().item(),
            'std_vr_mean': trace.posterior['std_rv'].mean().item(),
            'mu_vr_std': trace.posterior['mu_rv'].std().item(),
            'std_vr_std': trace.posterior['std_rv'].std().item(),
        }
        if return_trace:
            results['trace'] = trace
        return results
def rv_determination(data, prob_thresholds=[50,60, 70, 80], return_trace=False, progressbar=False):
    prob_thresholds = np.array(prob_thresholds) / 100
    results = []

    # Remove rows with NaN values in the radial_velocity column
    data = data[~np.isinf(data['radial_velocity'])]

    for prob_threshold in prob_thresholds:
        print(f"Processing for probability threshold: {prob_threshold * 100}%")
        
        # Filter data based on probability threshold
        filtered_data = data[data['probability'] >= prob_threshold]

        # Call radial velocity model with filtered data
        model_results = radial_velocity_model(filtered_data, return_trace, progressbar)

        len_data = len(filtered_data)
        # Store results
        results.append({
            'probability': prob_threshold,
            'model_results': model_results,
            'len_data' : len_data
        })

    return results
def graph_real(data,image,dt=None,projection=None,savefig_dir=None,prob_number=[60,70,80,90]):
    subplot_kw = {'projection': projection}
    prob_number = np.array(prob_number)/100
    if len(prob_number) == 1:
        fig, ax_real = plt.subplots(1, layout='constrained', figsize=(15,15), subplot_kw=subplot_kw)
        ax_real = np.array(ax_real)
    if len(prob_number) == 2:
        fig, ax_real = plt.subplots(1,2, layout='constrained', figsize=(15,15), subplot_kw=subplot_kw)
    if len(prob_number) == 3:
        fig, ax_real = plt.subplots(1,3, layout='constrained', figsize=(15,15), subplot_kw=subplot_kw)
    if len(prob_number) == 4:
        fig, ax_real = plt.subplots(2,2, layout='constrained', figsize=(15,15), subplot_kw=subplot_kw)
    if projection == None:
        transform = None
    if dt == None:
        dt = 0
    for i,ax in zip(prob_number,ax_real.flatten()):
        print(str(round(i*100))+'%')
        probability_selection = (data['probability'] >= i)
        iterative_data = data[probability_selection]
        ra = iterative_data['ra'] - iterative_data['pmra']*dt
        dec = iterative_data['dec'] - iterative_data['pmdec']*dt 
        ax.scatter(ra, dec, color='lightcyan', transform=ax.get_transform('world'), alpha=0.6, s=20, label='NGC 6383 sources', facecolors='none')
        ax.imshow(image, cmap='cubehelix', aspect='equal')
        ax.set(xlabel=r'$\alpha$ [{}]'.format(data['ra'].unit), ylabel=r'$\delta$ [{}]'.format(data['dec'].unit))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.coords.grid(True, color='white', linestyle='dotted', alpha=0.6)
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_aspect('equal')
        # Overlay sources
        ax.set_facecolor('black')
        ax.set_title(str(round(int(i*101)))+'% probability members')
        overlay = ax.get_coords_overlay('galactic')
        overlay.grid(color='white', ls=':',alpha=0.6)
        #ax.set_xlim(ra.min(),ra.max())
        #ax.set_ylim(dec.min(),dec.max())
    if savefig_dir != None:
        fig.savefig(savefig_dir+'real_sky_prob.pdf',dpi=500,bbox_inches='constrained');
def FitProperMotion2DGaussian(pm_RA, pm_Dec, return_trace=False, progressbar=False):
    pm_RA = ensure_units(pm_RA, u.mas/u.yr)
    pm_Dec = ensure_units(pm_Dec, u.mas/u.yr)
    mpm_RA = np.median(pm_RA)
    mpm_DEC = np.median(pm_Dec)
    std_RA = np.std(pm_RA)
    std_DEC = np.std(pm_Dec)
    with pm.Model() as pm_model:
        # Priors for 2D Gaussian parameters
        #mu_RA = pm.Uniform('mu_RA', lower=0.5*mpm_RA,upper=1.5*mpm_RA)
        #mu_Dec = pm.Uniform('mu_Dec', lower=0.5*mpm_DEC,upper=1.5*mpm_DEC)
        mu_RA = pm.Normal('mu_RA',mu=mpm_RA,sigma=std_RA)
        mu_Dec = pm.Normal('mu_Dec',mu=mpm_DEC,sigma=std_DEC)
        sigma_RA = pm.HalfNormal('sigma_RA', sigma=std_RA)
        sigma_Dec = pm.HalfNormal('sigma_Dec', sigma=std_DEC)
        corr = pm.Uniform('corr', lower=-1, upper=1)

        # Covariance matrix
        cov = pm.math.stack([[sigma_RA**2, corr * sigma_RA * sigma_Dec],
                             [corr * sigma_RA * sigma_Dec, sigma_Dec**2]])

        # Multivariate normal likelihood
        obs = pm.MvNormal('obs', mu=pm.math.stack([mu_RA, mu_Dec]), cov=cov, observed=np.stack([pm_RA, pm_Dec], axis=1))

        # Iterative sampling parameters
        max_rhat = 2.0  
        target_accept = 0.8
        tune = 4000

        while max_rhat > 1.0:
            with pm_model:
                pm_trace = pm_jax.sample_numpyro_nuts(draws=10000, tune=tune, target_accept=target_accept, progressbar=progressbar)
                rhat_vals = az.rhat(pm_trace)
                hdi = az.hdi(pm_trace)
            # Extract the maximum rhat value
            max_rhat = round(np.nanmax(rhat_vals.to_array().values),2)
            if target_accept <= 0.9999999:
                target_accept += 0.05
            tune += 2000
        
        # Extracting the posterior statistics
        mu_RA_mean = pm_trace.posterior['mu_RA'].mean().item()
        mu_Dec_mean = pm_trace.posterior['mu_Dec'].mean().item()
        sigma_RA_mean = pm_trace.posterior['sigma_RA'].mean().item()
        sigma_Dec_mean = pm_trace.posterior['sigma_Dec'].mean().item()
        corr_mean = pm_trace.posterior['corr'].mean().item()

        mu_RA_std = pm_trace.posterior['mu_RA'].std().item()
        mu_Dec_std = pm_trace.posterior['mu_Dec'].std().item()
        sigma_RA_std = pm_trace.posterior['sigma_RA'].std().item()
        sigma_Dec_std = pm_trace.posterior['sigma_Dec'].std().item()
        corr_std = pm_trace.posterior['corr'].std().item()

        bayesian_results = {
            'mu_RA_mean': mu_RA_mean,
            'mu_Dec_mean': mu_Dec_mean,
            'sigma_RA_mean': sigma_RA_mean,
            'sigma_Dec_mean': sigma_Dec_mean,
            'corr_mean': corr_mean,
            'mu_RA_std': mu_RA_std,
            'mu_Dec_std': mu_Dec_std,
            'sigma_RA_std': sigma_RA_std,
            'sigma_Dec_std': sigma_Dec_std,
            'corr_std': corr_std
        }
        results = {'results': bayesian_results}

        if return_trace:
            results['trace'] = pm_trace

        return results