from COSMIC_alpha import load_data, quality_selection, correct_zp_parallax, correct_pm, HDBSCAN_COSMIC
from astropy.io import ascii
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load cosmic data.')

    # Required argument
    parser.add_argument('data_file', type=str, help='Data file path')

    # Optional arguments
    parser.add_argument('--format', type=str, default='ascii.ecsv', help='File format, default is ascii.ecsv')
    parser.add_argument('--fidelity_file', type=str, help='Fidelity file path, default is None')
    parser.add_argument('--save_clustered', action='store_true', help='Save clustered file, default is False')
    parser.add_argument('--check_cluster', action='store_true', help='Check cluster, default is False')

    args = parser.parse_args()

    data = load_data(args.data_file, args.format, args.fidelity_file)
    print('Data loaded')

    data = quality_selection(data, threshold_fidelity=0.5)

    data = correct_zp_parallax(data)

    data = correct_pm(data)
    print('Doing the clustering')
    data = HDBSCAN_COSMIC(data, algorithm='best', metric='euclidean', allow_single_cluster=True, min_cluster_size=24, cluster_selection_method='eom')
    if args.save_clustered == True:
        ascii.write(data,'clustered_file.ecsv',format='ecsv',overwrite=True) # Create the clustered file
    if args.check_cluster == True:
        cluster_number = check_cluster(data)
    data = data[data['cluster'] == cluster_number
    print('Obtaining probability plot...')
    plot_probabilities(data, save_path=None)

    # Define probably and hp members
    probably_members = ((data['probability'] < 0.8) & (data['probability'] >= 0.6))
    hp_members = (data['probability'] >= 0.8)
    
    
    