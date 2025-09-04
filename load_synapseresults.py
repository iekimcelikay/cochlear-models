# load_synapseresults.py
import os
import pickle

def load_latest_pickle(folder_path):
    # List all the pickle files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f'No pickle files found in {folder_path}')

    # Full paths
    full_paths = [os.path.join(folder_path, f) for f in files]

    # Find the most recently modified file
    latest_file = max(full_paths, key=os.path.getmtime)

    # Load and return data
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded latest pickle file: {latest_file}")
    return data

# Usage:
folder = '/home/ekim/PycharmProjects/subcorticalSTRF/subcorticalSTRF/bruce2018_results'  # your save directory
data_bundle = load_latest_pickle(folder)

# Now access your data_bundle items:
synapse_df = data_bundle['synapse_df']
print(synapse_df.head())

