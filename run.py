import argparse
from main import main_multi
from load_data import load_dataset
import json
import os
import numpy as np 
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, help="Name of the dataset", default="maternal_health")
parser.add_argument("--mode", type=str, help="Mode of training (overlap / all)", default="overlap")
parser.add_argument("--method", type=str, help="Method for missing features (intersection, imputation, probability, etc.)", default="probability")
parser.add_argument("--nodes", type=int, help="Number of nodes", default=2)
parser.add_argument("--overlap_percentage", type=float, help="Percentage of features to be overlapping (e.g., 20 for 20%)", default=25.0)

args = vars(parser.parse_args())
 
data_arg = args["dataset"]
mode_arg = args["mode"]
method_arg = args["method"]
n_nodes_arg = args["nodes"]
overlap_percentage_arg = args["overlap_percentage"]

iterations_val = 100
print('____________________________________________________________________________________\n')
print(f'Starting the simulation for dataset: {data_arg}')
print(f'Overlap percentage for features: {overlap_percentage_arg}%') # alinged with Jaccard similarity Overlap/100
print()


df_original, feature_types_dict = load_dataset(data_arg)
if df_original is None or df_original.empty:
    print(f"Error: Dataset '{data_arg}' could not be loaded or is empty. Exiting.")
    exit()

all_columns = list(df_original.columns)
if not all_columns:
    print(f"Error: Dataset '{data_arg}' has no columns. Exiting.")
    exit()
    
label_column_name = all_columns[-1] # Assuming the last column is always the label


oversampling_val = False # Optional for some datasets


experiment_info = {
    "Dataset": data_arg,
    "Number of nodes": n_nodes_arg,
    "Oversampling": oversampling_val,
    "Iterations": iterations_val,
    "Overlap Percentage Setting": f"{overlap_percentage_arg}%",
    "List of all features": all_columns,
    "Label Column": label_column_name
}

results_dir = os.path.join('results', data_arg, f"p{n_nodes_arg}_{int(overlap_percentage_arg)}")
if not os.path.exists(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    
info_filename_base = f"{data_arg}_{method_arg}_{n_nodes_arg}_overlap{int(overlap_percentage_arg)}pct"
info_filename_json = os.path.join(results_dir, f"{info_filename_base}_experiment_info.json")


with open(info_filename_json, 'w') as f:
    json.dump(experiment_info, f, indent=4)


if mode_arg == 'overlap':
    main_multi(mode_arg, data_arg, overlap_percentage_arg, oversampling_val, iterations_val, method_arg, n_nodes_arg, results_dir)

        

