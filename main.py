import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef as mcc
from load_data import load_dataset
from fl import Node
from fl_multi import FLMulti
from sklearn.model_selection import train_test_split
from utils import *
from sklearn.metrics import roc_auc_score
import time
from sklearn.model_selection import StratifiedKFold
import os
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------

def main_multi(mode, dataset, overlap_percentage, oversampling, iterations, method, n_nodes, results_dir):
    
    metrics_to_run = ['mcc', 'auc', 'bacc', 'ap', 'auprc', 'f1', 'f1_macro', 'f1_weighted']
    total_fl_results = {metric: [[] for _ in range(n_nodes)] for metric in metrics_to_run}
    total_local_results = {metric: [[] for _ in range(n_nodes)] for metric in metrics_to_run}
    jaccard_stats = []
    
    df, feature_types = load_dataset(dataset)
    label = df.columns[-1]
    potential_features = [col for col in df.columns if col != label]
    print()
    print(f'potential_features: = {len(potential_features)}')
    print()
    print(f"Dataset: {dataset}")
    print(f"Mode: Overlapping features (Multi-node) with {n_nodes} nodes")
    print(f"Method: {method}\n")
        
    for i in range(iterations):
        start_time = time.perf_counter()
        seed  = i
        print(f"--- Run {i+1}/{iterations} (seed={seed}) ---")

        ov_int = int(overlap_percentage)

        seed_for_features = seed + 100 * ov_int
        
        print(f'seed_for_features = {seed_for_features}')
        
        overlap_features = setup_features_by_jaccard(potential_features, overlap_percentage, n_nodes, seed_for_features)
        all_node_columns = get_node_feature_sets(potential_features, overlap_features, n_nodes, label, seed=seed)
      
        node_feature_sets_for_J = [[f for f in cols if f != label] for cols in all_node_columns]
        _, j_mean, j_sd = jaccard_matrix(node_feature_sets_for_J)
        j_target = float(overlap_percentage) / 100.0
        abs_err = abs(j_mean - j_target)
        rel_err = abs_err / j_target if j_target > 0 else 0.0
        print(f"  Realized Jaccard: mean={j_mean:.3f}, sd={j_sd:.3f} (target={j_target:.2f}, abs_err={abs_err:.3f})")

        jaccard_stats.append([j_target, j_mean, j_sd, abs_err, rel_err])        
        
        
        node_row_parts = partition_rows_among_nodes(df, label, n_nodes, seed=seed)

        print()
        print(f'overlap features: {overlap_features}')
        nodes = []
        for idx in range(n_nodes):
            df_node_full = node_row_parts[idx][all_node_columns[idx]].copy()

            df_train_node, df_test_node = train_test_split(
                df_node_full, test_size=0.25, stratify=df_node_full[label], random_state=seed
            )

            x_train = df_train_node.drop(label, axis=1)
            y_train = df_train_node[label]
            x_test = df_test_node.drop(label, axis=1)
            y_test = df_test_node[label]

            node = Node(x_train, y_train, x_test, y_test, x_train.columns, idx, overlap_features, feature_types)
            nodes.append(node)
            
# --------------------------------------------------------------------------------------------------------------------------------------------------------------         
        
        if method == "baseline_intersection":
            ok, G = update_nodes_for_baseline(nodes)
            if not ok:
                print("  - Baseline undefined for this run (|G|=0). Skipping this run.")
                continue
                
            overlap_features = G        
     
        
        if method == 'model_imputation':
            train_and_share_imputation_models(nodes, overlap_features)
            

        elif method in {
            "informed_marginal_cat_simple",
            "probability_informed_cat_simple_fallback_fixed",
        }:            
            train_and_share_imputation_models(nodes, overlap_features)
            assign_other_features(nodes)

        for node in nodes:
            node.define_rf()
            node.train(oversampling)
            
# --------------------------------------------------------------------------------------------------------------------------------------------------------------           
        
        if method == 'surrogate_split_agreement_pr_01':
            all_surrogate_scores = [] 
            for node in nodes:
                per_tree_surrogate_counts = []
                node_scores = []
                for tree in node.rf.estimators_:
                    surrogates, scores = compute_surrogates_for_tree_agreement(
                        tree,
                        node.x_train,
                        list(node.feature_names),
                        overlap_features,
                        min_score=0.25
                    )
                    per_tree_surrogate_counts.append(len(surrogates))
                    tree.surrogates = surrogates
                    node_scores.extend(scores)
                all_surrogate_scores.extend(node_scores)
                
                if per_tree_surrogate_counts:
                    print(f"  Node {node.id}: Found avg {np.mean(per_tree_surrogate_counts):.1f} surrogates per tree.")
        
                if not node_scores:
                    print(f"  No surrogates were found for node {node.id}")
                else:
                    print(f"  Found {len(node_scores)} surrogates.")
                    print(f"  Mean quality (accuracy score): {np.mean(node_scores):.4f}")
                    print(f"  Max quality: {np.max(node_scores):.4f}")
                    print(f"  Min quality: {np.min(node_scores):.4f}")

            print(f"\nOverall surrogate quality for this round: Mean = {np.mean(all_surrogate_scores):.4f}")      
            
# --------------------------------------------------------------------------------------------------------------------------------------------------------------            
  
        for idx, node in enumerate(nodes):
            class_labels_local = list(node.rf.classes_)
            if len(class_labels_local) < 2:

                total_local_results['mcc'][idx].append(0.0)
                total_local_results['bacc'][idx].append(0.5)
                total_local_results['auc'][idx].append(0.5)
                total_local_results['ap'][idx].append(0.0)
                total_local_results['auprc'][idx].append(0.0)
                total_local_results['f1'][idx].append(0.0)
                total_local_results['f1_macro'][idx].append(0.0)
                total_local_results['f1_weighted'][idx].append(0.0)
                continue

            y_pred_labels_local = node.rf.predict(node.x_test)
            probas_local = node.rf.predict_proba(node.x_test) 
            scores_local = compute_all_metrics(node.y_test, y_pred_labels_local, probas_local, class_labels_local)

            for metric in metrics_to_run:
                total_local_results[metric][idx].append(scores_local[metric])                
                
# -------------------------------------------------------------------------------------------------------------------------------------------------------------- 

        fl_multi = FLMulti(nodes)
        fl_multi.define_rf()
        fl_multi.broadcast_model()
        
# --------------------------------------------------------------------------------------------------------------------------------------------------------------    

        for node in nodes:
            node.inference_stats = {'surrogates_used': 0, 'fallbacks_triggered': 0}
            
            
            
        for idx, node in enumerate(nodes):
            all_fl_scores = node.score(method)

            for metric_name, score_value in all_fl_scores.items():
                if metric_name in total_fl_results:
                    total_fl_results[metric_name][idx].append(score_value)
          
        
        print("\n  Inference Stats for this Round:")
        for node in nodes:
            used = node.inference_stats['surrogates_used']
            fallback = node.inference_stats['fallbacks_triggered']
            total = used + fallback
            if total > 0:
                sur_percent = 100 * used / total
                print(f"  Node {node.id}: Surrogates Used = {used}, Fallbacks = {fallback} ({sur_percent:.1f}% surrogate usage)")        
        
        
        mean_fl_mcc    = np.mean([total_fl_results['mcc'][j][i]    for j in range(n_nodes)])
        mean_fl_auc    = np.mean([total_fl_results['auc'][j][i]    for j in range(n_nodes)])
        mean_fl_auprc    = np.mean([total_fl_results['auprc'][j][i]    for j in range(n_nodes)])
        mean_fl_f1_macro    = np.mean([total_fl_results['f1_macro'][j][i]    for j in range(n_nodes)])
        
        
        print(f"\n  FL mean across nodes this run:    MCC={mean_fl_mcc:.3f}, AUC={mean_fl_auc:.3f}, AUPRC={mean_fl_auprc:.3f}, F1-Macro={mean_fl_f1_macro:.3f}")
        print(f'-----------------------------------------------------------------------------------------')

        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Round {i+1} completed in {duration:.2f} seconds.\n")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------        
        
    save_jaccard_stats(dataset, n_nodes, overlap_percentage, jaccard_stats)    
        
        
        
    for metric_name in metrics_to_run:
        metric_root_folder = f"results_{metric_name}"
 
        metric_results_dir = os.path.join(metric_root_folder, dataset, f"p{n_nodes}_{int(overlap_percentage)}")
        
        export_results(
            total_fl_results[metric_name],
            total_local_results[metric_name],
            mode,
            method,
            metric_results_dir
        )

# --------------------------------------------------------------------------------------------------------------------------------------------------------------
