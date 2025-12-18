from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import math
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score

from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_curve, auc, f1_score, matthews_corrcoef as mcc
)
from sklearn.preprocessing import label_binarize




def calculate_overlap_for_jaccard(total_features, overlap_percentage, num_peers):
    """
    Calculating the number of overlapping features required for a target Jaccard similarity.
    """
    target_jaccard = float(overlap_percentage) / 100.0
    
    if target_jaccard >= 1.0:
        return total_features
    if target_jaccard <= 0.0:
        return 0

    # J = O / (O + 2P)
    # F_total = O + N*P
    # O = F_total / (1 + N * (0.5/J - 0.5))
    
    denominator = 1 + num_peers * (0.5 / target_jaccard - 0.5)
    if denominator <= 0:
        return 0
        
    num_overlap = total_features / denominator
    
    return max(0, int(round(num_overlap)))



def setup_features_by_jaccard(potential_features, overlap_percentage, n_nodes, seed):

    rng = np.random.default_rng(seed)
    
    num_overlap = calculate_overlap_for_jaccard(len(potential_features), overlap_percentage, n_nodes)

    shuffled_features = rng.permutation(potential_features)
    overlap_features = list(shuffled_features[:num_overlap])
    
    return overlap_features



def jaccard_matrix(node_feature_sets):
    
    """Pairwise Jaccard + mean, sd"""
    
    P = len(node_feature_sets)
    sets = [set(s) for s in node_feature_sets]
    M = np.zeros((P, P), dtype=float)
    vals = []
    for i in range(P):
        for j in range(P):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            M[i, j] = inter / union if union else 0.0
            if i < j:
                vals.append(M[i, j])
    vals = np.array(vals, dtype=float)
    return M, float(vals.mean()) if vals.size else 0.0, float(vals.std(ddof=0)) if vals.size else 0.0



def traverse_tree(rf, sample_tree, node_index):
    if node_index != -1:
        feature_index = sample_tree.tree_.feature[node_index]
        feature_name = rf.get_feature_name_from_node(feature_index)

        left_child = sample_tree.tree_.children_left[node_index]
        right_child = sample_tree.tree_.children_right[node_index]

        traverse_tree(rf, sample_tree, left_child)
        traverse_tree(rf, sample_tree, right_child)
        

def offset_feature_indexes(tree, feature_offset):
    
    """Offset feature indexes for all nodes in the tree"""
    
    for node_id in range(tree.node_count):
        if tree.feature[node_id] != _tree.TREE_UNDEFINED:
            tree.feature[node_id] += feature_offset    
           


    
def export_results(results_fl, results_local, mode, method, results_dir):

    os.makedirs(results_dir, exist_ok=True)
    
    prefix = "all" if mode == "all" else method
    n_nodes = len(results_fl)
    for i in range(n_nodes):
        filename_fl = os.path.join(results_dir, f"{prefix}_fl_{i+1}.npy")
        filename_local = os.path.join(results_dir, f"{prefix}_local_{i+1}.npy")
        np.save(filename_fl, np.array(results_fl[i]))
        np.save(filename_local, np.array(results_local[i]))


        
def assign_other_features(nodes):
    """
    Computig the global mean and std dev of each feature averaged across
    nodes that have that feature
    """

    node_stats = []
    for node in nodes:
        stats = {}

        for col in node.x_train.columns:
            if node.feature_types.get(col) == 'numeric':
                stats[col] = {'mean': node.x_train[col].mean(), 'std': node.x_train[col].std()}

        node_stats.append(stats)
         
    all_features = set()
    for s in node_stats:
        all_features.update(s.keys())
    

    global_stats = {}
    for feat in all_features:
        means = [s[feat]['mean'] for s in node_stats if feat in s]
        stds = [s[feat]['std'] for s in node_stats if feat in s]
        
        if means:
            global_stats[feat] = {
                'mean': sum(means) / len(means),
                'std': sum(stds) / len(stds)
            }

            if global_stats[feat]['std'] == 0:
                global_stats[feat]['std'] = 1e-6

    for node in nodes:
        node.other_features_stats = {feat: global_stats[feat] for feat in global_stats if feat not in node.feature_names}
        


def compute_surrogates_for_tree_agreement(
    tree,
    X: pd.DataFrame,
    feature_names: list,
    overlap_features: list,
    min_score: float = 0.0,   # now interpreted as MINIMUM ADJUSTED AGREEMENT (recommend 0.0)
    min_node_n: int = 6
):
    """
    For each internal node in `tree`, find ONE surrogate split from OVERLAPPING features
    that best mimics that node’s primary split using CART-style ADJUSTED AGREEMENT.

    Stores a single rule per node:
        surrogates[node_idx] = {
            'feature': <str>,
            'threshold': <float>,          # mid-point threshold
            'invert': <bool>,              # True if reverse direction agrees better
            'adj': <float>                 # adjusted agreement in [0,1]
        }

    Returns:
        surrogates: dict[node_idx] -> rule dict
        surrogate_scores: list of adjusted-agreement scores (for reporting)
    """
    surrogates = {}
    surrogate_scores = []

    # Map: which training samples reached which nodes
    node_indicator = tree.decision_path(X.values)
    training_idx = X.index

    for node_idx in range(tree.tree_.node_count):
        primary_fidx = tree.tree_.feature[node_idx]
        if primary_fidx < 0:  # leaf
            continue

        mask = node_indicator[:, node_idx].toarray().ravel().astype(bool)
        idx_at_node = training_idx[mask]
        if len(idx_at_node) < min_node_n:
            continue

        # Primary split (convention: go RIGHT iff value > threshold)
        primary_feature = feature_names[primary_fidx]
        primary_thr = tree.tree_.threshold[node_idx]
        Xn = X.loc[idx_at_node]

        # Primary left/right labels for node samples: 1=right, 0=left
        y = (Xn[primary_feature].to_numpy() > primary_thr).astype(int)
        n_total = y.size
        n_pos = int(y.sum())
        n_neg = n_total - n_pos
        n_maj = max(n_pos, n_neg)
        if n_total == n_maj:
            # Degenerate node (pure / majority rule equals total) -> no meaningful surrogate
            continue

        best_feat = None
        best_thr = None
        best_invert = False
        best_adj = -1.0

        for g in overlap_features:
            if g == primary_feature:
                continue

            # v = Xn[g].to_numpy()
            v = Xn[g].to_numpy(dtype=np.float64, na_value=np.nan)
            # Drop NaNs for this candidate
            valid = ~np.isnan(v)
            if valid.sum() < 2:
                continue
            v = v[valid]
            y_g = y[valid]

            # Need at least two distinct values to split
            order = np.argsort(v, kind="mergesort")
            v_sorted = v[order]
            y_sorted = y_g[order]
            diff = np.diff(v_sorted)
            cut_positions = np.where(diff != 0)[0]
            if cut_positions.size == 0:
                continue

            # Prefix sums for fast agreement computation
            pos_prefix = np.cumsum(y_sorted)                # positives up to k
            neg_prefix = np.cumsum(1 - y_sorted)            # negatives up to k
            total_pos = int(pos_prefix[-1])
            total_neg = int(neg_prefix[-1])

            # Constants for adjusted agreement at this node (w.r.t. original y, not y_g)
            # Use full-node n_maj/n_total (common practice); you could recompute on 'valid' only if desired.
            for k in cut_positions:
                # Standard direction: go right if v > theta
                left_neg  = int(neg_prefix[k])
                right_pos = int(total_pos - pos_prefix[k])
                agree_std = left_neg + right_pos

                # Reverse direction: go right if v <= theta
                left_pos  = int(pos_prefix[k])
                right_neg = int(total_neg - neg_prefix[k])
                agree_rev = left_pos + right_neg

                if agree_std >= agree_rev:
                    n_surr = agree_std
                    invert = False
                else:
                    n_surr = agree_rev
                    invert = True

                # Adjusted agreement (CART): (n_surr - n_maj) / (n_total - n_maj)
                # Note: n_total and n_maj from the primary labels at this node.
                adj = (n_surr - n_maj) / (n_total - n_maj)

                if adj > best_adj:
                    best_adj = adj
                    best_invert = invert
                    # Threshold is midpoint between adjacent distinct values
                    best_thr = (v_sorted[k] + v_sorted[k + 1]) / 2.0
                    best_feat = g

        if best_feat is not None and best_adj >= min_score:
            # print('yes')
            surrogates[node_idx] = {
                "feature": best_feat,
                "threshold": float(best_thr),
                "invert": bool(best_invert),
                "adj": float(best_adj),
            }
            surrogate_scores.append(best_adj)

    return surrogates, surrogate_scores



    
    

def get_node_feature_sets(potential_features, overlap_features, n_nodes, label, seed=None):
    
    """
    If non-overlap >= n_nodes: split them evenly.
    If non-overlap <  n_nodes: give each peer exactly one private feature by
      repeating features as evenly as possible
    """
    
    rng = np.random.default_rng(seed)

    non_overlap = [f for f in potential_features if f not in overlap_features]

    if not non_overlap:
        return [list(overlap_features) + [label] for _ in range(n_nodes)]

    if len(non_overlap) >= n_nodes:
        non = np.array(non_overlap, dtype=object)
        rng.shuffle(non)
        chunks = np.array_split(non, n_nodes)
        return [list(ch) + list(overlap_features) + [label] for ch in chunks]

    m = len(non_overlap)
    base = n_nodes // m          # everyone gets this many copies
    rem  = n_nodes % m           # this many features get one extra copy

    # choose which features get the extra copy
    order = rng.permutation(m)
    counts = np.full(m, base, dtype=int)
    counts[order[:rem]] += 1 

    privates = []
    for idx, c in enumerate(counts):
        privates.extend([non_overlap[idx]] * c)

    rng.shuffle(privates)

    chunks = [[privates[i]] for i in range(n_nodes)]
    return [ch + list(overlap_features) + [label] for ch in chunks]




def partition_rows_among_nodes(dataset, label_column, n_nodes, seed):
    """
    Stratified split of DF's rows into N partitions.
    Only rows, not columns!
    """
    skf = StratifiedKFold(n_splits=n_nodes, shuffle=True, random_state=seed)
    
    X = dataset.drop(columns=[label_column])
    y = dataset[label_column]
    

    partition_indices = [indices for _, indices in skf.split(X, y)]
    
    node_partitions = [dataset.iloc[indices].copy() for indices in partition_indices]
    
    return node_partitions
    
    
    
# def compute_multivariate_routers_for_tree(
#     tree,
#     rf,
#     X_train_df: pd.DataFrame,
#     overlap_features,
#     min_node_abs: int = 6,     
#     min_node_frac: float = 0.015,  
#     min_samples_leaf: int = 3,
#     topk: int | None = None,       
#     random_state: int = 42,
#     method: str | None = None, 
# ):
    
#     routers = {}
#     t = tree.tree_
# # ---------------------------------------------Sanity Check-----------------------------------------------------

#     if not hasattr(rf, "router_elig_counts"):
#         rf.router_elig_counts = {
#             "missing_nodes_total": 0,      # RF nodes that split on a non-overlap feature
#             "present_nodes_total": 0,      # RF nodes that split on an overlap feature
#             "too_small_n": 0,              # skipped: node had < min_needed samples
#             "after_nan_too_small": 0,      # skipped: after dropping NaNs in overlaps
#             "no_overlap_features": 0,      # skipped: 0 usable overlap features
#             "fit_error": 0,                # skipped: model.fit failed
#             "trained_on_missing": 0        # routers actually trained on missing-split nodes
#         }

#     if not hasattr(rf, "router_choice_counts"):
#         rf.router_choice_counts = {
#             "trained_total": 0,
#             "logreg": 0,
#             "tree_d1": 0,
#             "tree_d2": 0,
#             "tree_d3": 0,
#         }        

# # --------------------------------------------------------------------------------------------------------------

#     node_indicator = tree.decision_path(X_train_df.values).tocoo()
#     samples_by_node = {}
#     for s, n in zip(node_indicator.row, node_indicator.col):
#         if t.feature[n] >= 0: 
#             samples_by_node.setdefault(n, []).append(s)

#     N_site = len(X_train_df)
#     rng = np.random.default_rng(random_state)

#     if topk is None:
#         ov_n = len(overlap_features)
#         topk = max(3, min(8, int(np.sqrt(max(1, ov_n)))))

#     ov_feats_list = list(overlap_features)

#     for node_id, idxs in samples_by_node.items():
#         feat_idx = t.feature[node_id]
#         if feat_idx < 0:
#             continue

#         feat_name = rf.get_feature_name_from_node(feat_idx)
        
# # ---------------------------------------------Sanity Check-----------------------------------------------------
#         is_missing_candidate = (feat_name not in overlap_features)
#         if is_missing_candidate:
#             rf.router_elig_counts["missing_nodes_total"] += 1
#         else:
#             rf.router_elig_counts["present_nodes_total"] += 1  
# # ---------------------------------------------Sanity Check-----------------------------------------------------        

#         thr = t.threshold[node_id]

#         if feat_name not in X_train_df.columns:
#             continue

#         n_node = len(idxs)
#         min_needed = max(min_node_abs, int(np.ceil(min_node_frac * N_site)))
        
# # ---------------------------------------------Sanity Check-----------------------------------------------------
#         if is_missing_candidate and n_node < min_needed:
#             rf.router_elig_counts["too_small_n"] += 1        
# # ---------------------------------------------Sanity Check-----------------------------------------------------        
        
#         if n_node < min_needed:
#             continue

#         X_node = X_train_df.iloc[idxs]
#         y_node = (X_node[feat_name].values > thr).astype(int)  # right=1, left=0

#         X_router = X_node[ov_feats_list].copy()
# # ---------------------------------------------Sanity Check-----------------------------------------------------        
#         if is_missing_candidate and X_router.shape[1] == 0:
#             rf.router_elig_counts["no_overlap_features"] += 1
#             continue        
# # ---------------------------------------------Sanity Check-----------------------------------------------------        
#         mask = ~X_router.isna().any(axis=1)
#         X_router = X_router[mask]
#         y_router = y_node[mask.values]
# # ---------------------------------------------Sanity Check-----------------------------------------------------          
#         if is_missing_candidate and len(y_router) < min_needed:
#             rf.router_elig_counts["after_nan_too_small"] += 1
# # ---------------------------------------------Sanity Check-----------------------------------------------------          
#         if len(y_router) < min_needed:
#             continue

#         feats_used = list(X_router.columns)
#         if len(feats_used) > topk:
#             try:
#                 mi = mutual_info_classif(X_router.values, y_router, random_state=random_state)
#                 order = np.argsort(mi)[::-1][:topk]
#                 feats_used = [feats_used[i] for i in order]
#                 X_router = X_router[feats_used]
#             except Exception:

#                 order = rng.choice(len(feats_used), size=topk, replace=False)
#                 feats_used = [feats_used[i] for i in order]
#                 X_router = X_router[feats_used]

#         p = X_router.shape[1]
#         N_min_for_DT = 10
#         k = 3.5

#         use_logreg = (n_node < N_min_for_DT) or (n_node < k * p)        

#         if use_logreg:
#             mdl = LogisticRegression(
#                 penalty="l2",
#                 C=1.0,
#                 solver="lbfgs",
#                 class_weight="balanced",
#                 max_iter=1000,
#                 random_state=random_state
#             )
# # ---------------------------------------------Sanity Check-----------------------------------------------------            
#             depth_used = None
# # ---------------------------------------------Sanity Check-----------------------------------------------------    
#         else:
#             if method == "multivariate_surrogate_pr":
#                 max_depth = 1 if n_node < 30 else 2

#             elif method == "multivariate_surrogate":
#                 max_depth = 1 if n_node < 30 else 2
                
#             elif method == "multivariate_surrogate_1d":
#                 max_depth = 1
            
#             elif method == "multivariate_surrogate_3d":
#                 if n_node >= 80:
#                     max_depth = 3
#                 elif n_node >= 30:
#                     max_depth = 2
#                 else:
#                     max_depth = 1
            
                
            
            
#             mdl = DecisionTreeClassifier(
#                 max_depth=max_depth,
#                 min_samples_leaf=min_samples_leaf,
#                 class_weight="balanced",
#                 random_state=random_state
#             )
# # ---------------------------------------------Sanity Check-----------------------------------------------------            
#             depth_used = max_depth
# # ---------------------------------------------Sanity Check-----------------------------------------------------
#         try:
#             mdl.fit(X_router, y_router)
# # ---------------------------------------------Sanity Check-----------------------------------------------------             
#             if use_logreg:
#                 rf.router_choice_counts["logreg"] += 1
#             else:
#                 if depth_used == 1:
#                     rf.router_choice_counts["tree_d1"] += 1
#                 elif depth_used == 2:
#                     rf.router_choice_counts["tree_d2"] += 1
#                 else:   
#                     rf.router_choice_counts["tree_d3"] += 1
#             rf.router_choice_counts["trained_total"] += 1
# # ---------------------------------------------Sanity Check-----------------------------------------------------            
#         except Exception:
# # ---------------------------------------------Sanity Check-----------------------------------------------------  
#             if is_missing_candidate:
#                 rf.router_elig_counts["fit_error"] += 1        
# # ---------------------------------------------Sanity Check-----------------------------------------------------          
#             continue

#         if mdl is None:
#             print(f"[ROUTER TRAIN ERR] mdl is None at node {node_id} (n_node={n_node}). Skipping.", flush=True)
#             continue    
    
#         routers[node_id] = {
#             "model": mdl,
#             "features": feats_used,
#             "n": int(len(y_router)),
#             "max_depth": int(depth_used) if depth_used is not None else 0  # 0 for LR
#         }
# # ---------------------------------------------Sanity Check-----------------------------------------------------
#         if is_missing_candidate:
#             rf.router_elig_counts["trained_on_missing"] += 1        
# # ---------------------------------------------Sanity Check-----------------------------------------------------        
        
#     return routers    
    
    
def normalize_distribution(counts, n_classes: int | None = None) -> np.ndarray:

    arr = np.asarray(counts, dtype=float).ravel()

    if n_classes is not None and arr.size != n_classes:
        out = np.zeros(n_classes, dtype=float)
        m = min(n_classes, arr.size)
        out[:m] = arr[:m]
        arr = out

    total = arr.sum()
    if not np.isfinite(total) or total <= 0:
        k = n_classes if n_classes is not None else (arr.size if arr.size > 0 else 1)
        return np.full(k, 1.0 / k, dtype=float)

    return arr / total    
    

    
def train_and_share_imputation_models(nodes, overlap_features):
    
    """
    Each node trains imputation models for its private (non-overlap) features
    using overlap_features as inputs.
    """
    
    print("\n--- Training and Sharing Imputation Models ---")

    # Phase 1: train on each node
    for node in nodes:
        unique_features = set(node.x_train.columns) - set(overlap_features)
        node.imputation_models_to_share = {}

        if not unique_features:
            print(f"Node {node.id} has no unique features to model.")
            continue

        print(f"Node {node.id} is training imputation models for: {list(unique_features)}")
        for feature_to_predict in unique_features:
            X_imp = node.x_train[overlap_features]
            y_imp = node.x_train[feature_to_predict]
            
            feature_type = node.feature_types.get(feature_to_predict)

            if feature_type == 'categorical':
                
                if y_imp.nunique() < 2:
                    print(f"  - Warning: Skipping imputer for '{feature_to_predict}' because its training data contains only one class.")
                    continue         
                    
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(random_state=42, max_iter=2000)
                )
                is_numeric_flag = False
                score = float('nan')
                try:
                    min_class_count = y_imp.value_counts().min()
                    n_splits = min(3, min_class_count)
                    print(f'\n n_splis = {n_splits}\n')
                    score = float(np.mean(cross_val_score(model, X_imp, y_imp, cv=n_splits, scoring='f1_macro')))

                except Exception as e:
                    print(f"  - Warning: Could not calculate cross-val score for categorical '{feature_to_predict}'. Error: {e}")                 
                    
                    
            elif feature_type == 'numeric':
                model = LinearRegression()
                is_numeric_flag = True
                score = float('nan')
                try:
                    score = float(np.mean(cross_val_score(model, X_imp, y_imp, cv=3, scoring='r2')))
                except Exception as e:
                    print(f"  - Warning: Could not calculate cross-val score for numeric '{feature_to_predict}'. Error: {e}")
            
            else:
                raise ValueError(
                    f"Unknown or missing feature type for '{feature_to_predict}'. "
                    f"Found: '{feature_type}'. Expected 'categorical' or 'numeric'."
                )

            model.fit(X_imp, y_imp)
            
            model_classes = None
            if not is_numeric_flag:
                model_classes = model.steps[-1][1].classes_
        
            node.imputation_models_to_share[feature_to_predict] = {
                "model": model,
                "score": score,
                "sender_id": node.id,
                "n_train": int(len(node.x_train)),
                "is_numeric": is_numeric_flag,
                "classes": model_classes
            }

            print(node.imputation_models_to_share[feature_to_predict].get("model"))

    # Phase 2: share to all other nodes
    for receiving_node in nodes:
        receiving_node.received_imputation_models = {}
        for sending_node in nodes:
            if receiving_node.id == sending_node.id:
                continue
            for feat, payload in sending_node.imputation_models_to_share.items():
                receiving_node.received_imputation_models.setdefault(feat, []).append(payload)

    print("--- Imputation Model Setup Complete ---\n") 
    
    
    
def save_jaccard_stats(dataset, n_nodes, overlap_percentage, jaccard_stats, root="results_jaccard"):
    
    """
    Save realized Jaccard stats per run for a scenario.

    jaccard_stats shaped: (iterations, 5): [target_J, realized_mean_J, realized_sd_J, abs_err, rel_err] per run
    """
    
    jdir = os.path.join(root, dataset, f"p{n_nodes}_{int(overlap_percentage)}")
    os.makedirs(jdir, exist_ok=True)
    np.save(os.path.join(jdir, "jaccard_per_run.npy"), np.array(jaccard_stats, dtype=float))   
    
    
    
def compute_all_metrics(y_true, y_pred_labels, y_pred_probas_dist, class_labels):

    scores = {}

    scores['bacc'] = balanced_accuracy_score(y_true, y_pred_labels)
    scores['mcc']  = mcc(y_true, y_pred_labels)

    n_classes = len(class_labels)

    if n_classes > 2:

        scores['auc'] = roc_auc_score(y_true, y_pred_probas_dist, multi_class='ovr')

        y_true_binarized = label_binarize(y_true, classes=class_labels)
        scores['ap'] = average_precision_score(y_true_binarized, y_pred_probas_dist, average='macro')

        probas = np.array(y_pred_probas_dist)
        auprc_scores = []
        for c in range(n_classes):
            prec, rec, _ = precision_recall_curve(y_true_binarized[:, c], probas[:, c])
            auprc_scores.append(auc(rec, prec))
        scores['auprc'] = float(np.mean(auprc_scores))

        scores['f1_macro']    = f1_score(y_true, y_pred_labels, average='macro')
        scores['f1_weighted'] = f1_score(y_true, y_pred_labels, average='weighted')
        scores['f1']          = scores['f1_weighted']

    else:

        positive_label = class_labels[1]
        positive_class_probs = np.asarray(y_pred_probas_dist)[:, 1]

        scores['auc'] = roc_auc_score(y_true, positive_class_probs)
        scores['ap']  = average_precision_score(y_true, positive_class_probs, pos_label=positive_label)

        prec, rec, _  = precision_recall_curve(y_true, positive_class_probs, pos_label=positive_label)
        scores['auprc'] = auc(rec, prec)

        scores['f1']          = f1_score(y_true, y_pred_labels, pos_label=positive_label)
        scores['f1_macro']    = f1_score(y_true, y_pred_labels, average='macro')
        scores['f1_weighted'] = f1_score(y_true, y_pred_labels, average='weighted')

    return scores    
    
    
    
    
def update_nodes_for_baseline(nodes):
    
    """
    Restrict every node to the global intersection G
    of their current x_train columns.
    
    """

    feature_sets = [set(n.x_train.columns) for n in nodes]
    if not feature_sets:
        return False, []
    G = set.intersection(*feature_sets) if len(feature_sets) > 1 else feature_sets[0]
    G = sorted(list(G))
    print(f"\n--- INTERSECTION_BASELINE ---")
    print(f"Global intersection |G|={len(G)}")

    if len(G) == 0:
        return False, []

    for node in nodes:
        node.x_train = node.x_train[G].copy()
        node.x_test  = node.x_test[G].copy()
        node.feature_names = node.x_train.columns

        if hasattr(node, "feature_types") and isinstance(node.feature_types, dict):
            node.feature_types = {f: node.feature_types[f] for f in G if f in node.feature_types}

        if hasattr(node, "categorical_features"):
            node.categorical_features = [f for f, t in node.feature_types.items() if t == 'categorical']
        if hasattr(node, "numeric_features"):
            node.numeric_features = [f for f, t in node.feature_types.items() if t == 'numeric']

        node.overlap_features = G

    return True, G    
