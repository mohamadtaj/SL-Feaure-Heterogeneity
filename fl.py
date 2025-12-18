import numpy as np
from imblearn.over_sampling import SMOTE
from classifier import *
from utils import *
import math
from sklearn.metrics import accuracy_score, matthews_corrcoef as mcc
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import balanced_accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score



class Node:
    def __init__(self, x_train, y_train, x_test, y_test, feature_names, ID, overlap_features=[], feature_types={}):
        self.id = ID
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = None
        self.rf = None
        self.trees = None
        self.feature_names = feature_names
        self.combined_rf = None
        self.overlap_features = overlap_features
        self.feature_types = feature_types
        self.received_imputation_models = {}
        
        self.other_features=None    
        self.other_features_stats = {}
        
        self.inference_stats = {'surrogates_used': 0, 'fallbacks_triggered': 0}
        
    def define_rf(self):
        rf = CustomRandomForestClassifier(feature_names=self.feature_names, n_estimators=100 ,random_state=42, n_jobs=1)
        self.rf = rf
    
    def train(self, oversampling):
        if(oversampling):
            x, y = self.smote(self.x_train, self.y_train)
            self.rf.fit(x, y)
        else:
            self.rf.fit(self.x_train, self.y_train)

    
    def smote(self, X, y):
        oversample = SMOTE()
        X, y = oversample.fit_resample(X, y)
        return X, y  
    

    def offset_features (self, other_node):  
        feature_offset = len(other_node.x_train.columns)

        for tree in self.rf.estimators_:
            offset_feature_indexes(tree.tree_, feature_offset)

        self.rf.update_feature_names_mapping(feature_offset)
    
    
    def calculate_mean (self):
        
        metadata = {}
        for column in self.x_train.columns:
            avg = np.mean(self.x_train[column])
            metadata[column] = avg
        
        return metadata



    def _best_imputer_for(self, feature_name):
        
        """
        Pick the highest-quality imputer for a given feature among received candidates.
        """
        
        candidates = self.received_imputation_models.get(feature_name, [])
        if not candidates:
            return None

        def key_fn(d):

            score = d.get("score", float('nan'))
            if np.isnan(score):
                return (-1e9, d.get("n_train", 0))  # treat NaN as very bad
            return (score, d.get("n_train", 0))

        best = max(candidates, key=key_fn)
        return best.get("model", None)


#---------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def manual_predict(self, sample_to_predict, class_labels, method, return_votes=False, **kwargs):
        """
        Calls the correct manual prediction method based on the 'method' string provided.
        """
        method_lower = method.lower()


        if method_lower == 'intersection':
            return self.manual_predict_intersection(sample_to_predict, class_labels, return_votes, **kwargs)

        elif method_lower == 'probability_weighted':
            return self.manual_predict_probability_weighted(sample_to_predict, class_labels, return_votes, **kwargs)

        elif method_lower == 'surrogate_split_agreement_pr_01':
            return self.manual_predict_surrogate_agreement_pr(sample_to_predict, class_labels, return_votes, **kwargs)
  
        elif method_lower == 'informed_marginal_cat_simple':
            return self.manual_predict_informed_marginal_cat_aware_simple(sample_to_predict, class_labels, return_votes, **kwargs)
        
        elif method_lower == 'probability_informed_cat_simple_fallback_fixed':
            return self.manual_predict_probability_informed_cat_aware_simple_fallback_fixed(sample_to_predict, class_labels, return_votes, **kwargs)        
   
        elif method_lower == 'model_imputation':
            return self.manual_predict_model_imputation(sample_to_predict, class_labels, return_votes, **kwargs)

        elif method_lower == 'treewise_filtering':
            return self.manual_predict_treewise_filtering(sample_to_predict, class_labels, return_votes, **kwargs) 
        
        elif method_lower == 'baseline_intersection':
            return self.manual_predict_baseline_intersection(sample_to_predict, class_labels, return_votes, **kwargs)         
             
        
        else:
            raise ValueError(f"Unsupported prediction method: '{method}'.")
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def score(self, method):
        """
        Calculates MCC, AUC, BACC, AP, AUPRC, F1, F1-macro, F1-weighted.
        """
        y_true = self.y_test
        if len(y_true) == 0:
            return {
                'mcc': 0.0, 'auc': 0.5, 'bacc': 0.5, 'ap': 0.0, 'auprc': 0.0,
                'f1': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0
            }

        class_labels = self.combined_rf.classes_
        y_pred_labels = []
        y_pred_probas_dist = []


        for i in range(len(self.x_test)):
            sample = self.x_test.iloc[[i]]

            result = self.manual_predict(
                sample,
                class_labels,
                method,
                return_votes=True
            )

            if isinstance(result, dict):
                label = result['label']
                proba_dist = result['proba_dist']
            elif isinstance(result, tuple):
                label = result[0]
                votes = result[1]
                if not votes:
                    proba_dist = [1.0 / len(class_labels)] * len(class_labels)
                else:
                    vote_counts = {l: votes.count(l) for l in class_labels}
                    proba_dist = [vote_counts.get(l, 0) / len(votes) for l in class_labels]
            else:
                label = result
                proba_dist = [1.0 if c == label else 0.0 for c in class_labels]

            y_pred_labels.append(label)
            y_pred_probas_dist.append(proba_dist)

        try:
            scores = compute_all_metrics(y_true, y_pred_labels, y_pred_probas_dist, class_labels)
        except Exception as e:
            print(f"Warning: Could not calculate a metric. Error: {e}")
            scores = {
                'mcc': 0.0, 'auc': 0.5, 'bacc': 0.5, 'ap': 0.0, 'auprc': 0.0,
                'f1': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0
            }

        return scores       
    
#------------------------------------------------------------------------------------------------------------------------------------------        

    def manual_predict_model_imputation(self, sample_df, class_labels, return_votes=False, **kwargs):
        """
        Model Imputation: Predicts a sample's class by using pre-trained models to impute missing feature values during tree traversal.
        """
        votes_idx = []
        sample_series = sample_df.iloc[0]

        for tree in self.combined_rf.estimators_:
            was_skipped = False
            node = 0
            while tree.tree_.children_left[node] != tree.tree_.children_right[node]:
                feature_idx = tree.tree_.feature[node]
                fname = self.combined_rf.get_feature_name_from_node(feature_idx)

                if fname in sample_series.index:
                    # Feature is available, proceed normally
                    threshold = tree.tree_.threshold[node]
                    test_value = sample_series[fname] <= threshold

                else:
                    # Feature is missing, try the best imputer (based on quality score)
                    imputer = self._best_imputer_for(fname)
                    if imputer is None:
                        was_skipped = True
                        break

                    imputer_input = sample_df[self.overlap_features]
                    imputed_value = imputer.predict(imputer_input)[0]
                    
                    try:
                        imputed_value = float(imputed_value)
                        
                    except Exception:
                        was_skipped = True
                        break

                    threshold = tree.tree_.threshold[node]
                    test_value = imputed_value <= threshold
                    
    
                node = tree.tree_.children_left[node] if test_value else tree.tree_.children_right[node]

            if was_skipped:
                continue

            vote = np.argmax(tree.tree_.value[node][0])
            votes_idx.append(vote)

        # Get the majority numeric vote index
        majority_idx = max(set(votes_idx), key=votes_idx.count)

        # The final single-label prediction
        final_prediction = class_labels[majority_idx]

        # The full list of votes (as actual class labels) for AUC calculation
        predictions = [class_labels[idx] for idx in votes_idx]

        if return_votes:
            return final_prediction, predictions
        else:
            return final_prediction
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------

    def manual_predict_surrogate_agreement_pr(self, sample, class_labels, return_votes=False, **kwargs):
        """
        Surrogate Splits.
        """        
        import numpy as np
        import pandas as pd

        votes_idx = []
        depth_threshold = kwargs.get('depth_threshold', 2)

        for tree in self.combined_rf.estimators_:
            node = 0
            depth = 0
            was_skipped = False

            fallback_counted = False
            
            while tree.tree_.children_left[node] != tree.tree_.children_right[node]:
                fidx = tree.tree_.feature[node]
                fname = self.combined_rf.get_feature_name_from_node(fidx)

                if fname in sample.columns:

                    thresh = tree.tree_.threshold[node]
                    go_left = sample[fname].iloc[0] <= thresh
                    node = tree.tree_.children_left[node] if go_left else tree.tree_.children_right[node]
                    depth += 1
                    continue

                # Feature is missing: try SURROGATE first
                rule = getattr(tree, "surrogates", {}).get(node, None)

                if rule is not None:
                    sfeat = rule["feature"]
                    if sfeat in sample.columns:
                        val = sample[sfeat].iloc[0]
                        if pd.isna(val):
                            pass
                        else:

                            self.inference_stats['surrogates_used'] += 1
                            go_right_std = (val > rule["threshold"])
                            go_right = (not rule["invert"] and go_right_std) or (rule["invert"] and (not go_right_std))
                            node = tree.tree_.children_right[node] if go_right else tree.tree_.children_left[node]
                            depth += 1
                            continue


                # FPV/PR fallback (no surrogate)
                if not fallback_counted:
                    self.inference_stats['fallbacks_triggered'] += 1
                    fallback_counted = True
                    
                if depth < depth_threshold:
                    was_skipped = True
                    break

                left  = tree.tree_.children_left[node]
                right = tree.tree_.children_right[node]
                total = int(tree.tree_.n_node_samples[node])
                if total <= 0:
                    was_skipped = True
                    break

                left_samples = int(tree.tree_.n_node_samples[left])
                p_left = left_samples / total
                node = np.random.choice([left, right], p=[p_left, 1.0 - p_left])
                depth += 1

            if not was_skipped:
                vote = np.argmax(tree.tree_.value[node][0])
                votes_idx.append(vote)

        if not votes_idx:
            final_prediction = self.y_train.value_counts().idxmax()
            predictions = []
        else:
            majority_idx = max(set(votes_idx), key=votes_idx.count)
            final_prediction = class_labels[majority_idx]
            predictions = [class_labels[idx] for idx in votes_idx]

        return (final_prediction, predictions) if return_votes else final_prediction    

#---------------------------------------------------------------------------------------------------------------------------------------------------------    
        
    def manual_predict_probability_informed_cat_aware_simple_fallback_fixed(self, sample_df, class_labels, return_votes=False, **kwargs):
        """
        Informed Probabilistic Routing: Uses a model-imputed feature value to inform the probabilistic choice.
        """
        k = kwargs.get('k', 1.0)

        depth_threshold = kwargs.get('depth_threshold', 2)
        
        def _sigmoid(x):
            return 1 / (1 + np.exp(-x + 1e-8))

        votes_idx = [] 
        sample_series = sample_df.iloc[0]
        trees = self.combined_rf.estimators_

        for tree in trees:
            current_node = 0
            depth = 0
            skip_tree = False
            
            while tree.tree_.children_left[current_node] != tree.tree_.children_right[current_node]:
                feature_idx = tree.tree_.feature[current_node]
                fname = self.combined_rf.get_feature_name_from_node(feature_idx)

                    
                if fname in sample_series.index and sample_series.get(fname) is not None:

                    threshold = tree.tree_.threshold[current_node]
                    test_value = sample_series[fname] <= threshold
                    current_node = tree.tree_.children_left[current_node] if test_value else tree.tree_.children_right[current_node]
                    depth += 1                    
                    
                    

                else: # Feature is missing

                    imputer = self._best_imputer_for(fname)
                    feature_type = self.feature_types.get(fname)

                    left_child = tree.tree_.children_left[current_node]
                    right_child = tree.tree_.children_right[current_node]

                    informed_guess_possible = True

                    try:
                        if imputer is not None and feature_type == 'numeric':
                            val = imputer.predict(sample_df[self.overlap_features])[0]
                            imputed_value = float(val)

                            if fname in self.other_features_stats:
                                threshold = tree.tree_.threshold[current_node]
                                std_dev = self.other_features_stats[fname].get('std', 1.0)
                                norm_distance = (imputed_value - threshold) / max(std_dev, 1e-6)
                                p_right = _sigmoid(k * norm_distance)
                                current_node = np.random.choice([left_child, right_child], p=[1.0 - p_right, p_right])
                                depth += 1
                            else:
                                informed_guess_possible = False

                        elif imputer is not None and feature_type == 'categorical':

                            proba_dist = imputer.predict_proba(sample_df[self.overlap_features])[0]
                            threshold = tree.tree_.threshold[current_node]
                            imputer_classes = imputer.steps[-1][1].classes_
                            p_left = sum(prob for i, prob in enumerate(proba_dist) if imputer_classes[i] <= threshold)
                            current_node = np.random.choice([left_child, right_child], p=[p_left, 1.0 - p_left])
                            depth += 1

                        else:
                            informed_guess_possible = False

                    except Exception:
                        informed_guess_possible = False

                        
                    if not informed_guess_possible:
                        # FALLBACK: FPV for shallow nodes, PR for deeper nodes
                        if depth < depth_threshold:
                            skip_tree = True
                            break  # stop traversing this tree, it will not vote

                        # depth >= depth_threshold
                        total_samples = tree.tree_.n_node_samples[current_node] + 1e-8
                        left_count = tree.tree_.n_node_samples[left_child]
                        p_left = left_count / total_samples
                        current_node = np.random.choice([left_child, right_child], p=[p_left, 1.0 - p_left])
                        depth += 1                        
             
         
            if not skip_tree:
                vote = tree.tree_.value[current_node][0].argmax()
                votes_idx.append(vote)            

        majority_idx = max(set(votes_idx), key=votes_idx.count)

        final_prediction = class_labels[majority_idx]

        predictions = [class_labels[idx] for idx in votes_idx]

        if return_votes:
            return final_prediction, predictions
        else:
            return final_prediction         
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def manual_predict_probability_weighted(self, sample, class_labels, return_votes=False, **kwargs):
        """
        Probabilistic Routing (with depth threshold).
        """
        
        depth_threshold = kwargs.get('depth_threshold', 2)

        votes_idx = [] 
        weights = []
        trees = self.combined_rf.estimators_

        for tree in trees:
            current_node, missing_depth, depth = 0, None, 0

            while tree.tree_.children_left[current_node] != tree.tree_.children_right[current_node]:
                feature_name = self.combined_rf.get_feature_name_from_node(tree.tree_.feature[current_node])

                if feature_name in sample.columns:
                    threshold = tree.tree_.threshold[current_node]
                    current_node = tree.tree_.children_left[current_node] if sample[feature_name].iloc[0] <= threshold else tree.tree_.children_right[current_node]
                else:
                    if missing_depth is None: missing_depth = depth
                    total_samples = tree.tree_.n_node_samples[current_node]
                    left_samples = tree.tree_.n_node_samples[tree.tree_.children_left[current_node]]
                    p_left = left_samples / total_samples if total_samples > 0 else 0.5
                    current_node = np.random.choice(
                        [tree.tree_.children_left[current_node], tree.tree_.children_right[current_node]],
                        p=[p_left, 1.0 - p_left]
                    )
                depth += 1

            vote = tree.tree_.value[current_node][0].argmax()
            votes_idx.append(vote)

            # Determine the weight for this tree's vote (1 for keep, 0 for drop)
            weights.append(1.0 if (missing_depth is None or missing_depth >= depth_threshold) else 0.0)

            
        vote_dict = {}
        for idx, weight in zip(votes_idx, weights):
            if weight > 0:
                vote_dict[idx] = vote_dict.get(idx, 0) + weight


        majority_idx = max(vote_dict, key=vote_dict.get)

        final_prediction = class_labels[majority_idx]

        predictions = [class_labels[idx] for idx, weight in zip(votes_idx, weights) if weight > 0]

        if return_votes:
            return final_prediction, predictions
        else:
            return final_prediction
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------

    def manual_predict_informed_marginal_cat_aware_simple(self, sample_df, class_labels, return_votes=False, **kwargs):
        """
        Marginal Prediction.
        """
        
        k = float(kwargs.get("k", 1.0))
        eps = 1e-8

        def _sigmoid(x: float) -> float:
            return 1.0 / (1.0 + np.exp(-x + eps))

        def _leaf_dist(tree, nid):
            vals = tree.tree_.value[nid][0]
            s = vals.sum()
            if s <= 0:
                return np.full(len(class_labels), 1.0 / len(class_labels), dtype=float)
            return vals / s

        def _expected_informed(tree, nid):
            
            # Base case: at a leaf, return its probability distribution
            t = tree.tree_
            left, right = t.children_left[nid], t.children_right[nid]
            if left == right:
                return _leaf_dist(tree, nid)

            feat_idx = t.feature[nid]
            fname    = self.combined_rf.get_feature_name_from_node(feat_idx)
            thr      = t.threshold[nid]

            # Feature is available, traverse normally
            if fname in sample_df.columns:
                val = sample_df[fname].iloc[0]
                next_id = left if val <= thr else right
                return _expected_informed(tree, next_id)


            p_left = None
            try:
                imputer = self._best_imputer_for(fname)
                feature_type = self.feature_types.get(fname)

                if imputer and feature_type == 'numeric':
                    imp_val = float(imputer.predict(sample_df[self.overlap_features])[0])
                    if fname in self.other_features_stats:
                        std = float(self.other_features_stats[fname].get("std", 1.0))
                        distance = k * (imp_val - thr) / max(std, 1e-6)
                        p_right = _sigmoid(distance)
                        p_left = 1.0 - p_right

                elif imputer and feature_type == 'categorical':
                    proba_dist = imputer.predict_proba(sample_df[self.overlap_features])[0]
                    imputer_classes = imputer.steps[-1][1].classes_
                    p_left = sum(prob for i, prob in enumerate(proba_dist) if imputer_classes[i] <= thr)

            except Exception:
                p_left = None

            if p_left is None:
                # Fallback: empirical split proportions
                total = max(float(t.n_node_samples[nid]), eps)
                p_left = float(t.n_node_samples[left]) / total

            # Mix child distributions
            dist_left  = _expected_informed(tree, left)
            dist_right = _expected_informed(tree, right)
            return p_left * dist_left + (1.0 - p_left) * dist_right


        trees = self.combined_rf.estimators_
        all_dists = []
        for tree in trees:
            all_dists.append(_expected_informed(tree, 0))

        if not all_dists:
            fallback = self.y_train.value_counts().idxmax()
            return {"label": fallback, "proba_dist": [1.0 / len(class_labels)] * len(class_labels)}

        final_prob_dist = np.mean(all_dists, axis=0)
        final_idx = int(np.argmax(final_prob_dist))
        final_label = class_labels[final_idx]

        return {"label": final_label, "proba_dist": final_prob_dist}    

#---------------------------------------------------------------------------------------------------------------------------------------------------------    

    def manual_predict_intersection(self, sample, class_labels, return_votes=False, **kwargs):
        """
        Feasible Path Voting: Predicts a sample's class by dropping any tree that requires a missing feature during traversal.
        """
        votes_idx = [] 
        trees = self.combined_rf.estimators_

        for tree in trees:
            current_node = 0
            skip_tree = False

            while tree.tree_.children_left[current_node] != tree.tree_.children_right[current_node]:
                feature_name = self.combined_rf.get_feature_name_from_node(tree.tree_.feature[current_node])

                if feature_name in sample.columns and sample[feature_name].iloc[0] is not None:

                    threshold = tree.tree_.threshold[current_node]
                    test_value = sample[feature_name].iloc[0] <= threshold
                    current_node = tree.tree_.children_left[current_node] if test_value else tree.tree_.children_right[current_node]
                else:
                    skip_tree = True
                    break

            if not skip_tree:
                vote = tree.tree_.value[current_node][0].argmax()
                votes_idx.append(vote)

        majority_idx = max(set(votes_idx), key=votes_idx.count)
        final_prediction = class_labels[majority_idx]
        predictions = [class_labels[idx] for idx in votes_idx]

        if return_votes:
            return final_prediction, predictions
        else:
            return final_prediction

#---------------------------------------------------------------------------------------------------------------------------------------------------------

    def manual_predict_treewise_filtering(self, sample, class_labels, return_votes=False, **kwargs):
        """
        Park et al: Tree-wise overlap filtering:
        Keep a tree only if all split features it ever uses
        are available in this sample; otherwise skip the whole tree.
        """
        votes_idx = []
        trees = self.combined_rf.estimators_
        available = set(sample.columns)

        for tree in trees:

            used_feature_names = set()
            feats = tree.tree_.feature
            for fid in feats:
                if fid == -2:
                    continue
                fname = self.combined_rf.get_feature_name_from_node(int(fid))
                used_feature_names.add(fname)

            # Skip entire tree if it uses any unavailable feature
            if not used_feature_names.issubset(available):
                continue

            # Otherwise, traverse normally (all needed features are present)
            current_node = 0
            while tree.tree_.children_left[current_node] != tree.tree_.children_right[current_node]:
                feature_name = self.combined_rf.get_feature_name_from_node(tree.tree_.feature[current_node])
                threshold = tree.tree_.threshold[current_node]
                test_value = sample[feature_name].iloc[0] <= threshold
                current_node = (
                    tree.tree_.children_left[current_node]
                    if test_value
                    else tree.tree_.children_right[current_node]
                )

            vote = tree.tree_.value[current_node][0].argmax()
            votes_idx.append(vote)

        majority_idx = max(set(votes_idx), key=votes_idx.count)
        final_prediction = class_labels[majority_idx]
        predictions = [class_labels[idx] for idx in votes_idx]

        if return_votes:
            return final_prediction, predictions
        else:
            return final_prediction

#---------------------------------------------------------------------------------------------------------------------------------------------------------        

    def manual_predict_baseline_intersection(self, sample, class_labels, return_votes=False, **kwargs):
        """
        Baseline (global intersection)
        """
        votes_idx = []
        trees = self.combined_rf.estimators_

        for tree in trees:
            current_node = 0

            while tree.tree_.children_left[current_node] != tree.tree_.children_right[current_node]:
                feature_name = self.combined_rf.get_feature_name_from_node(tree.tree_.feature[current_node])

                threshold = tree.tree_.threshold[current_node]
                test_value = sample[feature_name].iloc[0] <= threshold
                current_node = (
                    tree.tree_.children_left[current_node] if test_value
                    else tree.tree_.children_right[current_node]
                )

            vote = tree.tree_.value[current_node][0].argmax()
            votes_idx.append(vote)

        majority_idx = max(set(votes_idx), key=votes_idx.count)
        final_prediction = class_labels[majority_idx]
        predictions = [class_labels[idx] for idx in votes_idx]

        if return_votes:
            return final_prediction, predictions
        else:
            return final_prediction
        

#---------------------------------------------------------------------------------------------------------------------------------------------------------        
