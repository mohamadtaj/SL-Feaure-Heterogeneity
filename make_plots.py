import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Orange
from scipy.stats import rankdata, ttest_rel
from matplotlib.lines import Line2D

plt.rcParams.update({'figure.max_open_warning': 0})


# CONFIGURATION

DATASETS = [
    'cdc_3class_balanced', 'cdc_binary5050_stratified',
    's_500_num_not_corr', 's_500_cat_not_corr', 's_500_mix_not_corr',
    'thyroid', 'glioma', 'gallstone', 'diabetes'
]

METRICS = ["mcc", "auprc", "auc"]

PEERS = [2, 3, 4, 6]

OVERLAPS = [20, 30, 40, 50, 60, 70, 80, 90] 

METHODS = [
    "intersection",
    "baseline_intersection",
    "probability_weighted",
    "informed_marginal_cat_simple",
    "probability_informed_cat_simple_fallback_fixed",
    "surrogate_split_agreement_pr_01",
    "treewise_filtering",
]

METHOD_ALIAS = {
    "intersection": "Feasible Path Voting",
    "probability_weighted": "Probabilistic Routing",
    "informed_marginal_cat_simple": "Marginal Prediction",
    "probability_informed_cat_simple_fallback_fixed": "Informed Prob. Routing",
    "surrogate_split_agreement_pr_01": "Surrogate Splits",
    "treewise_filtering": "Park et al.",
    "baseline_intersection": "Intersection",
    "local": "Local" 
}

LOCAL_REFERENCE_METHOD = "intersection"

OUTPUT_DIR = "plots"
CD_SUBDIR = "cd_plots"


# VISUAL SETTINGS

# --- Font Sizes ---

# Lineplots
LINE_TITLE_SIZE = 14
LINE_LABEL_SIZE = 12
LINE_TICK_SIZE  = 10

# Heatmaps
HEAT_TITLE_SIZE = 20
HEAT_LABEL_SIZE = 18
HEAT_TICK_SIZE  = 16
HEAT_CELL_SIZE  = 14
HEAT_COLORBAR_SIZE = 16

# CD plots
CD_TITLE_SIZE   = 14

# --- Line Plots ---
LINE_WIDTH = 1.5            
FIG_SIZE_LINE = (10, 8)     
LINE_WSPACE = 0.1           
LINE_HSPACE = 0.2           
CI_ALPHA = 0.15             

#--------------------------------------------

# --- Heatmaps ---
FIG_SIZE_HEATMAP = (18, 12) 
HEAT_WSPACE = 0.10          
HEAT_HSPACE = 0.20          
HEATMAP_CMAP = "RdBu_r"     

# --- Colors ---
COLOR_METHOD = "#4c72b0"    
COLOR_BASELINE = "#c44e52"  
COLOR_LOCAL = "#7f7f7f"     

# --- THEME SETUP ---
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)



# DATA LOADING

def load_detailed_data():
    records = []
    print("Loading data...")

    for metric in METRICS:
        metric_dir = f"results_{metric}"
        if not os.path.exists(metric_dir):
            continue

        for dataset in DATASETS:
            dataset_dir = os.path.join(metric_dir, dataset)
            if not os.path.exists(dataset_dir):
                continue

            for p in PEERS:
                for ov in OVERLAPS:
                    scenario_dir = os.path.join(dataset_dir, f"p{p}_{ov}")
                    if not os.path.exists(scenario_dir):
                        continue
                    
                    jaccard_idx = ov / 100.0

                    def get_scenario_scores(method_name, is_local=False):
                        peer_arrays = []
                        for node_id in range(p):
                            suffix = "local" if is_local else "fl"
                            fname_1 = f"{method_name}_{suffix}_{node_id + 1}.npy"
                            fpath_1 = os.path.join(scenario_dir, fname_1)
                            fname_0 = f"{method_name}_{suffix}_{node_id}.npy"
                            fpath_0 = os.path.join(scenario_dir, fname_0)

                            if os.path.exists(fpath_1):
                                peer_arrays.append(np.load(fpath_1))
                            elif os.path.exists(fpath_0):
                                peer_arrays.append(np.load(fpath_0))
                        
                        if not peer_arrays: return None
                        try:
                            return np.mean(np.vstack(peer_arrays), axis=0) 
                        except ValueError: return None

                    for method in METHODS:
                        scores = get_scenario_scores(method, is_local=False)
                        if scores is not None:

                            records.append({
                                "Metric": metric, "Dataset": dataset,
                                "Peers": p, "Overlap": jaccard_idx,
                                "Method": method, "Type": "FL", 
                                "Scores": scores,
                                "Mean_Score": np.mean(scores)
                            })

                    local_scores = get_scenario_scores(LOCAL_REFERENCE_METHOD, is_local=True)
                    
                    if local_scores is not None:
                        records.append({
                            "Metric": metric, "Dataset": dataset,
                            "Peers": p, "Overlap": jaccard_idx,
                            "Method": "local", "Type": "Local", 
                            "Scores": local_scores,
                            "Mean_Score": np.mean(local_scores)
                        })

    return pd.DataFrame(records)


# PLOTTING FUNCTIONS

def plot_lineplots_with_ci(df, dataset, metric):
    subset = df[(df['Dataset'] == dataset) & (df['Metric'] == metric)]
    if subset.empty: return

    target_methods = ['informed_marginal_cat_simple', 'baseline_intersection', 'local']
    
    plot_data_list = []
    for _, row in subset.iterrows():
        if row['Method'] in target_methods:
            for s in row['Scores']:
                plot_data_list.append({
                    'Peers': row['Peers'], 'Overlap': row['Overlap'], 
                    'Method': row['Method'], 'Score': s
                })
    
    plot_data = pd.DataFrame(plot_data_list)
    if plot_data.empty: return
    
    plot_data['Method_Label'] = plot_data['Method'].map(METHOD_ALIAS)

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_LINE, sharex=True, sharey=True)
    axes = axes.flatten()
    
    palette = {
        METHOD_ALIAS['informed_marginal_cat_simple']: COLOR_METHOD,
        METHOD_ALIAS['baseline_intersection']: COLOR_BASELINE,
        METHOD_ALIAS['local']: COLOR_LOCAL
    }
    dashes = {
        METHOD_ALIAS['informed_marginal_cat_simple']: "",
        METHOD_ALIAS['baseline_intersection']: "",
        METHOD_ALIAS['local']: (2, 2)
    }

    for i, p in enumerate(PEERS):
        if i >= len(axes): break
        ax = axes[i]
        peer_data = plot_data[plot_data['Peers'] == p]
        
        if peer_data.empty: continue

        sns.lineplot(
            data=peer_data, x="Overlap", y="Score", 
            hue="Method_Label", style="Method_Label",
            dashes=dashes, palette=palette, 
            ax=ax, linewidth=LINE_WIDTH, errorbar=('ci', 95),
            err_kws={'alpha': CI_ALPHA},
            legend=False
        )
        
        ax.set_title(f"Peers: {p}", fontsize=LINE_TITLE_SIZE)

        all_ticks = [ov / 100.0 for ov in OVERLAPS]
        ax.set_xticks(all_ticks)
    
        if i >= 2: ax.set_xlabel("Jaccard Index", fontsize=LINE_LABEL_SIZE)
        else: ax.set_xlabel("")
        if i % 2 == 0: ax.set_ylabel(metric.upper(), fontsize=LINE_LABEL_SIZE)
        else: ax.set_ylabel("")  
            
        ax.tick_params(axis="both", labelsize=LINE_TICK_SIZE)
            
    legend_elements = [
        Line2D([0], [0], color=COLOR_METHOD, lw=2, label=METHOD_ALIAS['informed_marginal_cat_simple']),
        Line2D([0], [0], color=COLOR_BASELINE, lw=2, label=METHOD_ALIAS['baseline_intersection']),
        Line2D([0], [0], color=COLOR_LOCAL, lw=2, linestyle='--', label=METHOD_ALIAS['local'])
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), frameon=False)
    # plt.suptitle(f"{dataset} - {metric.upper()}", y=0.96)
    plt.subplots_adjust(bottom=0.15, wspace=LINE_WSPACE, hspace=LINE_HSPACE)
    
    save_dir = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"lineplot_{metric}_{dataset}.svg"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_heatmaps_stats(df, dataset, metric, baseline_method='baseline_intersection'):
    subset = df[(df['Dataset'] == dataset) & (df['Metric'] == metric)]
    if subset.empty: return

    pivot_means = subset.pivot_table(
        index=['Peers', 'Overlap'], columns='Method', values='Mean_Score'
    )
    
    if baseline_method not in pivot_means.columns: return

    methods_to_plot = [m for m in METHODS if m != baseline_method]
    
    n_plots = len(methods_to_plot)
    cols = 3
    rows = (n_plots // cols) + (1 if n_plots % cols > 0 else 0)
    
    fig, axes = plt.subplots(rows, cols, figsize=FIG_SIZE_HEATMAP, sharex=True, sharey=True)
    axes = axes.flatten()
    
    all_gains = []
    heatmap_data_list = []
    
    for method in methods_to_plot:
        if method not in pivot_means.columns: 
            heatmap_data_list.append(None)
            continue
        
        base_vals = pivot_means[baseline_method]
        meth_vals = pivot_means[method]
        gain_matrix = ((meth_vals - base_vals) / base_vals * 100).unstack(level=0)
        gain_matrix = gain_matrix.sort_index(ascending=False)
        all_gains.append(gain_matrix.values)
        heatmap_data_list.append(gain_matrix)

    if not all_gains: return
    flat_gains = np.concatenate([g.flatten() for g in all_gains if g is not None])
    abs_max = max(abs(flat_gains.min()), abs(flat_gains.max()))
    vmin, vmax = -abs_max, abs_max

    for i, method in enumerate(methods_to_plot):
        ax = axes[i]
        gain_matrix = heatmap_data_list[i]
        
        if gain_matrix is None: 
            ax.axis('off')
            continue

        annot_matrix = gain_matrix.copy().astype(object)
        
        for p_col in gain_matrix.columns:
            for ov_idx in gain_matrix.index:
                try:

                    row_meth = subset[(subset['Peers']==p_col) & (subset['Overlap']==ov_idx) & (subset['Method']==method)]
                    row_base = subset[(subset['Peers']==p_col) & (subset['Overlap']==ov_idx) & (subset['Method']==baseline_method)]
                    
                    if not row_meth.empty and not row_base.empty:
                        a = row_meth.iloc[0]['Scores']
                        b = row_base.iloc[0]['Scores']
                        val = gain_matrix.loc[ov_idx, p_col]
                        
                        if len(a) == len(b) and len(a) > 1:
                            stat, pval = ttest_rel(a, b)
                            stars = ""
                            if pval < 0.001: stars = "***"
                            elif pval < 0.01: stars = "**"
                            elif pval < 0.05: stars = "*"
                            annot_matrix.loc[ov_idx, p_col] = f"{val:.1f}%\n{stars}"
                        else:
                            annot_matrix.loc[ov_idx, p_col] = f"{val:.1f}%"
                    else:
                        annot_matrix.loc[ov_idx, p_col] = "N/A"
                except Exception:
                    annot_matrix.loc[ov_idx, p_col] = "N/A"

        sns.heatmap(
            gain_matrix, annot=annot_matrix, fmt="", 
            cmap=HEATMAP_CMAP, center=0, vmin=vmin, vmax=vmax,
            ax=ax, cbar=False,
            annot_kws={"va": "center", "ha": "center", "fontsize": HEAT_CELL_SIZE}
        )
        
        ax.set_title(f"{METHOD_ALIAS.get(method, method)}", fontsize=HEAT_TITLE_SIZE)
        
        if i >= (rows - 1) * cols: ax.set_xlabel("Peers", fontsize=HEAT_LABEL_SIZE)
        else: ax.set_xlabel("") 
        if i % cols == 0: ax.set_ylabel("Jaccard Index", fontsize=HEAT_LABEL_SIZE)
        else: ax.set_ylabel("")
            
        ax.tick_params(axis="both", labelsize=HEAT_TICK_SIZE)

    for j in range(i+1, len(axes)): axes[j].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=HEATMAP_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=f"Relative Gain over {METHOD_ALIAS.get(baseline_method)}")

    # plt.suptitle(f"{dataset} - {metric.upper()}", y=0.96)
    plt.subplots_adjust(right=0.9, wspace=HEAT_WSPACE, hspace=HEAT_HSPACE)
    
    save_dir = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"heatmap_{metric}_{dataset}.svg"), dpi=300, bbox_inches='tight')
    plt.close()


# STATISTICAL RANKING (CD DIAGRAMS with T-TEST)

def _rank_by_significance(block_df):

    methods = block_df['Method'].values
    score_arrays = block_df['Scores'].values
    n = len(methods)
    
    ranks = []
    
    for i in range(n):

        scores_i = score_arrays[i]
        mean_i = np.mean(scores_i)
        
        better_count = 0
        tie_count = 0
        
        for j in range(n):
            if i == j: continue
            
            scores_j = score_arrays[j]
            mean_j = np.mean(scores_j)
            
            stat, pval = ttest_rel(scores_j, scores_i)
            
            if pval > 0.05:
                tie_count += 1
            else:

                if mean_j > mean_i:
                    better_count += 1
        
        rank_val = 1 + better_count + (tie_count / 2.0)
        ranks.append(rank_val)
        
    return pd.Series(ranks, index=methods)

def compute_and_plot_cd(df, scope_name, filename_prefix):
    """
    Generates CD diagrams using T-Test based ranking logic.
    """
    df_methods = df[df['Method'].isin(METHODS)].copy()
    
    
    if scope_name == "dataset":
        group_keys = ['Dataset', 'Peers', 'Overlap']

        for ds in df_methods['Dataset'].unique():
            ds_data = df_methods[df_methods['Dataset'] == ds]
            
            ranks_list = []
            for (p, ov), block in ds_data.groupby(['Peers', 'Overlap']):
                if len(block) < 2: continue
                r = _rank_by_significance(block)

                r_dict = r.to_dict()
                ranks_list.append(r_dict)
            
            ranks_df = pd.DataFrame(ranks_list)
            _generate_cd_plot_from_ranks(ranks_df, f"{ds}_cd_{filename_prefix}")

    elif scope_name == "overlap":

        for ov in df_methods['Overlap'].unique():
            ov_data = df_methods[df_methods['Overlap'] == ov]
            
            ranks_list = []

            for (ds, p), block in ov_data.groupby(['Dataset', 'Peers']):
                if len(block) < 2: continue
                r = _rank_by_significance(block)
                ranks_list.append(r.to_dict())
                
            ranks_df = pd.DataFrame(ranks_list)
            ov_int = int(ov * 100)
            _generate_cd_plot_from_ranks(ranks_df, f"per_overlap_cd_{ov_int}_{filename_prefix}")

    elif scope_name == "global":

        ranks_list = []

        for (ds, p, ov), block in df_methods.groupby(['Dataset', 'Peers', 'Overlap']):
            if len(block) < 2: continue
            r = _rank_by_significance(block)
            ranks_list.append(r.to_dict())
            
        ranks_df = pd.DataFrame(ranks_list)
        _generate_cd_plot_from_ranks(ranks_df, f"global_cd_{filename_prefix}")

def _generate_cd_plot_from_ranks(ranks_df, title_suffix):

    if ranks_df.empty: return
    
    cols = [c for c in ranks_df.columns if c in METHODS]
    ranks_df = ranks_df[cols].dropna()
    
    if ranks_df.empty: return

    avg_ranks = ranks_df.mean()
    names = [METHOD_ALIAS.get(m, m) for m in avg_ranks.index]
    avranks = avg_ranks.values
    number_of_datasets = len(ranks_df)
    
    cd = Orange.evaluation.compute_CD(avranks, number_of_datasets)
    
    try:
        plt.figure(figsize=(12, 8))
        Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
        
        save_path = os.path.join(OUTPUT_DIR, CD_SUBDIR, f"{title_suffix}.svg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # plt.title(f"CD Diagram (N={number_of_datasets})", fontsize=CD_TITLE_SIZE)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting CD {title_suffix}: {e}")

        
#-----------------------------------------------------------------------------------------        
        
if __name__ == "__main__":
    df_all = load_detailed_data()
    if df_all.empty:
        print("No data found.")
        exit()
    
    df_all.to_csv("df_all_results.csv")

    for metric in METRICS:
        print(f"Processing metric: {metric}")
        df_metric = df_all[df_all['Metric'] == metric].copy()
        if df_metric.empty: continue

        for dataset in DATASETS:
            plot_lineplots_with_ci(df_metric, dataset, metric)
            plot_heatmaps_stats(df_metric, dataset, metric, baseline_method='baseline_intersection')

        
        compute_and_plot_cd(df_metric, "dataset", metric)
        compute_and_plot_cd(df_metric, "overlap", metric)
        compute_and_plot_cd(df_metric, "global",  metric)


    print("Done! Plots generated in 'plots' folder.")
