import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

METHODS = {
    'DALES-2025-12-22_16-10-48_none_ce': 'uni',
    'DALES-2025-12-31_05-31-50_none_ldam': 'LDAM',

    # 'S3DIS-2025-12-19_14-20-20_none': 'uni',
    # 'S3DIS-2025-12-25_10-03-45_none_ldam': 'LDAM',

    # 'STPLS3D-2026-01-02_14-30-31_none_ce': 'uni',
    # 'STPLS3D-2026-01-03_16-36-49_comf_ce': 'comf',
}

DATASET = 'DALES'
DESIRED_PERCENTAGES=(0.1, 1.0, 10.0, 20.0)
K = 20
SEED = 42

# (bottom, top) window for the boxplot y-axis; None autoscales to the data.
# The largest delta observed here is ~23 on DALES, ~221 on S3DIS and ~35 on STPLS3D.
# On the wider panels the small-alpha boxes collapse under a linear autoscale.
YLIM = None


def plot_boxplot_loss_comparison(dataset=DATASET, methods=METHODS, rhos=DESIRED_PERCENTAGES, K=K, seeds=[SEED], ylim=YLIM):
    """
    Plot boxplots comparing loss difference flatness metrics across different models and percent values.
    Creates separate plots for train and validation losses.
    
    Args:
        rhos: List of desired_percent values to plot
        K: Number of samples
        seeds: List of seeds to aggregate (for robustness)
    """
    
    base_dir = f'results/{dataset}'
    font_size = 14
    
    plt.rcParams.update({'font.size': font_size})
    sns.set_style("whitegrid")
    
    train_plot_data = []
    
    for method_dir, method_name in methods.items():
        results_dir = os.path.join(base_dir, method_dir)
        
        if not os.path.exists(results_dir):
            print(f"Warning: Directory {results_dir} not found, skipping {method_name}")
            continue
        
        for desired_percent in rhos:
            for seed in seeds:
                
                samples_file = f"{results_dir}/flatness_samples_percent_{desired_percent}_K_{K}_seed_{seed}.csv"
                summary_file = f"{results_dir}/flatness_summary_percent_{desired_percent}_K_{K}_seed_{seed}.csv"
                
                if os.path.exists(samples_file) and os.path.exists(summary_file):
                    
                    df_samples = pd.read_csv(samples_file)
                    
                    # Get original losses from summary file
                    df_summary = pd.read_csv(summary_file)
                    original_train_loss = df_summary[df_summary['metric'] == 'original_train_loss']['value'].values[0]
                    
                    # Calculate delta losses for each sample
                    train_delta_losses = df_samples['perturbed_train_loss'].values - float(original_train_loss)
                    
                    # Add each sample to plot_data
                    for train_delta_loss in train_delta_losses:
                        train_plot_data.append({
                            'percent': desired_percent,
                            'method': method_name,
                            'delta_loss': train_delta_loss
                        })

                else:
                    if not os.path.exists(samples_file):
                        print(f"Warning: Samples file not found: {samples_file}")
                    if not os.path.exists(summary_file):
                        print(f"Warning: Summary file not found: {summary_file}")
    
    if not train_plot_data:
        print("No data found for plotting!")
        return
    
    if train_plot_data:
        df_train = pd.DataFrame(train_plot_data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(
            data=df_train,
            x='percent',
            y='delta_loss',
            hue='method',
            ax=ax,
            palette='tab20',
            showfliers=True,
            width=0.7
        )
        
        ax.set_xlabel(r'Perturbation Percentage $\alpha$ (%)', fontsize=font_size)
        ax.set_ylabel(r'Train Loss Difference ($\Delta L$)', fontsize=font_size)
        ax.set_title(f'Training loss flatness ({dataset})', fontsize=font_size+2)
        
        unique_percents = sorted(df_train['percent'].unique())
        ax.set_xticks(range(len(unique_percents)))
        ax.set_xticklabels([f'{p}%' for p in unique_percents])
        
        ax.minorticks_on()
        ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.xaxis.grid(False)
        if ylim is not None:
            ax.set_ylim(*ylim)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Method')
        
        plt.tight_layout()
        
        plot_name = f"{dataset}_boxplot_train_loss_comparison_K_{K}_seeds_{'_'.join(map(str, seeds))}"
        plt.savefig(plot_name + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(plot_name + ".pdf", bbox_inches='tight')
        print(f"Train loss boxplot comparison saved to: {plot_name}")
        plt.show()

if __name__ == '__main__':

    plot_boxplot_loss_comparison()