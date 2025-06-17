import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import to_rgb
import colorsys

os.makedirs('./plots', exist_ok=True)

uniref90_df = pd.read_csv('8M_35M_our_model_uniref90.csv')  # CSV containing Uniref90 models
uniref50_df = pd.read_csv('8M_35M_150M_650M.csv')  # CSV containing baseline and our models

uniref50_df_filtered = uniref50_df[uniref50_df['Model'].isin(['8M_our_model', '35M_our_model'])].copy()

def prepare_data(df):
    validation = df[df['Set'] == 'validation']
    testing = df[df['Set'].str.startswith('test')].copy()
    testing['Bin'] = testing['Set'].apply(lambda x: '_'.join(x.split('_')[2:]))
    numeric_cols = ['FCD', 'Cosine Similarity', 'Mahalanobis', 'Unfamiliarity', 'Perplexity']
    mean_testing = testing.groupby(['Model', 'Bin'])[numeric_cols].mean().reset_index()
    plot_data = validation[['Model'] + numeric_cols].copy()
    for bin_group in mean_testing['Bin'].unique():
        bin_df = mean_testing[mean_testing['Bin'] == bin_group].drop(columns=['Bin']).set_index('Model')
        bin_df.columns = [f"{col}_{bin_group}" for col in bin_df.columns]
        plot_data = plot_data.merge(bin_df, left_on='Model', right_index=True, how='left')
    return plot_data

plot_data_90 = prepare_data(uniref90_df)
plot_data_50 = prepare_data(uniref50_df_filtered)

metrics = ['FCD', 'Cosine Similarity', 'Mahalanobis', 'Unfamiliarity', 'Perplexity']
bins = ['validation', '40_60', '60_80', '80_90', '90_100']

model_colors = {
    '8M_our_model_Uniref90': '#2ca02c',   # dark green
    '8M_our_model_Uniref50': '#98df8a',   # light green
    '35M_our_model_Uniref90': '#ff7f0e',  # dark orange
    '35M_our_model_Uniref50': '#ffbb78',  # light orange
}

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
bar_width = 0.35
spacing = 1.0

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for i, model in enumerate(['8M_our_model', '35M_our_model']):
        uniref90_vals = [plot_data_90.loc[plot_data_90['Model'] == model, f"{metric}" if bin == 'validation' else f"{metric}_{bin}"].values[0] for bin in bins]
        uniref50_vals = [plot_data_50.loc[plot_data_50['Model'] == model, f"{metric}" if bin == 'validation' else f"{metric}_{bin}"].values[0] for bin in bins]

        base_x = np.arange(len(bins)) * (2 * bar_width + spacing) + i * (bar_width * 2)
        key_90 = f'{model}_Uniref90'
        key_50 = f'{model}_Uniref50'
        ax.bar(base_x, uniref90_vals, bar_width, label=key_90, color=model_colors[key_90], alpha=0.8)
        ax.bar(base_x + bar_width, uniref50_vals, bar_width, label=key_50, color=model_colors[key_50], alpha=0.8)

    ax.set_xticks(np.arange(len(bins)) * (2 * bar_width + spacing) + bar_width)
    ax.set_xticklabels(bins)
    ax.set_title(metric)
    ax.set_ylabel(metric)

if len(metrics) < len(axes):
    for ax in axes[len(metrics):]:
        fig.delaxes(ax)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.92, 0.5))
fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.suptitle("Comparison between Uniref90 and Uniref50 models", fontsize=16, y=1.02)
plt.savefig('./plots/uniref90_vs_uniref50_comparison.png', bbox_inches='tight')
plt.close()
