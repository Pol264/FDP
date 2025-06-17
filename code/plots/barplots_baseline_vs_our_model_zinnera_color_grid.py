import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib.colors import to_rgb
import colorsys

os.makedirs('./plots', exist_ok=True)

df = pd.read_csv('8M_35M_150M.csv', index_col=0)

models_to_plot = ['8M', '35M', '150M']
df_baseline = df[df['Model'].str.contains('baseline')].copy()
df_our_model = df[df['Model'].str.contains('our_model')].copy()

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

plot_data_baseline = prepare_data(df_baseline)
plot_data_our = prepare_data(df_our_model)

metrics = ['FCD', 'Cosine Similarity', 'Mahalanobis', 'Unfamiliarity', 'Perplexity']
bins = ['validation', '40_60', '60_80', '80_90', '90_100']

def lighten_color(color, amount=0.2):
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_r, new_g, new_b = colorsys.hls_to_rgb(h, min(1, l + amount), s)
    return (new_r, new_g, new_b)

base_palette = sns.color_palette("Set2", len(models_to_plot))
model_colors = {}
for i, model in enumerate(models_to_plot):
    base_color = base_palette[i]
    light_color = lighten_color(base_color, amount=0.15)
    model_colors[f'{model}_baseline'] = base_color
    model_colors[f'{model}_our_model'] = light_color

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
bar_width = 0.6
model_pairs = [(f'{m}_baseline', f'{m}_our_model') for m in models_to_plot]
total_groups = len(model_pairs)
spacing_factor = 4.0
positions = [j * (total_groups * 2 * bar_width + spacing_factor) for j in range(len(bins))]
legend_handles = {}

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for i, (baseline_label, our_model_label) in enumerate(model_pairs):
        for j, bin_label in enumerate(bins):
            if bin_label == 'validation':
                baseline_value = plot_data_baseline.loc[plot_data_baseline['Model'] == baseline_label, metric].values[0]
                our_model_value = plot_data_our.loc[plot_data_our['Model'] == our_model_label, metric].values[0]
            else:
                baseline_value = plot_data_baseline.loc[plot_data_baseline['Model'] == baseline_label, f"{metric}_{bin_label}"].values[0]
                our_model_value = plot_data_our.loc[plot_data_our['Model'] == our_model_label, f"{metric}_{bin_label}"].values[0]

            offset = positions[j] + i * 2 * bar_width + i * 0.3
            bar1 = ax.bar(offset, baseline_value, bar_width, color=model_colors[baseline_label])
            bar2 = ax.bar(offset + bar_width, our_model_value, bar_width, color=model_colors[our_model_label])

            if baseline_label not in legend_handles:
                legend_handles[baseline_label] = bar1[0]
            if our_model_label not in legend_handles:
                legend_handles[our_model_label] = bar2[0]

    mid_points = [pos + (total_groups * 2 * bar_width + (total_groups - 1) * 0.3) / 2 for pos in positions]
    ax.set_title(metric)
    ax.set_xticks(mid_points)
    ax.set_xticklabels(bins)
    ax.set_ylabel(metric)

if len(metrics) < len(axes):
    for ax in axes[len(metrics):]:
        fig.delaxes(ax)

fig.legend(legend_handles.values(), legend_handles.keys(), loc='center left', bbox_to_anchor=(0.92, 0.5))
fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.suptitle("Metric Comparison: Baseline vs Our Model", fontsize=16, y=1.02)
plt.savefig('./plots/baseline_vs_our_model_seaborn_color_grid.png', bbox_inches='tight')
plt.close()

