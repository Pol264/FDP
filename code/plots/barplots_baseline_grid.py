import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


os.makedirs('./plots', exist_ok=True)

df = pd.read_csv('8M_35M_150M_650M.csv', index_col=0)

baseline_models = ['8M_baseline', '35M_baseline', '150M_baseline','650M_baseline']
df = df[df['Model'].isin(baseline_models)]

validation = df[df['Set'] == 'validation']
testing = df[df['Set'].str.startswith('test')].copy()
testing['Bin'] = testing['Set'].apply(lambda x: '_'.join(x.split('_')[2:]))

metrics = ['FCD', 'Cosine Similarity', 'Mahalanobis', 'Unfamiliarity', 'Perplexity']
bins = ['validation', '40_60', '60_80', '80_90', '90_100']

mean_testing = testing.groupby(['Model', 'Bin'])[numeric_cols].mean().reset_index()

plot_data = validation[['Model'] + metrics].copy()
for bin_group in mean_testing['Bin'].unique():
    bin_df = mean_testing[mean_testing['Bin'] == bin_group].drop(columns=['Bin']).set_index('Model')
    bin_df.columns = [f"{col}_{bin_group}" for col in bin_df.columns]
    plot_data = plot_data.merge(bin_df, left_on='Model', right_index=True, how='left')

models = plot_data['Model'].tolist()
palette = sns.cubehelix_palette(len(models), rot=-.4, dark=0.3, light=0.6)
model_colors = dict(zip(models, palette))

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
bar_width = 0.15
positions = np.arange(len(bins))

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for i, model in enumerate(models):
        values = [
            plot_data.loc[plot_data['Model'] == model, metric].values[0],
            plot_data.loc[plot_data['Model'] == model, f"{metric}_40_60"].values[0],
            plot_data.loc[plot_data['Model'] == model, f"{metric}_60_80"].values[0],
            plot_data.loc[plot_data['Model'] == model, f"{metric}_80_90"].values[0],
            plot_data.loc[plot_data['Model'] == model, f"{metric}_90_100"].values[0],
        ]
        ax.bar(positions + i * bar_width, values, bar_width, label=model if idx == 0 else "", color=model_colors[model])

    ax.set_title(metric)
    ax.set_xticks(positions + bar_width * (len(models)-1)/2)
    ax.set_xticklabels(bins)
    ax.set_ylabel(metric)

if len(metrics) < len(axes):
    for ax in axes[len(metrics):]:
        fig.delaxes(ax)

handles = [plt.Rectangle((0,0),1,1, color=model_colors[m]) for m in models]
fig.legend(handles, models, loc='center left', bbox_to_anchor=(0.92, 0.5))
fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.suptitle("Baseline Models Across Bins", fontsize=16, y=1.02)
plt.savefig('./plots/all_baselines_metrics_grid_with_650M.png', bbox_inches='tight')
plt.close()
