import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from matplotlib.colors import to_rgb
import colorsys

os.makedirs('./plots', exist_ok=True)

files = {
    '8M_baseline': 'external_datasets_8M.txt',
    '8M_our_model': 'external_datasets_8M_our_model.txt',
    '35M_baseline': 'external_datasets_35M.txt',
    '35M_our_model': 'external_datasets_35M_our_model.txt',
    '150M_baseline': 'external_datasets_150M.txt',
    '150M_our_model': 'external_datasets_150M_our_model.txt',
}

data = []

for model, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"Missing file: {filepath}")
        continue
    with open(filepath, 'r') as f:
        content = f.read().strip()
        entries = re.findall(
            r'(\w+_\w+)\s+FCD: ([\d\.\-]+)\n\1\s+Cosine Similarity: ([\d\.\-]+)\n\1\s+Mahalanobis: ([\d\.\-]+)\n\1\s+Unfamiliarity: ([\d\.\-]+)\n\1\s+Dataset Perplexity: ([\d\.\-]+)',
            content
        )
        for entry in entries:
            task, fcd, cos_sim, maha, unfam, perp = entry
            data.append({
                'Model': model,
                'Task': task,
                'FCD': float(fcd),
                'Cosine Similarity': float(cos_sim),
                'Mahalanobis': float(maha),
                'Unfamiliarity': float(unfam),
                'Perplexity': float(perp)
            })

df = pd.DataFrame(data)

models_to_plot = ['8M', '35M', '150M']
df_baseline = df[df['Model'].str.contains('baseline')]
df_our_model = df[df['Model'].str.contains('our_model')]

def pivot_data(df):
    return df.pivot(index='Model', columns='Task', values=['FCD', 'Cosine Similarity', 'Mahalanobis', 'Unfamiliarity', 'Perplexity'])

pivot_baseline = pivot_data(df_baseline)
pivot_our = pivot_data(df_our_model)

metrics = ['FCD', 'Cosine Similarity', 'Mahalanobis', 'Unfamiliarity', 'Perplexity']
tasks = ['secondary_structure', 'remote_homology']

def lighten_color(color, amount=0.2):
    r, g, b = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, min(1, l + amount), s)

base_palette = sns.color_palette("Set2", len(models_to_plot))
model_colors = {}
for i, model in enumerate(models_to_plot):
    base = base_palette[i]
    light = lighten_color(base, amount=0.15)
    model_colors[f'{model}_baseline'] = base
    model_colors[f'{model}_our_model'] = light

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
bar_width = 0.35
legend_handles = {}

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for i, task in enumerate(tasks):
        for j, model in enumerate(models_to_plot):
            bl = f"{model}_baseline"
            om = f"{model}_our_model"
            offset = i * (len(models_to_plot) * 2 * bar_width + 2) + j * 2 * bar_width

            if bl in pivot_baseline.index and om in pivot_our.index:
                bl_val = pivot_baseline.loc[bl, (metric, task)]
                om_val = pivot_our.loc[om, (metric, task)]

                bar1 = ax.bar(offset, bl_val, bar_width, color=model_colors[bl])
                bar2 = ax.bar(offset + bar_width, om_val, bar_width, color=model_colors[om])

                if bl not in legend_handles:
                    legend_handles[bl] = bar1[0]
                if om not in legend_handles:
                    legend_handles[om] = bar2[0]

    ax.set_title(metric)
    xticks = [i * (len(models_to_plot) * 2 * bar_width + 2) + (len(models_to_plot) * bar_width) for i in range(len(tasks))]
    ax.set_xticks(xticks)
    ax.set_xticklabels(tasks)
    ax.set_ylabel(metric)

if len(metrics) < len(axes):
    for ax in axes[len(metrics):]:
        fig.delaxes(ax)

fig.legend(legend_handles.values(), legend_handles.keys(), loc='center left', bbox_to_anchor=(0.92, 0.5))
fig.tight_layout(rect=[0, 0, 0.9, 1])
fig.suptitle("Metric Comparison per Task: Baseline vs Our Model", fontsize=16, y=1.02)
plt.savefig('./plots/parsed_external_datasets_grid.png', bbox_inches='tight')
plt.close()

