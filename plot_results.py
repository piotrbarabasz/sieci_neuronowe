import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

results_file = 'results/training/model_results.csv'
df = pd.read_csv(results_file)

df.set_index('Model', inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC Score']
for i, metric in enumerate(metrics):
    ax = axes[i//3, i%3]
    df[metric].plot(kind='bar', ax=ax)
    ax.set_title(metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)

if len(metrics) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
output_file = 'results/training/performance_comparison.png'
plt.savefig(output_file, bbox_inches='tight')
plt.show()
