"""Figures for Problem 2: accuracy comparison and PGD-step convergence."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KS = [10, 20, 50]
data = {k: {r['name']: r for r in json.load(open(f'results/k{k}.json'))} for k in KS}
MODELS = ['Linf-trained', 'L2-trained', 'RAMP']
PRETTY = {'Linf-trained': r'$\ell_\infty$-trained', 'L2-trained': r'$\ell_2$-trained',
          'RAMP': 'RAMP'}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))

# ---- panel 1: the four accuracies at k = 50 (strongest attack) -------------
ax = axes[0]
metrics = [('clean', 'standard'), ('linf', r'$\ell_\infty$ PGD'),
           ('l2', r'$\ell_2$ PGD'), ('union', 'union')]
colors = ['#9e9e9e', '#1f77b4', '#ff7f0e', '#d62728']
w = 0.2
for j, (key, lab) in enumerate(metrics):
    vals = [data[50][m][key] for m in MODELS]
    xs = [i + (j - 1.5) * w for i in range(len(MODELS))]
    bars = ax.bar(xs, vals, w, label=lab, color=colors[j])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8)
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([PRETTY[m] for m in MODELS])
ax.set_ylabel('accuracy (%)')
ax.set_title('Accuracy at PGD-50\n' r'$\epsilon_\infty=8/255$, $\epsilon_2=0.75$')
ax.set_ylim(0, 100)
ax.legend(fontsize=9, ncol=2)
ax.grid(axis='y', alpha=0.3)

# ---- panel 2: convergence in the number of PGD steps ----------------------
ax = axes[1]
styles = {'Linf-trained': 'o-', 'L2-trained': 's-', 'RAMP': '^-'}
for m in MODELS:
    ax.plot(KS, [data[k][m]['union'] for k in KS], styles[m],
            label=PRETTY[m] + ' union')
for m in MODELS:
    ax.plot(KS, [data[k][m]['l2'] for k in KS], styles[m], alpha=0.35,
            linestyle='--', label=PRETTY[m] + r' $\ell_2$')
ax.set_xlabel('PGD steps $k$')
ax.set_ylabel('robust accuracy (%)')
ax.set_title('Attack strength vs. number of steps\n(solid = union, dashed = $\\ell_2$)')
ax.set_xticks(KS)
ax.legend(fontsize=7, ncol=2)
ax.grid(alpha=0.3)

# ---- panel 3: which adversary breaks which points (k = 50) ----------------
ax = axes[2]
keys = [('only_l2', r'broken by $\ell_2$ only', '#ff7f0e'),
        ('only_linf', r'broken by $\ell_\infty$ only', '#1f77b4'),
        ('both_broken', 'broken by both', '#7b1fa2')]
bottom = [0.0] * len(MODELS)
for key, lab, col in keys:
    vals = [data[50][m][key] for m in MODELS]
    ax.bar(range(len(MODELS)), vals, 0.5, bottom=bottom, label=lab, color=col)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.6:
            ax.text(i, b + v / 2, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color='white')
    bottom = [b + v for b, v in zip(bottom, vals)]
union_vals = [data[50][m]['union'] for m in MODELS]
ax.bar(range(len(MODELS)), union_vals, 0.5, bottom=bottom,
       label='survives both (union acc.)', color='#2e7d32')
for i, (v, b) in enumerate(zip(union_vals, bottom)):
    ax.text(i, b + v / 2, f'{v:.2f}', ha='center', va='center',
            fontsize=8, color='white')
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels([PRETTY[m] for m in MODELS])
ax.set_ylabel('% of all test points')
ax.set_title('Decomposition of the union threat model\n(PGD-50; remainder = clean errors)')
ax.legend(fontsize=8, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/robustness.png', dpi=150, bbox_inches='tight')
print('wrote results/robustness.png')
