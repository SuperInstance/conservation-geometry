"""
Visualization 5: Alignment Coefficient Visualization
Plot α(G,a) for various graphs. Show boundary α=0.5 (conjecture C1).
High α = conservation works, low α = doesn't.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import os

OUT = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

def compute_alpha(L, a):
    """
    Alignment coefficient α(G,a) = (a^T L a) / (||L||_2 * ||a||^2)
    Measures how well the attribute aligns with the graph Laplacian's smooth eigenvectors.
    """
    eigs = np.linalg.eigvalsh(L)
    lambda_max = max(eigs)
    if lambda_max == 0 or np.dot(a, a) == 0:
        return 0
    return (a @ L @ a) / (lambda_max * np.dot(a, a))

def build_laplacian(A):
    D = np.diag(A.sum(axis=1))
    return D - A

# --- Generate diverse graphs and attributes ---
results = []

# Helper to create different graph types
def chain_graph(n):
    A = np.zeros((n, n))
    for i in range(n-1): A[i,i+1] = A[i+1,i] = 1
    return A

def cycle_graph(n):
    A = chain_graph(n)
    A[0, n-1] = A[n-1, 0] = 1
    return A

def complete_graph(n):
    return np.ones((n, n)) - np.eye(n)

def star_graph(n):
    A = np.zeros((n, n))
    for i in range(1, n): A[0,i] = A[i,0] = 1
    return A

def erdos_renyi(n, p):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if np.random.random() < p:
                A[i,j] = A[j,i] = 1
    # Ensure connected
    for i in range(n-1):
        A[i,i+1] = A[i+1,i] = 1
    return A

def watts_strogatz_approx(n, k=4, p_rewire=0.3):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k//2 + 1):
            if np.random.random() < p_rewire:
                target = np.random.randint(0, n)
            else:
                target = (i + j) % n
            if target != i:
                A[i,target] = A[target,i] = 1
    return A

# Smooth attributes (high conservation expected)
def smooth_attr(n):
    return np.linspace(0, 1, n) + np.random.normal(0, 0.05, n)

def constant_attr(n):
    return np.ones(n) * (2 + np.random.normal(0, 0.1))

# Rough attributes (low conservation expected)
def random_attr(n):
    return np.random.randn(n)

def oscillating_attr(n):
    return np.array([(-1)**k * (1 + np.random.normal(0, 0.1)) for k in range(n)])

def spike_attr(n):
    a = np.zeros(n)
    a[n//2] = 5.0
    return a

graph_builders = [
    ('Chain-8', lambda: chain_graph(8)),
    ('Chain-16', lambda: chain_graph(16)),
    ('Cycle-8', lambda: cycle_graph(8)),
    ('Cycle-16', lambda: cycle_graph(16)),
    ('Complete-6', lambda: complete_graph(6)),
    ('Complete-10', lambda: complete_graph(10)),
    ('Star-8', lambda: star_graph(8)),
    ('Star-16', lambda: star_graph(16)),
    ('ER-12-p0.3', lambda: erdos_renyi(12, 0.3)),
    ('ER-12-p0.5', lambda: erdos_renyi(12, 0.5)),
    ('ER-12-p0.7', lambda: erdos_renyi(12, 0.7)),
    ('WS-12-k4', lambda: watts_strogatz_approx(12, 4, 0.3)),
    ('WS-16-k6', lambda: watts_strogatz_approx(16, 6, 0.3)),
]

attr_builders = [
    ('smooth', smooth_attr),
    ('constant', constant_attr),
    ('random', random_attr),
    ('oscillating', oscillating_attr),
    ('spike', spike_attr),
]

for gname, gbuild in graph_builders:
    A = gbuild()
    n = A.shape[0]
    L = build_laplacian(A)
    eigs = np.sort(np.linalg.eigvalsh(L))
    spectral_gap = eigs[1] if len(eigs) > 1 else 0
    avg_degree = A.sum(axis=1).mean()

    for aname, abuild in attr_builders:
        a = abuild(n)
        alpha = compute_alpha(L, a)
        dirichlet = a @ L @ a
        results.append({
            'graph': gname,
            'attr': aname,
            'alpha': alpha,
            'dirichlet': dirichlet,
            'spectral_gap': spectral_gap,
            'avg_degree': avg_degree,
            'n': n,
        })

# --- Plot 1: Alignment coefficient heatmap ---
fig, ax = plt.subplots(figsize=(14, 8))

graph_names = sorted(set(r['graph'] for r in results))
attr_names = sorted(set(r['attr'] for r in results))

matrix = np.zeros((len(graph_names), len(attr_names)))
for r in results:
    gi = graph_names.index(r['graph'])
    ai = attr_names.index(r['attr'])
    matrix[gi, ai] = r['alpha']

im = ax.imshow(matrix, cmap=cm.RdYlGn, aspect='auto', vmin=0, vmax=0.5)
ax.set_xticks(range(len(attr_names)))
ax.set_xticklabels(attr_names, fontsize=10, rotation=30)
ax.set_yticks(range(len(graph_names)))
ax.set_yticklabels(graph_names, fontsize=10)

# Annotate cells
for i in range(len(graph_names)):
    for j in range(len(attr_names)):
        val = matrix[i, j]
        color = 'white' if val > 0.35 or val < 0.1 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8,
                color=color, fontweight='bold')

# α = 0.5 boundary line (conjecture)
ax.axhline(y=-0.5, color='red', linewidth=0)  # placeholder for legend

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Alignment Coefficient α(G, a)', fontsize=12)

ax.set_title('Alignment Coefficient α(G,a) Across Graph Types × Attribute Types\n'
             'Conjecture C1: α(G,a) ≤ 0.5  |  Green = conservation works, Red = doesn\'t',
             fontsize=13, fontweight='bold')
fig.savefig(os.path.join(OUT, '5_alignment_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 5_alignment_heatmap.png')

# --- Plot 2: α vs spectral gap scatter ---
fig, ax = plt.subplots(figsize=(12, 8))

attr_colors = {'smooth': '#4CAF50', 'constant': '#2196F3', 'random': '#FF5722',
               'oscillating': '#9C27B0', 'spike': '#FF9800'}

for r in results:
    color = attr_colors[r['attr']]
    size = 30 + 5 * r['n']
    ax.scatter(r['spectral_gap'], r['alpha'], c=color, s=size,
              alpha=0.7, edgecolors='black', linewidths=0.5, zorder=3)

# α = 0.5 boundary
ax.axhline(y=0.5, color='red', linewidth=2.5, linestyle='--', label='α = 0.5 (Conjecture C1)')
ax.axhline(y=0.25, color='orange', linewidth=1.5, linestyle=':', alpha=0.7, label='α = 0.25 (practical threshold)')

# Shade regions
ax.fill_between([0, 10], 0.5, 1.0, alpha=0.05, color='red')
ax.text(5, 0.65, 'PREDICTED EMPTY\n(conjecture C1)', fontsize=11,
        ha='center', color='red', alpha=0.5, fontweight='bold')

ax.set_xlabel('Spectral Gap λ₂', fontsize=13)
ax.set_ylabel('Alignment Coefficient α(G, a)', fontsize=13)
ax.set_title('Alignment Coefficient vs Spectral Gap\nConjecture C1: α(G,a) never exceeds 0.5',
             fontsize=13, fontweight='bold')

# Legend with attr types
for aname, acolor in attr_colors.items():
    ax.scatter([], [], c=acolor, s=80, label=f'{aname} attr', edgecolors='black')
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, None)
ax.set_ylim(-0.02, 0.6)
ax.grid(alpha=0.3)

fig.savefig(os.path.join(OUT, '5_alpha_vs_spectral_gap.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 5_alpha_vs_spectral_gap.png')

# --- Plot 3: α distribution by attribute type (violin-style) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: box plot by attribute type
by_attr = {aname: [] for aname in attr_names}
for r in results:
    by_attr[r['attr']].append(r['alpha'])

positions = range(len(attr_names))
bp = axes[0].boxplot([by_attr[a] for a in attr_names], positions=positions,
                     patch_artist=True, widths=0.6)
for patch, aname in zip(bp['boxes'], attr_names):
    patch.set_facecolor(attr_colors[aname])
    patch.set_alpha(0.6)
axes[0].set_xticks(positions)
axes[0].set_xticklabels(attr_names, fontsize=10)
axes[0].axhline(y=0.5, color='red', linewidth=2, linestyle='--', label='α = 0.5 (C1)')
axes[0].set_ylabel('Alignment Coefficient α(G, a)', fontsize=12)
axes[0].set_title('α Distribution by Attribute Type', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Right: histogram of all α values
all_alphas = [r['alpha'] for r in results]
axes[1].hist(all_alphas, bins=25, color='#2196F3', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0.5, color='red', linewidth=2.5, linestyle='--', label='α = 0.5 (C1)')
axes[1].axvline(x=0.25, color='orange', linewidth=1.5, linestyle=':', label='α = 0.25 (practical)')
axes[1].set_xlabel('Alignment Coefficient α(G, a)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of α Across All Experiments', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)

fig.suptitle('Alignment Coefficient Analysis — Testing Conjecture C1: α(G,a) ≤ 0.5',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT, '5_alpha_distribution.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 5_alpha_distribution.png')

# --- Summary statistics ---
print('\n  Alignment coefficient summary:')
print(f'    Total experiments: {len(results)}')
print(f'    Max α observed: {max(r["alpha"] for r in results):.4f}')
print(f'    Mean α (smooth): {np.mean([r["alpha"] for r in results if r["attr"]=="smooth"]):.4f}')
print(f'    Mean α (random): {np.mean([r["alpha"] for r in results if r["attr"]=="random"]):.4f}')
print(f'    Mean α (spike): {np.mean([r["alpha"] for r in results if r["attr"]=="spike"]):.4f}')
print(f'    Conjecture C1 (α ≤ 0.5) holds: {all(r["alpha"] <= 0.5 for r in results)}')
print('Done: Alignment coefficient visualizations')
