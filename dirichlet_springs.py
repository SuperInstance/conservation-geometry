"""
Visualization 3: Dirichlet Energy as a Spring System
Each edge is a spring, attribute values are displacements.
Dirichlet energy = total spring energy = f^T L f.
Animate: perturb one node, watch energy propagate.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

OUT = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

# --- Build a larger graph for visual impact ---
n = 8
# Ring graph with cross-connections
edges = [(i, (i+1) % n) for i in range(n)]
edges += [(0, 3), (1, 5), (2, 6), (4, 7)]  # cross-connections

A = np.zeros((n, n))
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1
D = np.diag(A.sum(axis=1))
L = D - A

# Spring layout positions (circular + perturbation)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
pos = np.column_stack([np.cos(angles), np.sin(angles)])

# --- Animation frames: perturb node 0 and propagate ---
n_frames = 30

# Initial smooth state
f_init = np.ones(n) * 2.0

# Perturbation at node 0
f_perturbed = f_init.copy()
f_perturbed[0] = 5.0

# Diffusion: f(t+1) = f(t) - alpha * L @ f(t)
alpha = 0.05  # diffusion rate
frames = [f_perturbed.copy()]
f = f_perturbed.copy()
for _ in range(n_frames - 1):
    f = f - alpha * L @ f
    frames.append(f.copy())

# --- Plot 1: Energy propagation ---
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
frame_indices = [0, 2, 5, 10, 15, 20, 25, 29]

for idx, fi in enumerate(frame_indices):
    ax = axes[idx // 4][idx % 4]
    f_curr = frames[fi]
    dirichlet = f_curr @ L @ f_curr

    # Draw edges as springs (color by strain)
    for i, j in edges:
        strain = abs(f_curr[i] - f_curr[j])
        max_strain = max(abs(f_curr[a] - f_curr[b]) for a, b in edges)
        strain_norm = strain / max(max_strain, 0.01)
        color = cm.coolwarm(strain_norm)
        lw = 1 + 3 * strain_norm
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=color, linewidth=lw, alpha=0.7, zorder=1)

    # Draw nodes (size ∝ attribute, color ∝ value)
    node_sizes = 200 + 100 * f_curr
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=f_curr, cmap=cm.viridis,
                        s=node_sizes, edgecolors='black', linewidths=1.5, zorder=5)

    # Labels
    for k in range(n):
        ax.text(pos[k, 0] * 1.25, pos[k, 1] * 1.25, f'{f_curr[k]:.1f}',
                fontsize=7, ha='center', va='center')

    ax.set_title(f't={fi} | Dirichlet E = {dirichlet:.2f}', fontsize=10, fontweight='bold')
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')

fig.suptitle('Dirichlet Energy Propagation: Perturbation Diffuses Through the Graph\n'
             'Edge color = spring strain | Node size ∝ attribute value | Conservation = low energy equilibrium',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT, '3_spring_energy_propagation.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 3_spring_energy_propagation.png')

# --- Plot 2: Energy decay over time ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

energies = [f @ L @ f for f in frames]
ax1.plot(range(n_frames), energies, 'b-', linewidth=2.5)
ax1.fill_between(range(n_frames), energies, alpha=0.2)
ax1.set_xlabel('Time step (diffusion)', fontsize=12)
ax1.set_ylabel('Dirichlet Energy f^T L f', fontsize=12)
ax1.set_title('Energy Decay Under Diffusion\nConservation = system settles to low energy', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Node values over time
for k in range(n):
    values = [frames[t][k] for t in range(n_frames)]
    ax2.plot(range(n_frames), values, linewidth=1.5, label=f'Node {k}')
ax2.set_xlabel('Time step (diffusion)', fontsize=12)
ax2.set_ylabel('Attribute value', fontsize=12)
ax2.set_title('Attribute Values Converging\nConservation = values stay close during diffusion', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, ncol=2)
ax2.grid(alpha=0.3)

fig.savefig(os.path.join(OUT, '3_energy_decay.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 3_energy_decay.png')

# --- Plot 3: Spring energy decomposition ---
fig, ax = plt.subplots(figsize=(12, 6))

# Show energy per edge at different time steps
frame_indices2 = [0, 5, 15, 29]
x_pos = np.arange(len(edges))
width = 0.2
colors = ['#FF5722', '#FF9800', '#4CAF50', '#2196F3']

for idx, fi in enumerate(frame_indices2):
    f_curr = frames[fi]
    edge_energies = [(f_curr[i] - f_curr[j])**2 for i, j in edges]
    ax.bar(x_pos + idx * width, edge_energies, width,
           label=f't={fi}', color=colors[idx], alpha=0.8, edgecolor='black')

ax.set_xlabel('Edge', fontsize=12)
ax.set_ylabel('Spring Energy (f_i - f_j)²', fontsize=12)
ax.set_title('Per-Edge Spring Energy Over Time\nConservation = all springs relax to low energy',
             fontsize=13, fontweight='bold')
ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels([f'{i}-{j}' for i, j in edges], fontsize=8, rotation=45)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

fig.savefig(os.path.join(OUT, '3_edge_energy_decomposition.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 3_edge_energy_decomposition.png')
print('Done: Dirichlet energy spring visualizations')
