"""
Visualization 2: Eigenvalue Shells
For each eigenvalue, draw a sphere of radius sqrt(lambda_k).
Conservation = inner spheres much smaller than outer.
Compare structured vs random graphs.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

OUT = os.path.dirname(os.path.abspath(__file__))

def get_eigenvalues(L):
    """Get sorted eigenvalues of Laplacian."""
    eigs = np.linalg.eigvalsh(L)
    return np.sort(eigs)

def build_laplacian(A):
    D = np.diag(A.sum(axis=1))
    return D - A

# --- Build example graphs ---
np.random.seed(42)

# Music-style graph (highly structured)
n_music = 10
A_music = np.zeros((n_music, n_music))
# Chain with some cross-connections (scale-like structure)
for i in range(n_music - 1):
    A_music[i, i+1] = 1
    A_music[i+1, i] = 1
# Add some harmony connections (every 3rd)
for i in range(n_music - 3):
    A_music[i, i+3] = 1
    A_music[i+3, i] = 1
L_music = build_laplacian(A_music)
eigs_music = get_eigenvalues(L_music)

# Random graph
n_random = 10
A_random = np.zeros((n_random, n_random))
for i in range(n_random):
    for j in range(i+1, n_random):
        if np.random.random() < 0.4:
            A_random[i, j] = 1
            A_random[j, i] = 1
# Ensure connected
for i in range(n_random - 1):
    A_random[i, i+1] = 1
    A_random[i+1, i] = 1
L_random = build_laplacian(A_random)
eigs_random = get_eigenvalues(L_random)

# Protein-style graph (modular)
n_prot = 12
A_prot = np.zeros((n_prot, n_prot))
# 3 modules of 4 nodes each, fully connected within
for mod in range(3):
    base = mod * 4
    for i in range(4):
        for j in range(i+1, 4):
            A_prot[base+i, base+j] = 1
            A_prot[base+j, base+i] = 1
# Inter-module connections (sparse)
A_prot[0, 4] = 1; A_prot[4, 0] = 1
A_prot[4, 8] = 1; A_prot[8, 4] = 1
A_prot[3, 8] = 1; A_prot[8, 3] = 1
L_prot = build_laplacian(A_prot)
eigs_prot = get_eigenvalues(L_prot)

graphs = {
    'Music (Scale/Harmony)': (eigs_music, '#2196F3'),
    'Protein (Modular)': (eigs_prot, '#4CAF50'),
    'Random (Erdős–Rényi)': (eigs_random, '#FF5722'),
}

# --- Plot 1: Eigenvalue spectrum comparison ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for idx, (name, (eigs, color)) in enumerate(graphs.items()):
    ax = axes[idx]
    ax.bar(range(len(eigs)), eigs, color=color, alpha=0.7, edgecolor='black')
    ax.set_ylabel('λ_k', fontsize=12)
    ax.set_title(f'{name} — Eigenvalue Spectrum', fontsize=12, fontweight='bold')

    # Mark spectral gap
    if len(eigs) > 1:
        gap_idx = 1  # First non-zero eigenvalue gap
        gap = eigs[gap_idx] - eigs[0]
        ax.annotate(f'Spectral gap: {gap:.2f}',
                    xy=(gap_idx, eigs[gap_idx]),
                    xytext=(gap_idx + 2, eigs[gap_idx] * 0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')

    ax.grid(axis='y', alpha=0.3)

axes[-1].set_xlabel('Eigenvalue index k', fontsize=12)
fig.suptitle('Spectral Structure: Eigenvalue Spectra of Graph Laplacians\nConservation ∝ inner eigenvalues much smaller than outer',
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT, '2_eigenvalue_spectrum.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 2_eigenvalue_spectrum.png')

# --- Plot 2: Eigenvalue shells (3D) ---
fig = plt.figure(figsize=(18, 6))

for idx, (name, (eigs, color)) in enumerate(graphs.items()):
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

    # Draw nested spheres for each eigenvalue
    n_eigs = len(eigs)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones(np.size(u)), np.cos(v))

    max_r = np.sqrt(max(eigs[-1], 1))
    for k in range(n_eigs - 1, -1, -1):
        r = np.sqrt(max(eigs[k], 0.01))
        alpha = 0.05 + 0.15 * (k / max(n_eigs - 1, 1))
        ax.plot_surface(r * x_unit, r * y_unit, r * z_unit,
                       color=color, alpha=alpha, linewidth=0)

    # Draw axis line showing shell radii
    ax.plot([0, 0], [0, 0], [0, max_r * 1.1], 'k-', linewidth=2)
    for k in [0, n_eigs // 2, n_eigs - 1]:
        r = np.sqrt(max(eigs[k], 0.01))
        ax.scatter([0], [0], [r], c='black', s=50, zorder=10)
        ax.text(0.1, 0, r, f'λ_{k}={eigs[k]:.1f}', fontsize=8)

    ax.set_title(f'{name}\nShell radii ∝ √λ_k', fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

fig.suptitle('Eigenvalue Shells — Nested Spheres of Radius √λ_k\nConservation = inner shells much smaller',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT, '2_eigenvalue_shells.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 2_eigenvalue_shells.png')

# --- Plot 3: Cumulative eigenvalue distribution ---
fig, ax = plt.subplots(figsize=(10, 6))

for name, (eigs, color) in graphs.items():
    cum = np.cumsum(eigs) / np.sum(eigs)
    ax.plot(range(len(eigs)), cum, 'o-', color=color, label=name, linewidth=2, markersize=6)
    # Shade the "conservation zone"
    if len(eigs) > 1:
        pct_50 = np.searchsorted(cum, 0.5)
        ax.axvline(x=pct_50, color=color, linestyle='--', alpha=0.4)
        ax.text(pct_50 + 0.1, cum[pct_50], f'50% at k={pct_50}',
                fontsize=8, color=color)

ax.set_xlabel('Eigenvalue index k', fontsize=12)
ax.set_ylabel('Cumulative eigenvalue fraction', fontsize=12)
ax.set_title('Cumulative Eigenvalue Distribution\nConservation: 50% of spectral energy in few eigenvalues',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT, '2_cumulative_eigenvalues.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 2_cumulative_eigenvalues.png')
print('Done: Eigenvalue shell visualizations')
