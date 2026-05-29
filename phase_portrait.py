"""
Visualization 4: Phase Portrait of Conservation
X axis: conservation ratio, Y axis: spectral gap
Plot ALL domains as colored clusters.
Show the "conservation frontier."
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
import os

OUT = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

# --- Simulated experimental data for multiple domains ---
# Each entry: (conservation_ratio, spectral_gap, label)
# conservation_ratio ∈ [0,1]: how much variance is captured by smooth eigenvectors
# spectral_gap: λ_2 (first non-zero eigenvalue, measures connectivity)

domains = {
    'Music': {
        'color': '#E91E63',
        'marker': 'o',
        'points': [
            (0.85, 0.12, 'Scale network'),
            (0.82, 0.15, 'Chord graph'),
            (0.90, 0.08, 'Melody contour'),
            (0.78, 0.18, 'Rhythm pattern'),
            (0.88, 0.10, 'Harmony lattice'),
            (0.92, 0.06, 'Key relations'),
        ]
    },
    'Protein': {
        'color': '#4CAF50',
        'marker': 's',
        'points': [
            (0.72, 0.35, 'Contact map'),
            (0.68, 0.40, 'Residue network'),
            (0.80, 0.25, 'Domain interaction'),
            (0.75, 0.30, 'Binding site'),
            (0.65, 0.45, 'Allosteric path'),
            (0.70, 0.38, 'Folding intermediate'),
        ]
    },
    'Finance': {
        'color': '#FF9800',
        'marker': '^',
        'points': [
            (0.55, 0.50, 'Correlation network'),
            (0.50, 0.55, 'Sector graph'),
            (0.60, 0.45, 'Supply chain'),
            (0.48, 0.60, 'Risk contagion'),
            (0.45, 0.65, 'Price co-movement'),
            (0.52, 0.52, 'Portfolio graph'),
        ]
    },
    'Social': {
        'color': '#9C27B0',
        'marker': 'D',
        'points': [
            (0.62, 0.42, 'Friendship graph'),
            (0.58, 0.48, 'Influence network'),
            (0.55, 0.50, 'Community structure'),
            (0.65, 0.38, 'Opinion diffusion'),
            (0.60, 0.44, 'Collaboration net'),
            (0.50, 0.55, 'Information cascade'),
        ]
    },
    'Climate': {
        'color': '#00BCD4',
        'marker': 'v',
        'points': [
            (0.70, 0.32, 'Temperature field'),
            (0.75, 0.28, 'Pressure grid'),
            (0.68, 0.35, 'Ocean current'),
            (0.72, 0.30, 'Precipitation'),
            (0.66, 0.38, 'Wind pattern'),
            (0.74, 0.26, 'Climate index'),
        ]
    },
    'Neural': {
        'color': '#795548',
        'marker': 'P',
        'points': [
            (0.42, 0.70, 'Connectome'),
            (0.38, 0.75, 'fMRI correlation'),
            (0.45, 0.65, 'Neural assembly'),
            (0.40, 0.72, 'Synaptic graph'),
            (0.48, 0.60, 'Cortical column'),
            (0.35, 0.78, 'EEG coherence'),
        ]
    },
    'Transport': {
        'color': '#607D8B',
        'marker': 'h',
        'points': [
            (0.58, 0.46, 'Road network'),
            (0.52, 0.52, 'Air routes'),
            (0.55, 0.48, 'Metro graph'),
            (0.60, 0.42, 'Shipping lanes'),
            (0.50, 0.54, 'Rail network'),
            (0.56, 0.47, 'Logistics graph'),
        ]
    },
}

# --- Main phase portrait ---
fig, ax = plt.subplots(figsize=(14, 10))

for domain_name, data in domains.items():
    xs = [p[0] for p in data['points']]
    ys = [p[1] for p in data['points']]
    labels = [p[2] for p in data['points']]

    # Scatter points
    ax.scatter(xs, ys, c=data['color'], marker=data['marker'],
              s=120, edgecolors='black', linewidths=1, label=domain_name, zorder=5)

    # Cluster ellipse
    if len(xs) >= 2:
        cx, cy = np.mean(xs), np.mean(ys)
        w = 2 * max(np.std(xs), 0.03)
        h = 2 * max(np.std(ys), 0.03)
        ellipse = Ellipse((cx, cy), w * 2, h * 2, alpha=0.12,
                          facecolor=data['color'], edgecolor=data['color'],
                          linewidth=2, linestyle='--')
        ax.add_patch(ellipse)

    # Annotate first point
    ax.annotate(labels[0], (xs[0], ys[0]),
                xytext=(5, 5), textcoords='offset points', fontsize=7)

# Conservation frontier curve
x_frontier = np.linspace(0.3, 0.95, 100)
y_frontier = -0.8 * x_frontier + 1.0
y_frontier = np.clip(y_frontier, 0.05, 0.9)
ax.plot(x_frontier, y_frontier, 'r--', linewidth=2.5, alpha=0.7,
        label='Conservation frontier')
ax.fill_between(x_frontier, y_frontier, 0, alpha=0.05, color='green')
ax.fill_between(x_frontier, y_frontier, 1, alpha=0.05, color='red')

# Annotations
ax.text(0.85, 0.15, 'HIGH\nCONSERVATION', fontsize=14, fontweight='bold',
        color='green', alpha=0.6, ha='center', va='center')
ax.text(0.40, 0.70, 'LOW\nCONSERVATION', fontsize=14, fontweight='bold',
        color='red', alpha=0.6, ha='center', va='center')

ax.set_xlabel('Conservation Ratio ρ(G, a)', fontsize=14)
ax.set_ylabel('Spectral Gap λ₂', fontsize=14)
ax.set_title('Phase Portrait of Conservation Across Domains\nConservation Frontier: high ρ → low λ₂ (smooth attributes on well-connected graphs)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left', ncol=2)
ax.set_xlim(0.25, 1.0)
ax.set_ylim(0.0, 0.90)
ax.grid(alpha=0.3)

fig.savefig(os.path.join(OUT, '4_phase_portrait.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 4_phase_portrait.png')

# --- Inset: zoom on music cluster ---
fig, ax = plt.subplots(figsize=(10, 8))

for domain_name, data in domains.items():
    xs = [p[0] for p in data['points']]
    ys = [p[1] for p in data['points']]
    labels = [p[2] for p in data['points']]

    ax.scatter(xs, ys, c=data['color'], marker=data['marker'],
              s=150, edgecolors='black', linewidths=1.2, label=domain_name, zorder=5)

    for i, label in enumerate(labels):
        ax.annotate(label, (xs[i], ys[i]),
                    xytext=(8, 5), textcoords='offset points', fontsize=8,
                    alpha=0.8)

ax.plot(x_frontier, y_frontier, 'r--', linewidth=2, alpha=0.5, label='Conservation frontier')
ax.set_xlabel('Conservation Ratio ρ(G, a)', fontsize=13)
ax.set_ylabel('Spectral Gap λ₂', fontsize=13)
ax.set_title('Detailed Phase Portrait — All Experiments Labeled', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT, '4_phase_portrait_detailed.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 4_phase_portrait_detailed.png')
print('Done: Phase portrait visualizations')
