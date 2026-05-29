"""
Visualization 1: Laplacian as a Rubber Sheet
Embed a graph in 2D, show attribute as height (3D surface),
apply Laplacian to show how it "flattens" the surface.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# --- 5-node graph ---
# Edges: 0-1, 1-2, 2-3, 3-4, 0-4, 1-3
edges = [(0,1),(1,2),(2,3),(3,4),(0,4),(1,3)]
n = 5

# Spring layout positions (hand-tuned for aesthetics)
pos = np.array([
    [0.0, 0.5],    # 0
    [1.0, 1.0],    # 1
    [2.0, 0.5],    # 2
    [1.5, -0.3],   # 3
    [0.5, -0.3],   # 4
])

# Build adjacency and Laplacian
A = np.zeros((n, n))
for i, j in edges:
    A[i, j] = 1
    A[j, i] = 1
D = np.diag(A.sum(axis=1))
L = D - A

# Attribute vectors to visualize
attributes = {
    'Smooth (conserved)': np.array([3.0, 3.1, 3.0, 2.9, 3.0]),
    'Rough (not conserved)': np.array([0.0, 5.0, 1.0, 4.0, 2.0]),
    'Gradient': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
}

for title_attr, attr in attributes.items():
    lap_attr = L @ attr  # Laplacian applied

    fig = plt.figure(figsize=(16, 6))

    # --- Original surface ---
    ax1 = fig.add_subplot(131, projection='3d')
    xs, ys = pos[:, 0], pos[:, 1]

    # Draw triangulated surface
    from scipy.spatial import Delaunay
    tri = Delaunay(pos)
    ax1.plot_trisurf(xs, ys, attr, triangles=tri.simplices,
                     cmap=cm.viridis, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax1.scatter(xs, ys, attr, c=attr, cmap=cm.viridis, s=200, edgecolors='black', zorder=5)

    for i, j in edges:
        ax1.plot([xs[i], xs[j]], [ys[i], ys[j]], [attr[i], attr[j]],
                 'k-', alpha=0.5, linewidth=1.5)

    for k in range(n):
        ax1.text(xs[k], ys[k], attr[k] + 0.3, f'{attr[k]:.1f}', fontsize=9,
                 ha='center', fontweight='bold')

    ax1.set_title(f'Original: {title_attr}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Attribute value')
    ax1.set_zlim(-1, 6)

    # --- Laplacian-transformed surface ---
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(xs, ys, lap_attr, triangles=tri.simplices,
                     cmap=cm.coolwarm, alpha=0.6, edgecolor='k', linewidth=0.5)
    ax2.scatter(xs, ys, lap_attr, c=lap_attr, cmap=cm.coolwarm, s=200, edgecolors='black', zorder=5)

    for i, j in edges:
        ax2.plot([xs[i], xs[j]], [ys[i], ys[j]], [lap_attr[i], lap_attr[j]],
                 'k-', alpha=0.5, linewidth=1.5)

    for k in range(n):
        ax2.text(xs[k], ys[k], lap_attr[k] + 0.2, f'{lap_attr[k]:.1f}', fontsize=9,
                 ha='center', fontweight='bold')

    dirichlet = attr @ L @ attr
    ax2.set_title(f'After Laplacian: L·f\nDirichlet energy = {dirichlet:.2f}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Laplacian value')
    ax2.set_zlim(-6, 6)

    # --- Comparison bar chart ---
    ax3 = fig.add_subplot(133)
    x_pos = np.arange(n)
    width = 0.35
    bars1 = ax3.bar(x_pos - width/2, attr, width, label='Original f', color='#2196F3', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, lap_attr, width, label='Laplacian L·f', color='#FF5722', alpha=0.8)
    ax3.set_xlabel('Node')
    ax3.set_ylabel('Value')
    ax3.set_title(f'Node-wise comparison\n‖f‖² = {np.dot(attr,attr):.1f}, ‖Lf‖² = {np.dot(lap_attr,lap_attr):.1f}',
                  fontsize=11, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.legend()
    ax3.axhline(y=0, color='k', linewidth=0.5)
    ax3.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    safe_name = title_attr.replace(' ', '_').replace('(', '').replace(')', '')
    fig.savefig(os.path.join(OUT, f'1_rubber_sheet_{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: 1_rubber_sheet_{safe_name}.png')

# --- Conservation summary figure ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (title_attr, attr) in enumerate(attributes.items()):
    ax = axes[idx]
    lap_attr = L @ attr
    dirichlet = attr @ L @ attr
    norm_sq = np.dot(attr, attr)
    ratio = dirichlet / (norm_sq * np.max(np.linalg.eigvalsh(L))) if norm_sq > 0 else 0

    # Draw graph with node colors = attribute
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    node_colors = attr
    nx.draw(G, pos, ax=ax, node_color=node_colors, cmap=cm.viridis,
            node_size=500, edgecolors='black', with_labels=True,
            font_color='white', font_weight='bold', width=2)
    ax.set_title(f'{title_attr}\nDirichlet={dirichlet:.2f}, ratio={ratio:.3f}',
                 fontsize=11, fontweight='bold')

fig.suptitle('Conservation = Low Dirichlet Energy for Smooth Attributes', fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT, '1_rubber_sheet_summary.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved: 1_rubber_sheet_summary.png')
print('Done: Laplacian rubber sheet visualizations')
