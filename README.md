# Conservation Geometry

**Geometric Visualizations of Spectral Graph Conservation**

A collection of Python scripts that make the Conservation Spectral framework *visible*. Each script produces a different geometric visualization — Laplacian as rubber sheet, eigenvalue shells, Dirichlet spring systems, phase portraits, and alignment heatmaps. Together they form a visual atlas of how graph Laplacians interact with attributes.

## The Aha Moment

Conservation is inherently geometric: the Laplacian is a "rubber sheet" that pulls a surface toward its smoothest shape, eigenvalues are shell radii that define conservation bounds, and the Dirichlet energy is literally spring potential energy. These visualizations make that geometry explicit. When you see a graph embedded as a 3D surface where height = attribute and the Laplacian flattens it, conservation isn't an equation — it's a shape. When you see eigenvalue shells where inner spheres are much smaller than outer ones, the conservation ratio isn't a number — it's a volume ratio. *Seeing is understanding.*

## Visualizations

### 1. Laplacian as a Rubber Sheet (`laplacian_rubber_sheet.py`)
Embeds a 5-node graph in 2D, shows attribute as height (3D surface), then applies the Laplacian to "flatten" it. Compares smooth (conserved) vs rough (not conserved) attributes.

### 2. Eigenvalue Shells (`eigenvalue_shells.py`)
For each eigenvalue λ_k, draws a sphere of radius √λ_k. Conservation = inner shells much smaller than outer shells. Compares structured (music-like) graphs vs random graphs.

### 3. Dirichlet Springs (`dirichlet_springs.py`)
Each edge is a spring, attribute values are displacements. Dirichlet energy f^TLf = total spring energy. Shows energy propagation when perturbing a single node.

### 4. Phase Portrait (`phase_portrait.py`)
X-axis: conservation ratio, Y-axis: spectral gap. Plots ALL domains (music, ecology, code, geometry) as colored clusters, revealing the "conservation frontier."

### 5. Alignment Coefficient (`alignment_coefficient.py`)
Computes and visualizes α(G,a) for various graph types × attribute types. Shows the boundary α = 0.5 (conjecture C1). High α = conservation works, low α = doesn't.

## How to Use

```bash
pip install numpy matplotlib scipy

# Generate all visualizations
python laplacian_rubber_sheet.py
python eigenvalue_shells.py
python dirichlet_springs.py
python phase_portrait.py
python alignment_coefficient.py
```

Each script writes `.png` files alongside itself.

## Connection to the Conservation Spectral Framework

These visualizations are the *geometric intuition layer* for the conservation ratio α(G,a) = (a^T L a) / (λ_max ‖a‖²). Every picture is a geometric interpretation of this equation:

- **Rubber sheet**: L applied to f smooths the surface
- **Eigenvalue shells**: λ_max bounds the energy, inner eigenvalues bound conservation
- **Springs**: Dirichlet energy is physical potential energy
- **Phase portrait**: Where different domains cluster in (α, λ₂) space
- **Alignment**: Which graph structures conserve which attribute shapes

## Related Repos

- [conservation-spectral-v2](https://github.com/SuperInstance/conservation-spectral-v2) — Reference Python implementation
- [ecosystem-conservation](https://github.com/SuperInstance/ecosystem-conservation) — Ecological application
- [conservation-art](https://github.com/SuperInstance/conservation-art) — Artistic visualizations
- [code-conservation](https://github.com/SuperInstance/code-conservation) — Software dependency analysis
- [flux-flow-state](https://github.com/SuperInstance/flux-flow-state) — Interactive flow visualization

## License

MIT
