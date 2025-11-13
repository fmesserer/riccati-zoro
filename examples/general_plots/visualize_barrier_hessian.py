import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D
import casadi as ca 

from src.utils import latexify_plots, ellipsoid_surface_2D

latexify_plots(fontsize=8)
outfolder = str(Path(__file__).parent)

golden_ratio = (1 + 5 ** 0.5) / 2
plt.figure(figsize=(3.5, 3.5 / golden_ratio))

# constraint function
z = ca.SX.sym('z', 2)
constr = []
constr.append(1 - z[0] - z[1])
constr.append(-1 + z[0] - z[1])
constr_func = ca.Function('constr_func', [z], [ca.vertcat(*constr)])

# resulting barrier Hessian
barrier_hess_expr = 0
for c in constr:
    grad_c = ca.gradient(c, z)
    barrier_hess_expr += (1 / (c**2)) * (grad_c @ grad_c.T)
barrier_hess_fun = ca.Function('barrier_hess', [z], [barrier_hess_expr])

# example points to visualize barrier Hessian
hess_points = []
hess_points.append(np.array([[1, .3]]).T)
eps = 0.1
hess_points.append(np.array([[2.5-eps, 1.5 + eps]]).T)
hess_points.append(np.array([[0, 2.5]]).T)

colors = ['C0', 'C1', 'C2']
lines = []

for i, z in enumerate(hess_points):
    C = barrier_hess_fun(z).full()                  # Hessian matrix
    Q = np.linalg.inv(C)                            # ellipsoid matrix for corresponding level lines
    Q_surf = ellipsoid_surface_2D(Q, n=100)         # level line ellipsoid centered at origin
    for j in np.linspace(0.01, 1.1 , 6):
        Q_surf_z = z + np.sqrt(j) *  Q_surf                  # shift to center at z   
        plt.plot(Q_surf_z[0, :], Q_surf_z[1, :], linestyle='-', color=colors[i], lw=.6)
    label = fr'$\tilde l(\tilde z; z_{{{i+1}}})$'
    lines += [Line2D([0], [0], color=colors[i], linewidth=.6, linestyle='-', label=label)]

plt.legend(handles = lines, handlelength=1)
plt.text(1.9, 2.65, r'$\tilde z \in \mathbb{R}^2$')

xmin = -1
xmax = 5
# constraint boundaries
y_xmin = constr_func(ca.vertcat(xmin, 0)).full().flatten()
y_xmax = constr_func(ca.vertcat(xmax, 0)).full().flatten()

infeasible_1_x = [xmin, xmax, xmin]
infeasible_1_y = [y_xmin[0], y_xmax[0], -5]
plt.fill(infeasible_1_x, infeasible_1_y, color='lightgray', edgecolor='none')

infeasible_2_x = [xmin, xmax, xmax]
infeasible_2_y = [y_xmin[1], y_xmax[1], -5]
plt.fill(infeasible_2_x, infeasible_2_y, color='lightgray', edgecolor='none')


plt.gca().set_aspect('equal', 'box')
plt.gca().set_xlim([-1, 4])
plt.gca().set_ylim([-.1, 2.9])
# remove ticks
plt.gca().set_xticks([])
plt.gca().set_yticks([])
filename_str = 'barrier_hessian.pdf'
plt.savefig(Path(outfolder) / filename_str, dpi=300, bbox_inches='tight', pad_inches=0.03)
print(f'Saved figure to {filename_str}')
# plt.show()