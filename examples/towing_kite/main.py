import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import numpy as np
import matplotlib.pylab as plt

from src.ocp_description import slack_the_constraints
from src.solver_nominal_ocp import OptionsSolver, SolverNominalOcp
from src.solver_ellipsoid_ocp_fixed_statefb import SolverEllipsoidOcpFixedStatefb
from src.solver_ellipsoid_ocp_optimal_statefb import SolverEllipsoidOcpOptimalStatefb
from src.solver_riccati_zoro import SolverRiccatiZoro, OptionsRiccatiZoro, RiccatiConstraintCost
from src.utils import symmToVec

from setup_model import setup_model, KiteParams
from plot_utils import plotKitePositionInAngleSpace, plotPsiControlThrustOverTime
from src.utils import latexify_plots
latexify_plots(fontsize=9)

RUN_NOMINAL = True
RUN_ZORO = True
RUN_RICCATI_ZORO_CONST_HESS = True
RUN_RICCATI_ZORO_ADAPTIVE_HESS = True
RUN_SIRO = True
SOLVE_FIXEDFB_WITH_IPOPT = True
SOLVE_OPTIMALFB_WITH_IPOPT = False # IPOPT does not converge

params_ocp = KiteParams()
ocp = setup_model(params_ocp)
x0 = params_ocp.x0  # initial state

x_init = np.array([x0]*(params_ocp.N+1)).T
u_init = np.zeros((ocp.nu, params_ocp.N))
P_init = np.array([ symmToVec(1e-2 * np.eye(ocp.nx)).full().flatten() ]*(params_ocp.N+1)).T 
opts_nlpsol = {}
opts_nlpsol["ipopt.print_level"] = 0
opts_nlpsol["print_time"] = 0

backoff_eps = 1e-3

if RUN_NOMINAL:
    solver_nom = SolverNominalOcp(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    solver_nom.set_initial_guess(x_init, u_init)
    x_nom, u_nom = solver_nom.solve(x0)
    plotKitePositionInAngleSpace(params_ocp, x_nom, u_nom, title='nominal OCP')

if RUN_ZORO:
    opts = OptionsRiccatiZoro(compute_riccati=False, riccati_constraint_cost=RiccatiConstraintCost.NONE, backoff_eps=backoff_eps, opts_nlpsol=opts_nlpsol)
    solver_zoro = SolverRiccatiZoro(ocp, options=opts)
    solver_zoro.set_initial_guess(x_nom, u_nom)
    x_zoro, u_zoro, P_zoro, K_zoro = solver_zoro.solve(x0)
    plotKitePositionInAngleSpace(params_ocp, x_zoro, u_zoro, P=P_zoro, title='ZORO')

if RUN_RICCATI_ZORO_CONST_HESS:
    opts = OptionsRiccatiZoro(riccati_constraint_cost=RiccatiConstraintCost.NONE, backoff_eps=backoff_eps, opts_nlpsol=opts_nlpsol)
    solver_rzoro = SolverRiccatiZoro(ocp, options=opts)
    solver_rzoro.set_initial_guess(x_nom, u_nom)
    x_riccati, u_riccati, P_riccati, K_riccati = solver_rzoro.solve(x0)
    plotKitePositionInAngleSpace(params_ocp, x_riccati, u_riccati, P=P_riccati, title='Riccati-ZORO')

if RUN_RICCATI_ZORO_ADAPTIVE_HESS:
    opts = OptionsRiccatiZoro(riccati_constraint_cost=RiccatiConstraintCost.HESSIAN_LOG, riccati_use_const_cost=False, backoff_eps=backoff_eps, opts_nlpsol=opts_nlpsol)
    solver_rzoro_b = SolverRiccatiZoro(ocp, options=opts)
    solver_rzoro_b.set_initial_guess(x_nom, u_nom)
    x_riccati_b, u_riccati_b, P_riccati_b, K_riccati_b = solver_rzoro_b.solve(x0)
    plotKitePositionInAngleSpace(params_ocp, x_riccati_b, u_riccati_b, P=P_riccati_b, title='Riccati-ZORO with barrier Hessian')

if RUN_SIRO:
    opts = OptionsRiccatiZoro(gradient_correction=True, riccati_constraint_cost=RiccatiConstraintCost.LAGRANGE_MULTIPLIERS, riccati_use_const_cost=False, backoff_eps=backoff_eps, opts_nlpsol=opts_nlpsol)
    solver_siro_gc = SolverRiccatiZoro(ocp, options=opts)
    solver_siro_gc.set_initial_guess(x_nom, u_nom)
    x_siro_gc, u_siro_gc, P_siro_gc, K_siro_gc = solver_siro_gc.solve(x0)
    plotKitePositionInAngleSpace(params_ocp, x_siro_gc, u_siro_gc, P=P_siro_gc, title='SIRO with gradient correction')

if SOLVE_FIXEDFB_WITH_IPOPT:
    solver_olr = SolverEllipsoidOcpFixedStatefb(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    solver_olr.set_initial_guess(x_nom, u_nom, P_init, Beta=1e-2, beta_N = 1e-2)
    x_olr, u_olr, P_olr = solver_olr.solve(x0)
    plotKitePositionInAngleSpace(params_ocp, x_olr, u_olr, P=P_olr, title='open loop robust')

if SOLVE_OPTIMALFB_WITH_IPOPT:
    solver_ofr = SolverEllipsoidOcpOptimalStatefb(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    solver_ofr.set_initial_guess(x_nom, u_nom, P_init, Beta=1e-2, beta_N = 1e-2)
    x_ofr, u_ofr, P_ofr = solver_ofr.solve(x0)
    plotKitePositionInAngleSpace(params_ocp, x_ofr, u_ofr, P=P_ofr, title='optimal feedback robust')

if RUN_ZORO and RUN_RICCATI_ZORO_CONST_HESS and RUN_RICCATI_ZORO_ADAPTIVE_HESS and RUN_SIRO:
    fig, ax = plt.subplots(1,4, sharey=True, figsize=(8,2))
    plotKitePositionInAngleSpace(params_ocp, x_zoro, u_zoro, P=P_zoro, ax=ax[0], yaxis_label=True, title=r'ZORO, $K=0$')
    plotKitePositionInAngleSpace(params_ocp, x_riccati, u_riccati, P=P_riccati, ax=ax[1], yaxis_label=False, title='Riccati-ZORO, constant Hess.')
    plotKitePositionInAngleSpace(params_ocp, x_riccati_b, u_riccati_b, P=P_riccati_b, ax=ax[2], yaxis_label=False, title='Riccati-ZORO, adaptive Hess.')
    plotKitePositionInAngleSpace(params_ocp, x_siro_gc, u_siro_gc, P=P_siro_gc, ax=ax[3], yaxis_label=False, title='Optimal feedback')
    for a in ax:
        a.tick_params(axis="y",direction="in",)
        a.tick_params(axis="x",direction="in",)

    plt.tight_layout(w_pad=0)
    folderstr = str(Path(__file__).parent) + '/'
    filestr = 'towing_kite_4.pdf'    
    plt.savefig(folderstr + filestr, dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f'Saved figure to {filestr}')

# plt.show()