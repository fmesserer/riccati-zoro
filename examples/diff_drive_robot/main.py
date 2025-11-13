import dataclasses
import sys
from pathlib import Path
# # Add the root directory to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import numpy as np
import matplotlib.pylab as plt

from src.ocp_description import slack_the_constraints
from src.solver_nominal_ocp import SolverNominalOcp, OptionsSolver
from src.solver_ellipsoid_ocp_fixed_statefb import SolverEllipsoidOcpFixedStatefb
from src.solver_ellipsoid_ocp_optimal_statefb import SolverEllipsoidOcpOptimalStatefb
from src.solver_riccati_zoro import SolverRiccatiZoro, OptionsRiccatiZoro, RiccatiConstraintCost
from src.utils import symmToVec

from setup_robot import setup_robot_ocp, RobotOcpParams
from plot_utils import plotTrajectoryInTime, plotTrajectoryInSpace

RUN_NOMINAL = True
SOLVE_FIXEDFB_0_WITH_IPOPT = False
SOLVE_FIXEDFB_ADHOC_WITH_IPOPT = False
SOLVE_OPTIMALFB_WITH_IPOPT = True
RUN_ZORO_FIXED_FB = False
RUN_RICCATI_ZORO_CONST_HESS = False
RUN_RICCATI_ZORO_ADAPTIVE_HESS = True
RUN_SIRO = True

nsub = np.sum(np.array([SOLVE_FIXEDFB_0_WITH_IPOPT, SOLVE_FIXEDFB_ADHOC_WITH_IPOPT, SOLVE_OPTIMALFB_WITH_IPOPT, RUN_ZORO_FIXED_FB, RUN_RICCATI_ZORO_CONST_HESS, RUN_RICCATI_ZORO_ADAPTIVE_HESS, RUN_SIRO]))

params_ocp = RobotOcpParams()
ocp = setup_robot_ocp(params_ocp)
ocp_slacked = slack_the_constraints(ocp)
opts_nlpsol = {'ipopt.print_level': 0, 'print_time': 0}

x0 = np.array([9, 5, 8*np.pi/7, 0, 0])  # initial state
P0 = np.zeros((ocp.nx, ocp.nx)) # initial ellipsoid
par_ref = np.array([0, 0])
Par_prob = np.tile( par_ref[:, None], (1, params_ocp.N+1) )

# initial guess
x_init = np.array([x0]*(params_ocp.N+1)).T
u_init = np.zeros((ocp.nu, params_ocp.N))

fig_1, ax_1 = plt.subplots(nrows=6, ncols=nsub, sharex=True, sharey='row', figsize=(11, 10))
fig_2, ax_2 = plt.subplots(nrows=1, ncols=nsub, figsize=(15, 4))
if len(ax_1.shape) < 2:
    ax_1 = np.atleast_2d(ax_1).T
ax_2 = np.atleast_1d(ax_2)
nsp = 0

if RUN_NOMINAL:
    solver_nom = SolverNominalOcp(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    solver_nom.set_initial_guess(x_init, u_init)
    solver_nom.set_parameter_values(Par_prob)
    x_nom, u_nom = solver_nom.solve(x0)
    # plotTrajectoryInTime(x_nom, u_nom, title='nominal OCP', ax=ax_1[:, nsp])
    # plotTrajectoryInSpace(params_ocp, x_nom, title='nominal OCP', ax=ax_2[nsp])
    # nsp += 1

if SOLVE_FIXEDFB_0_WITH_IPOPT:
    solver_olr = SolverEllipsoidOcpFixedStatefb(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    P_init = np.tile( symmToVec(P0).full(), (1, params_ocp.N+1) )
    solver_olr.set_initial_guess(x_init, u_init, P_init)
    solver_olr.set_parameter_values(Par_prob)
    x_olr, u_olr, P_olr = solver_olr.solve(x0, P0)
    plotTrajectoryInTime(x_olr, u_olr, P=P_olr, title='ellipsoids, open loop', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_olr, P=P_olr, title='ellipsoids, open loop', ax=ax_2[nsp])
    nsp += 1

if SOLVE_FIXEDFB_ADHOC_WITH_IPOPT:
    ocp_fb = dataclasses.replace(ocp, state_feedback_gain=params_ocp.K_fb)
    solver_clf = SolverEllipsoidOcpFixedStatefb(ocp_fb, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    P_init = np.tile( symmToVec(P0).full(), (1, params_ocp.N+1) )
    solver_clf.set_initial_guess(x_init, u_init, P_init)
    solver_clf.set_parameter_values(Par_prob)
    x_cfr, u_cfr, P_cfr = solver_clf.solve(x0, P0)
    plotTrajectoryInTime(x_cfr, u_cfr, P=P_cfr, K=[ocp_fb.state_feedback_gain] * (params_ocp.N+1), title='ellipsoids, heuristic feedback', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_cfr, P=P_cfr, title='ellipsoids, heuristic feedback', ax=ax_2[nsp])
    nsp += 1

if SOLVE_OPTIMALFB_WITH_IPOPT:
    solver_clo = SolverEllipsoidOcpOptimalStatefb(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    P_init = np.tile( symmToVec(P0).full(), (1, params_ocp.N+1) )
    solver_clo.set_initial_guess(x_init, u_init, P_init)
    solver_clo.set_parameter_values(Par_prob)
    x_clo, u_clo, P_clo, K_clo = solver_clo.solve(x0, P0)
    plotTrajectoryInTime(x_clo, u_clo, P=P_clo, K=K_clo, title='ellipsoids, optimal feedback', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_clo, P=P_clo, title='ellipsoids, optimal feedback', ax=ax_2[nsp])
    nsp += 1

if RUN_ZORO_FIXED_FB:
    opts = OptionsRiccatiZoro(opts_nlpsol=opts_nlpsol, compute_riccati=False, riccati_constraint_cost=RiccatiConstraintCost.NONE, riccati_use_const_cost=False)
    solver_zoro = SolverRiccatiZoro(ocp, options=opts)
    solver_zoro.set_initial_guess(x_nom, u_nom)
    solver_zoro.set_parameter_values(Par_prob)
    x_zoro, u_zoro, P_zoro, _ = solver_zoro.solve(x0, P0)
    plotTrajectoryInTime(x_zoro, u_zoro, P=P_zoro, title='ZORO', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_zoro, P=P_zoro, title='ZORO', ax=ax_2[nsp])
    nsp += 1

if RUN_RICCATI_ZORO_CONST_HESS:
    opts = OptionsRiccatiZoro(opts_nlpsol=opts_nlpsol, compute_riccati=True, riccati_constraint_cost=RiccatiConstraintCost.NONE, riccati_use_const_cost=True)
    solver_riccati_zoro = SolverRiccatiZoro(ocp, options=opts)
    solver_riccati_zoro.set_initial_guess(x_nom, u_nom)
    solver_riccati_zoro.set_parameter_values(Par_prob)
    x_rzoro, u_rzoro, P_rzoro, K_rzoro = solver_riccati_zoro.solve(x0, P0)
    plotTrajectoryInTime(x_rzoro, u_rzoro, P=P_rzoro, K=K_rzoro, title='Riccati-ZORO', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_rzoro, P=P_rzoro, title='Riccati-ZORO', ax=ax_2[nsp])
    nsp += 1

if RUN_RICCATI_ZORO_ADAPTIVE_HESS:
    opts = OptionsRiccatiZoro(opts_nlpsol=opts_nlpsol, compute_riccati=True, riccati_constraint_cost=RiccatiConstraintCost.HESSIAN_LOG, riccati_use_const_cost=False)
    solver_rzoro_b = SolverRiccatiZoro(ocp_slacked, options=opts)
    solver_rzoro_b.set_initial_guess(x_nom, u_nom)
    solver_rzoro_b.set_parameter_values(Par_prob)
    x_rzoro_b, u_rzoro_b, P_rzoro_b, K_rzoro_b = solver_rzoro_b.solve(x0, P0)
    plotTrajectoryInTime(x_rzoro_b, u_rzoro_b, P=P_rzoro_b, K=K_rzoro_b, title='Riccati-ZORO with barrier Hessian', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_rzoro_b, P=P_rzoro_b, title='Riccati-ZORO with barrier Hessian', ax=ax_2[nsp])
    nsp += 1

if RUN_SIRO:
    opts = OptionsRiccatiZoro(gradient_correction=True, compute_riccati=True, opts_nlpsol=opts_nlpsol, riccati_constraint_cost=RiccatiConstraintCost.LAGRANGE_MULTIPLIERS, riccati_use_const_cost=False)
    solver_siro_gc = SolverRiccatiZoro(ocp_slacked, options=opts)
    solver_siro_gc.set_initial_guess(x_nom, u_nom)
    solver_siro_gc.set_parameter_values(Par_prob)
    x_siro_gc, u_siro_gc, P_siro_gc, K_siro_gc = solver_siro_gc.solve(x0, P0)
    plotTrajectoryInTime(x_siro_gc, u_siro_gc, P=P_siro_gc, K=K_siro_gc, title='SIRO with gradient correction', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_siro_gc, P=P_siro_gc, title='SIRO with gradient correction', ax=ax_2[nsp])
    nsp += 1

plt.show()
