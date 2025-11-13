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
from src.solver_ellipsoid_ocp_optimal_statefb import SolverEllipsoidOcpOptimalStatefb
from src.solver_riccati_zoro import SolverRiccatiZoro, OptionsRiccatiZoro, RiccatiConstraintCost
from src.utils import symmToVec

from setup_model import setup_model, RobotOcpParams
from plot_utils import plotTrajectoryInTime, plotTrajectoryInSpace

RUN_NOMINAL = True
RUN_ZORO = True
RUN_RZORO_HESSBARRIER = True
RUN_SIRO = True
RUN_FEEDBACKOPT_IPOPT = False
nsub = np.sum(np.array([RUN_NOMINAL, RUN_ZORO, RUN_RZORO_HESSBARRIER, RUN_SIRO, RUN_FEEDBACKOPT_IPOPT]))

params_ocp = RobotOcpParams()
ocp = setup_model(params_ocp)
ocp_slacked = slack_the_constraints(ocp, weight_stage=1e4, weight_terminal=1e4)
opts_nlpsol = {'ipopt.print_level': 0, 'print_time': 0}
riccati_use_const_cost = False

# initial states
x0_r = np.array([-2.5, 0, 0, 1.8, 0])  # initial state robot
x0_h = np.array([2.5, .2])  # initial state human
x0 = np.concatenate((x0_r, x0_h))  # initial state full system
P0 = np.zeros((ocp.nx, ocp.nx)) # initial ellipsoid

time_grid = np.linspace(0, params_ocp.T, params_ocp.N+1)

# reference trajectory
v_r_ref = 1.8 # robot reference velocity
p_r_x_ref = x0_r[0] + v_r_ref * time_grid # robot reference x position
x_ref_traj = np.zeros((5, params_ocp.N+1))
x_ref_traj[0, :] = p_r_x_ref
x_ref_traj[3, :] = v_r_ref * np.ones((1, params_ocp.N+1))

# human nominal trajectory
v_x_h_nom = np.array([-1, 0.0]) # nominal human velocity
p_h_x_nom = x0_h[0] + v_x_h_nom[0] * time_grid # human reference x position
p_h_y_nom = x0_h[1] + v_x_h_nom[1] * time_grid # human reference y position
p_h_nom = np.concatenate((p_h_x_nom[None, :], p_h_y_nom[None, :]), axis=0)

# human trajectory (nominal)
p_h_traj = np.tile( v_x_h_nom[:, None], (1, params_ocp.N+1) )  # nominal human velocity
param_traj = np.concatenate((x_ref_traj, p_h_traj), axis=0)

# initial guess
x_init = np.concatenate((x_ref_traj, p_h_nom), axis=0)
u_init = np.zeros((ocp.nu, params_ocp.N))

fig_1, ax_1 = plt.subplots(nrows=7, ncols=nsub, sharex=True, sharey='row', figsize=(11, 10))
if len(ax_1.shape) < 2:
    ax_1 = np.atleast_2d(ax_1).T
fig_2, ax_2 = plt.subplots(nrows=2, ncols=nsub, figsize=(15, 4))
if len(ax_2.shape) < 2:
    ax_2 = np.atleast_2d(ax_2).T
nsp = 0

if RUN_NOMINAL:
    solver_nom = SolverNominalOcp(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    solver_nom.set_initial_guess(x_init, u_init)
    solver_nom.set_parameter_values(param_traj)
    x_nom, u_nom = solver_nom.solve(x0)
    plotTrajectoryInTime(params_ocp, x_nom, u_nom, title='nominal OCP', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_nom, title='nominal OCP', ax=ax_2[:,nsp])
    nsp += 1

if RUN_ZORO:
    opts = OptionsRiccatiZoro(compute_riccati=False, riccati_constraint_cost=RiccatiConstraintCost.NONE, riccati_use_const_cost=False, opts_nlpsol=opts_nlpsol,)
    solver_zoro = SolverRiccatiZoro(ocp_slacked, options=opts)
    solver_zoro.set_initial_guess(x_nom, u_nom)
    solver_zoro.set_parameter_values(param_traj)
    x_zoro, u_zoro, P_zoro, _ = solver_zoro.solve(x0, P0)
    plotTrajectoryInTime(params_ocp, x_zoro, u_zoro, P=P_zoro, title='ZORO', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_zoro, P=P_zoro, title='ZORO', ax=ax_2[:,nsp])
    nsp += 1

if RUN_RZORO_HESSBARRIER:
    opts = OptionsRiccatiZoro(compute_riccati=True, riccati_constraint_cost=RiccatiConstraintCost.HESSIAN_LOG, riccati_use_const_cost=riccati_use_const_cost, opts_nlpsol=opts_nlpsol)
    solver_rzoro_b = SolverRiccatiZoro(ocp_slacked, options=opts)
    solver_rzoro_b.set_initial_guess(x_nom, u_nom)
    solver_rzoro_b.set_parameter_values(param_traj)
    x_rzoro_b, u_rzoro_b, P_rzoro_b, K_rzoro_b = solver_rzoro_b.solve(x0, P0)
    plotTrajectoryInTime(params_ocp, x_rzoro_b, u_rzoro_b, P=P_rzoro_b, K=K_rzoro_b, title='Riccati-ZORO with barrier Hessian', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_rzoro_b, P=P_rzoro_b, title='Riccati-ZORO with barrier Hessian', ax=ax_2[:,nsp])
    nsp += 1

if RUN_SIRO:
    opts = OptionsRiccatiZoro(compute_riccati=True, gradient_correction=True, riccati_constraint_cost=RiccatiConstraintCost.LAGRANGE_MULTIPLIERS, riccati_use_const_cost=riccati_use_const_cost, opts_nlpsol=opts_nlpsol)
    solver_siro = SolverRiccatiZoro(ocp_slacked, options=opts)
    solver_siro.set_initial_guess(x_nom, u_nom)
    solver_siro.set_parameter_values(param_traj)
    x_siro, u_siro, P_siro, K_siro = solver_siro.solve(x0, P0)
    plotTrajectoryInTime(params_ocp, x_siro, u_siro, P=P_siro, K=K_siro, title='SIRO', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_siro, P=P_siro, title='SIRO', ax=ax_2[:,nsp])
    nsp += 1

if RUN_FEEDBACKOPT_IPOPT:
    solver_fbopt = SolverEllipsoidOcpOptimalStatefb(ocp, options=OptionsSolver(opts_nlpsol=opts_nlpsol))
    P_init = np.tile( symmToVec(P0).full(), (1, params_ocp.N+1) )
    solver_fbopt.set_initial_guess(x_nom, u_nom, P=P_init * (params_ocp.N +1))
    solver_fbopt.set_parameter_values(param_traj)
    x_fbopt, u_fbopt, P_fbopt, K_fbopt = solver_fbopt.solve(x0, P0)
    P_fbopt = [P + np.diag(1e-14 * np.ones(ocp.nx)) for P in P_fbopt]  # counteract numerical issues
    plotTrajectoryInTime(params_ocp, x_fbopt, u_fbopt, P=P_fbopt, K=K_fbopt, title='Feedback opt. IPOPT', ax=ax_1[:, nsp])
    plotTrajectoryInSpace(params_ocp, x_fbopt, P=P_fbopt, title='Feedback opt. IPOPT', ax=ax_2[:,nsp])
    nsp += 1

plt.show()
