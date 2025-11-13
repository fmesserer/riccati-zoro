import numpy as np
import matplotlib.pylab as plt
import casadi as ca

from setup_model import RobotOcpParams
from src.utils import ellipsoid_surface_2D

def plotTrajectoryInTime(params: RobotOcpParams, X, U, Xref=None, P=None, K=None, title='', ax=None, with_ylabels=True):

    N = X.shape[1] - 1

    if ax is None:
        fig, ax = plt.subplots(nrows=7, sharex=True, figsize=(4, 7.5))

    # robot states
    ax[0].plot(X[0,:], label=r'robot', color='C0')
    ax[1].plot(X[1,:], label=r'robot', color='C0')
    ax[2].plot(X[2,:], label=r'$\theta$', color='C0')
    ax[3].plot(X[3,:], label='$v$', color='C0')
    ax[4].plot(X[4,:], label=r'$\omega$', color='C0')
    # controls
    ax[5].stairs(U[0,:], label='$a$', color='C0', linewidth=1.1)
    ax[6].stairs(U[1,:], label=r'$\alpha$', color='C0', linewidth=1.1)
    # human states
    ax[0].plot(X[5,:], label=r'human', color='C2')
    ax[1].plot(X[6,:], label=r'human', color='C2')


    if P is not None:
        # plot robot state uncertainty
        for i in range(5):
            std_0 = np.sqrt([Pk[i, i] for Pk in P])
            ax[i].fill_between(np.arange(N+1), X[i,:] - std_0, X[i,:] + std_0, alpha=0.4, color='C0')
        # plot human state uncertainty
        for i in range(2):
            std_0 = np.sqrt([Pk[5+i, 5+i] for Pk in P])
            ax[i].fill_between(np.arange(N+1), X[5+i,:] - std_0, X[5+i,:] + std_0, alpha=0.4, color='C2')

    # reference trajectory
    if Xref is not None:
        ax[0].plot(Xref[0,:], '--', label=r'ref', color='C0')
        ax[1].plot(Xref[1,:], '--', label=r'ref', color='C0')
        ax[2].plot(Xref[2,:], '--', label=r'ref', color='C0')
        ax[3].plot(Xref[3,:], '--', label=r'ref', color='C0')
        ax[4].plot(Xref[4,:], '--', label=r'ref', color='C0')


    if K is not None:
        for i in range(2):
            xvals = np.repeat(np.arange(N+1), 2)
            xvals = xvals[1:-1]
            std_0 = np.sqrt([(K[k] @ P[k] @ K[k].T)[i, i] for k in range(N)])
            lower = np.repeat(U[i,:] - std_0, 2)
            upper = np.repeat(U[i,:] + std_0, 2)
            ax[5+i].fill_between(xvals, lower, upper, alpha=0.4, color='C0')

    # constraints
    ax[1].axhline(y=params.p_y_min + params.radius_robot, color='k', linestyle='--')
    ax[1].axhline(y=params.p_y_max - params.radius_robot, color='k', linestyle='--')
    ax[3].axhline(y=params.v_r_min, color='k', linestyle='--')
    ax[3].axhline(y=params.v_r_max, color='k', linestyle='--')
    ax[4].axhline(y=params.omega_min, color='k', linestyle='--')
    ax[4].axhline(y=params.omega_max, color='k', linestyle='--')


    # hand set limits
    ax[0].set_ylim(-2.2, 2)
    ax[1].set_ylim(params.p_y_min - .2, params.p_y_max + .2)
    ax[2].set_ylim(-1.5, 1.5)
    ax[3].set_ylim(params.v_r_min - 0.2, params.v_r_max + 0.2)
    ax[4].set_ylim(params.omega_min - 0.2, params.omega_max + 0.2)
    ax[5].set_ylim(-.5, .5)
    ax[6].set_ylim(-2,2)

    if with_ylabels:
        ax[0].set_ylabel(r'position $p_\mathrm{x}$')
        ax[1].set_ylabel(r'position $p_\mathrm{y}$')
        ax[2].set_ylabel(r'orientation $\theta$')
        ax[3].set_ylabel(r'velocity $v$')
        ax[4].set_ylabel(r'ang. velocity $\omega$')
        ax[5].set_ylabel(r'acc. $a$')
        ax[6].set_ylabel(r'ang. acc. $\alpha$')

    ax[0].legend(loc='upper left')
    for i in range(7):
        # ax[i].legend(loc='upper left')
        ax[i].grid()

    ax[-1].set_xlabel('discrete time k')
    ax[-1].set_xlim(0, X.shape[1]-1)
    ax[0].set_title(title, fontsize=10)


def plotTrajectoryInSpace(params: RobotOcpParams, X, P=None, Xref=None, title='', ax=None, with_ylabels=True):

    N = X.shape[1] - 1

    idx_plotunc = range(0, N+1, 5)

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(4, 4))
    #############
    # position plot
    #############
    ax[0].plot(X[0, :], X[1,:], '--', color='C0', markersize=4, label='robot')
    ax[0].plot(X[5, :], X[6,:], '--', color='C2', markersize=4, label='human')

    # robot body
    robot_circle = params.radius_robot * ellipsoid_surface_2D(np.eye(2))
    p_r_surf = [X[:2,k][:,None] +  robot_circle for k in idx_plotunc]
    for p in p_r_surf:
        ax[0].fill(p[0, :], p[1,:], color='C0', alpha=1, edgecolor=None)
    
    # human body
    human_circle = params.radius_human * ellipsoid_surface_2D(np.eye(2))
    p_h_surf = [X[5:7,k][:,None] +  human_circle for k in idx_plotunc]
    for p in p_h_surf:
        ax[0].fill(p[0, :], p[1,:], color='C2', alpha=1, edgecolor=None)

    if P is not None:
        # robot body uncertainty (circle plus ellipsoids)
        p_r_unc = [X[:2,k][:,None] + robot_circle + ellipsoid_surface_2D(P[k][:2,:2]) for k in idx_plotunc]
        for p in p_r_unc:
            ax[0].fill(p[0, :], p[1,:], color='C0', alpha=0.4, edgecolor=None)
        # human body uncertainty (circle plus ellipsoids)
        p_h_unc = [X[5:7,k][:,None] + human_circle + ellipsoid_surface_2D(P[k][5:7,5:7]) for k in idx_plotunc]
        for p in p_h_unc:
            ax[0].fill(p[0, :], p[1,:], color='C2', alpha=0.4, edgecolor=None)

    if Xref is not None:
        ax[0].plot(Xref[0, :], Xref[1,:], linestyle='--', color='C0')

    # ax.axvline(x=params.rxmin, color='C0')
    ax[0].axhline(y=params.p_y_min, color='k', linestyle='--')
    ax[0].axhline(y=params.p_y_max, color='k', linestyle='--')
    ax[0].set_xlabel(r'$p_\mathrm{x}$')

    if with_ylabels:
        ax[0].set_ylabel(r'$p_\mathrm{y}$')

    ax[0].axis('equal')
    ax[0].set_title(title)
    ax[0].grid()
    ax[0].legend()

    #############
    # distance plot
    #############
    radius_sum = params.radius_robot + params.radius_human
    x = ca.SX.sym('x', X.shape[0])
    x_r = x[:5]
    x_h = x[5:7]
    dist_exp = ca.norm_2(x_r[:2] - x_h) - radius_sum  # collision avoidance constraint
    dist_grad = ca.gradient(dist_exp, x)
    dist_fun = ca.Function('dist_fun', [x], [dist_exp])
    dist_grad_fun = ca.Function('dist_grad_fun', [x], [dist_grad])
    dist_list = []
    backoff_list = []
    for k in range(N+1):
        dist_k = dist_fun(X[:, k]).full().flatten()
        dist_grad_k = dist_grad_fun(X[:, k]).full().flatten()
        backoff_k = np.sqrt(dist_grad_k.T @ P[k] @ dist_grad_k) if P is not None else 0.0
        dist_list.append(dist_k)
        backoff_list.append(backoff_k)

    dist = np.array(dist_list).flatten()
    backoff = np.array(backoff_list).flatten()

    tvals = np.arange(N+1)

    ax[1].plot(tvals, dist, color='C0')
    if P is not None:
        ax[1].fill_between(tvals, dist - backoff, dist + backoff, alpha=0.4, color='C0')
    ax[1].axhline(y=0.0, color='k', linestyle='--')
    ax[1].set_ylabel('distance robot / human')
