import numpy as np
import matplotlib.pylab as plt

from src.utils import ellipsoid_surface_2D

def plotTrajectoryInTime(X, U, Xref=None, Uref=None, P=None, K=None, title='', ax=None, with_ylabels=True):

    if ax is None:
        fig, ax = plt.subplots(nrows=6, sharex=True, figsize=(4, 7.5))

    if P is not None:
        num_steps = len(P)
        std_0 = np.sqrt([P[k][0, 0] for k in range(num_steps)])
        ax[0].fill_between(np.arange(num_steps), X[0,:] - std_0, X[0,:] + std_0, alpha=0.4, color='C0')
        std_1 = np.sqrt([P[k][1, 1] for k in range(num_steps)])
        ax[0].fill_between(np.arange(num_steps), X[1,:] - std_1, X[1,:] + std_1, alpha=0.4, color='C1')

    ax[0].plot(X[0,:], label=r'$p_x$', color='C0')
    ax[0].plot(X[1,:], label=r'$p_y$', color='C1')

    if P is not None:
        for n in range(2, 5):
            std_0 = np.sqrt([P[k][n, n] for k in range(num_steps)])
            ax[n-1].fill_between(np.arange(num_steps), X[n,:] - std_0, X[n,:] + std_0, alpha=0.4, color='C0')

    ax[1].plot(X[2,:], label=r'$\theta$', color='C0')
    ax[2].plot(X[3,:], label='$v$', color='C0')
    ax[3].plot(X[4,:], label=r'$\omega$', color='C0')

    if Xref is not None:
        ax[0].plot(Xref[0,:], '--', label=r'$p_{x,ref}$', color='C0')
        ax[0].plot(Xref[1,:], '--', label=r'$p_{y,ref}$', color='C1')
        ax[1].plot(Xref[2,:], '--', label=r'$\theta_{ref}$', color='C0')
        ax[2].plot(Xref[3,:], '--', label='$v_{ref}$', color='C0')
        ax[3].plot(Xref[4,:], '--', label=r'$\omega_{ref}$', color='C0')

    if K is not None:
        for i in range(2):
            xvals = np.repeat(np.arange(num_steps), 2)
            xvals = xvals[1:-1]
            std_0 = np.sqrt([(K[k] @ P[k] @ K[k].T)[i, i] for k in range(num_steps-1)])
            lower = np.repeat(U[i,:] - std_0, 2)
            upper = np.repeat(U[i,:] + std_0, 2)
            ax[4+i].fill_between(xvals, lower, upper, alpha=0.4, color=f'C{0}')

    ax[4].stairs(U[0,:], label='$a$', color='C0', linewidth=1.1)
    ax[5].stairs(U[1,:], label=r'$\alpha$', color='C0', linewidth=1.1)

    if Uref is not None:
        ax[4].stairs( Uref[0,:], linestyle='--', label='$a_{ref}$', color='C0')
        ax[5].stairs(Uref[1,:],  linestyle='--', label=r'$\alpha_{ref}$', color='C0')

    if with_ylabels:
        ax[0].set_ylabel('position')
        ax[1].set_ylabel('orientation')
        ax[2].set_ylabel('velocity')
        ax[3].set_ylabel('angular velocity')
        ax[4].set_ylabel('acceleration')
        ax[5].set_ylabel('angular acceleration')

    for i in range(6):
        ax[i].legend(loc='upper left')
        ax[i].grid()

    ax[-1].set_xlabel('discrete time k')
    ax[-1].set_xlim(0, X.shape[1]-1)
    ax[0].set_title(title, fontsize=10)

def plotTrajectoryInSpace(params, X, P=None, Xref=None, title='', ax=None, with_ylabels=True):

    N = X.shape[1] - 1

    theta = np.linspace(0, 2*np.pi, 100)
    c = params.obstacle_center
    R = params.obstacle_radius

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4))

    ax.plot(X[0, :], X[1,:], 'o', color='C1', markersize=4)
    if Xref is not None:
        ax.plot(Xref[0, :], Xref[1,:], 'x', color='C2', markersize=4)

    if P is not None:
        P_surf = [X[:2,k][:,None] + ellipsoid_surface_2D(P[k][:2,:2]) for k in range(N+1)]
        for p in P_surf:
            ax.plot(p[0, :], p[1,:], color='C1', alpha=0.6)

    ax.plot(c[0] + R * np.sin(theta), c[1] + R * np.cos(theta), color='C0')
    ax.axvline(x=params.rxmin, color='C0')
    ax.axhline(y=params.rymin, color='C0')
    ax.set_xlabel(r'$r_x$')

    if with_ylabels:
        ax.set_ylabel(r'$r_y$')

    ax.set_xlim(-2, 10.5)
    ax.set_ylim(-2, 10.5)
    ax.axis('equal')
    ax.set_title(title)
    ax.grid()
