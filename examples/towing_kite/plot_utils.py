from typing import List
import numpy as np
import matplotlib.pylab as plt

from setup_model import KiteParams, rad_per_deg
from src.utils import ellipsoid_surface_2D


def plotKitePositionInAngleSpace(params: KiteParams, X: np.ndarray, U: np.ndarray, P:List[np.ndarray]=None, title='', ax=None, yaxis_label=True):
    """
    plots kite position in angle (i.e. phi-theta) space.
    X[0, :] contains the time series of theta (in rad)
    X[1, :] contains the time series of phi (in rad)
    P = list of uncertainty ellipsoids
    """

    hmin = params.hmin
    L = params.L
    N = X.shape[1] - 1
    theta = X[0,:]
    phi = X[1,:]

    if P is not None:
        P_surf = [ X[:2,k][:,None] / rad_per_deg + ellipsoid_surface_2D(P[k][:2,:2]) / rad_per_deg for k in range(N+1) ]
    
    # obtained thrust
    TF = [params.thrust_force(theta=theta[k], phi=phi[k], u=U[:, k]) / 1000 for k in range(N-1) ]
    TFavg = np.mean(TF)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    phi_constr = np.linspace(-1, 1, 100) * 70 * rad_per_deg
    theta_constr = np.arcsin(hmin / L / np.cos(phi_constr ))
    ax.plot(phi_constr / rad_per_deg, theta_constr / rad_per_deg, '-', color='C2', label=r'height constraint')

    ax.plot(phi[0] / rad_per_deg, theta[0] / rad_per_deg, 'x', color='C0', label='initial state')
    if P is not None:
        for p in P_surf:
            ax.plot(p[1,:], p[0,:], color='C0', lw=1)
            # plt.plot(p[1,:], p[0,:], 'b', label='trajectory')
            ax.fill(p[1,:], p[0,:], color='C0')#, alpha=.4 )
    else:
        ax.plot( phi / rad_per_deg, theta / rad_per_deg, '.', color='C0', label='trajectory', ms=4)

    ylim = list(ax.get_ylim())
    ylim[0] = 12
    ylim[1] = 45
    ax.set_ylim(ylim)

    tf_avg_str = r'$\hat T_\mathrm{F} =' + '{:.3f}'.format(TFavg) + r'\;\mathrm{kN}$'
    ax.text(0, 41, tf_avg_str, horizontalalignment='center')

    # plt.legend()
    # if title != '':
    #     title += r'$,\quad$'
    # title += tf_avg_str

    ax.set_xlabel(r'azimuth angle $\phi$ in deg')
    if yaxis_label:
        ax.set_ylabel(r"zenith angle $\theta$ in deg")
    ax.set_title(title)


def plotPsiControlThrustOverTime(params: KiteParams, X: np.ndarray, U: np.ndarray, title=''):

    DT = params.dt
    theta = X[0, :]
    phi = X[1, :]
    psi = X[2, :]
    N = psi.size
    T = np.arange(N) * DT

    # compute thrust
    TF = [params.thrust_force(theta=theta[k], phi=phi[k], u=U[:, k])[0] / 1000 for k in range(N-1) ]
    # breakpoint()

    plt.figure()
    ax = plt.subplot(3,1,1)
    plt.plot(T, psi / rad_per_deg)
    plt.ylabel(r'$\psi$ in deg')
    ax.set_xticklabels([])

    ax = plt.subplot(3,1,2)
    plt.step(T, np.concatenate((U.flatten(), [np.nan])))
    plt.plot([T[0], T[-1]], [params.umin, params.umin], 'r--')
    plt.plot([T[0], T[-1]], [params.umax, params.umax], 'r--')
    plt.ylabel(r'control $u$')
    ax.set_xticklabels([])

    plt.subplot(3,1,3)
    plt.step(T, TF +  [np.nan])
    plt.ylabel(r'thrust $T_F$ in kN')
    plt.xlabel(r'time $t$ in s')
    plt.title(title)


# def plotAngleSpaceCompareNom(params, Xlist):

#     hmin = params['hmin']
#     L = params['L']
#     N = Xlist[0].shape[1] - 1
    
#     plt.figure()
#     for X in Xlist:
#         theta = X[0,:] / rad_per_deg
#         phi = X[1,:] / rad_per_deg
#         plt.plot( phi, theta)

#     phi_constr = np.linspace(-1, 1, 100) * 70 
#     theta_constr = np.arcsin( hmin / L / np.cos(phi_constr * rad_per_deg) ) / rad_per_deg
#     plt.plot( phi_constr, theta_constr, 'r', label=r'$h_{\min}$ constraint')
#     plt.legend()
#     plt.xlabel(r'$\phi$ in  deg')
#     plt.ylabel(r"$\theta$ in  deg")

