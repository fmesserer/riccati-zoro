import shutil
from typing import Union
import matplotlib
import numpy as np
import casadi as ca
import numpy as np


def integrate_RK4(xdot: ca.SX, dt:float, x: ca.SX, u: ca.SX, w:ca.SX=ca.SX(), p:ca.SX=ca.SX(), n_steps:int=1):
    h = dt / n_steps

    xdot_fun = ca.Function('xdot', [x, u, w, p], [xdot])
    x_end = x + 0

    for _ in range(n_steps):
        k_1 = xdot_fun(x_end, u, w, p)
        k_2 = xdot_fun(x_end + 0.5 * h * k_1, u, w, p)
        k_3 = xdot_fun(x_end + 0.5 * h * k_2, u, w, p)
        k_4 = xdot_fun(x_end + k_3 * h, u, w, p)

        x_end = x_end + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
    return x_end


def sampleFromEllipsoid(P:np.ndarray, n:int=100):
    """
    draws samples from ellipsoid defined by matrix P
    """
    n_P = P.shape[0]                  # dimension

    # sample in hypersphere
    r = np.random.rand(n)**(1/n_P)     # radial position of sample
    x = np.random.randn(n_P, n)

    x = r*(x/np.linalg.norm(x, axis=0))
    # project to ellipsoid
    lam, v = np.linalg.eig(P)
    y = v @ (np.diag(np.sqrt(lam)) @ x)
    return y


def ellipsoid_surface_2D(P:np.ndarray, n:int=100):
    
    phi = np.linspace(0, 2 * np.pi, n)
    c = np.vstack([np.cos(phi), np.sin(phi)]) # points on unit circle / vectors of all directions
    # lam, V = np.linalg.eig(P)
    # a = (V @ np.diag(np.sqrt(lam))) @ np.vstack([np.cos(phi), np.sin(phi)])

    v = np.zeros((2, n))
    for i in range(n):
        dir = c[:,i]
        # support function argmax in direction [c:,i]
        v[:, i] = (P @ dir) / np.sqrt(dir.T @ P @ dir + 1e-8)
    return v


def vecToSymm(Pvec, nx):
    """
    in: vector encoding symmetrix matrix (its entries)
    out: corresponding symmetric matrix
    """
    if isinstance(Pvec, ca.SX):
        P_symm = ca.tril2symm(ca.SX(ca.Sparsity.lower(nx), Pvec))
    elif isinstance(Pvec, ca.MX):
        P_symm = ca.tril2symm(ca.MX(ca.Sparsity.lower(nx), Pvec))
    elif isinstance(Pvec, ca.DM):
        P_symm = ca.tril2symm(ca.DM(ca.Sparsity.lower(nx), Pvec))
    elif isinstance(Pvec, np.ndarray):
        P_symm = ca.DM(ca.tril2symm(ca.SX(ca.Sparsity.lower(nx), Pvec))).full()
    else:
        Exception("Unsupported type in vecToSymm.")

    return P_symm


def symmToVec(P: Union[ca.SX, np.ndarray]) -> ca.DM:
    """
    in: symmetrix matrix
    out: vector encoding of its entries
    """
    if isinstance(P, np.ndarray):
        P = ca.DM(P)

    return P[P.sparsity().makeDense()[0].get_lower()]


def latexify_plots(fontsize:float=12) -> None:
    text_usetex = True if shutil.which('latex') else False
    params = {
            'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath, amssymb}",
            'font.size': fontsize,
            # 'axes.labelsize': fontsize,
            'axes.titlesize': fontsize,
            # 'legend.fontsize': fontsize,
            # 'xtick.labelsize': fontsize,
            # 'ytick.labelsize': fontsize,
            'text.usetex': text_usetex,
            'font.family': 'serif',
    }

    matplotlib.rcParams.update(params)
    return
