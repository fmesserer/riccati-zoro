from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
import casadi as ca
from src.ocp_description import OcpDescription
from src.utils import integrate_RK4


@dataclass
class RobotOcpParams:
    T: float = 3.5                        # cont time horizon
    N: int = 10                         # time steps
    umax: Tuple[float, float] = (2., 1.5)      # control box constr, max
    rxmin: float = -0.5                 # minimal x position
    rymin: float = -0.5                 # minimal y position
    obstacle_center: Tuple[float, float] = (4., 4.)
    obstacle_radius: float = 3.
    std_w: Tuple[float, float] = (0.1, 0.1)          # std of process noise
    k_fb: float = -5                  # feedback gain for tube robust OCP
    p_ref: Tuple[float, float] = (0, 0)

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def K_fb(self) -> np.ndarray:
        return self.k_fb * np.array([[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])


def setup_robot_ocp(params: RobotOcpParams) -> OcpDescription:
    # states
    px = ca.SX.sym("px")    # position x
    py = ca.SX.sym("py")    # position y
    beta = ca.SX.sym("beta")  # heading angle
    v = ca.SX.sym("v")      # velocity
    omega = ca.SX.sym("omega")  # angular velocity
    x = ca.vertcat(px, py, beta, v, omega)

    # controls
    a = ca.SX.sym("a")    # acceleration
    alpha = ca.SX.sym("alpha")  # angular acceleration
    u = ca.vertcat(a, alpha)

    # noise
    w = ca.SX.sym("w", 2)  # process noise

    # parameters
    p_ref = ca.SX.sym("p_ref", 2)  # reference position
    par = ca.veccat(p_ref)

    # dynamics
    px_dot = v*ca.cos(beta)
    py_dot = v*ca.sin(beta)
    beta_dot = omega
    v_dot = a + params.std_w[0] * w[0]
    omega_dot = alpha + params.std_w[1] * w[1]

    xdot = ca.vertcat(px_dot, py_dot, beta_dot, v_dot, omega_dot)
    f_discrete = integrate_RK4(xdot=xdot, dt=params.dt, x=x, u=u, w=w)

    # cost
    R = np.diag([2, 1])
    eps = 1
    delta_p = x[:2] - p_ref
    cost_stage = 100*ca.sqrt(delta_p.T @ delta_p + eps) + u.T @ R @ u
    cost_terminal = 100 * ca.sqrt(delta_p.T @ delta_p + eps) + 10*x[3]**2

    # constraints
    obs_r = params.obstacle_radius
    obs_c = params.obstacle_center
    constr_obst = obs_r - ca.sqrt((x[0] - obs_c[0])**2 + (x[1] - obs_c[1])**2 + 1e-4)
    constr_pxmin = params.rxmin - x[0]
    constr_pymin = params.rymin - x[1] 
    constr_umin = -u - ca.vertcat(*params.umax)
    constr_umax = u - ca.vertcat(*params.umax)

    constr_stage = ca.vertcat(constr_obst, constr_pxmin, constr_pymin, constr_umax, constr_umin)
    constr_terminal = ca.vertcat(constr_obst, constr_pxmin, constr_pymin)

    # uncertainty cost hessian
    Q_ricc_k = np.eye(x.shape[0])
    R_ricc_k = np.eye(u.shape[0])
    Q_ricc_N = np.eye(x.shape[0])

    ocp = OcpDescription(
        x = x,
        u = u,
        w = w,
        par = par,
        T = params.T,
        N = params.N,
        dyn_discr_expr = f_discrete,
        cost_stage_expr = cost_stage,
        cost_terminal_expr = cost_terminal,
        constr_stage_expr = constr_stage,
        constr_terminal_expr = constr_terminal,
        cost_uncertainty_Q_stage = Q_ricc_k,
        cost_uncertainty_R = R_ricc_k,
        cost_uncertainty_Q_terminal = Q_ricc_N,
    )

    return ocp
