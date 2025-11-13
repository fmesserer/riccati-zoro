from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
import casadi as ca
from src.ocp_description import OcpDescription
from src.utils import integrate_RK4


@dataclass
class RobotOcpParams:
    T: float = 2 # cont time horizon
    N: int = 20 # time steps
    p_y_max: float = 1.3 # maximal y position
    p_y_min: float = -1.3 # minimal y position
    v_r_min: float = 0 # minimal robot velocity
    v_r_max: float = 2.0 # maximal robot velocity
    omega_min: float = -1.0 # minimal robot angular velocity
    omega_max: float = 1.0 # maximal robot angular velocity
    radius_robot = 0.15
    radius_human = 0.15
    std_v_h: Tuple[float, float] = (4*0.4, 4*0.4) # std of process noise (human velocity)
    k_fb: float = -5 # feedback gain for tube robust OCP

    @property
    def dt(self) -> float:
        return self.T / self.N

    @property
    def K_fb(self) -> np.ndarray:
        return self.k_fb * np.array([[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]])


def setup_model(params: RobotOcpParams) -> OcpDescription:
    # states
    # robot
    p_r_x = ca.SX.sym("p_r_x") # position x
    p_r_y = ca.SX.sym("p_r_y") # position y
    beta = ca.SX.sym("beta")  # heading angle
    v_r = ca.SX.sym("v_r")      # velocity
    omega = ca.SX.sym("omega")  # angular velocity
    x_r = ca.veccat(p_r_x, p_r_y, beta, v_r, omega)
    # human
    p_h = ca.SX.sym("p_h", 2)  # position human
    x_h = ca.veccat(p_h)
    x = ca.veccat(x_r, x_h)

    # controls
    a = ca.SX.sym("a")    # acceleration
    alpha = ca.SX.sym("alpha")  # angular acceleration
    u = ca.veccat(a, alpha)

    # noise
    delta_v_h = ca.SX.sym("delta_v_h", 2) # disturbance human velocity
    w = ca.veccat(delta_v_h) # process noise

    # parameters
    x_r_ref = ca.SX.sym("x_r_ref", x_r.shape[0])  # reference state robot
    v_h_nom = ca.SX.sym("v_h_nom", 2)  # predicted nominal human velocity
    par = ca.veccat(x_r_ref, v_h_nom)

    # dynamics
    # robot
    p_r_x_dot = v_r * ca.cos(beta)
    p_r_y_dot = v_r * ca.sin(beta)
    beta_dot = omega
    v_dot = a
    omega_dot = alpha
    x_r_dot = ca.veccat(p_r_x_dot, p_r_y_dot, beta_dot, v_dot, omega_dot)

    # human
    p_h_x_dot = v_h_nom[0] + params.std_v_h[0] * w[0]
    p_h_y_dot = v_h_nom[1] + params.std_v_h[1] * w[1]
    x_h_dot = ca.veccat(p_h_x_dot, p_h_y_dot)
    
    # continuous time dynamics
    xdot = ca.veccat(x_r_dot, x_h_dot)
    # discrete time dynamics
    f_discrete = integrate_RK4(xdot=xdot, dt=params.dt, x=x, u=u, w=w,p=par)

    # cost
    Q = np.diag([50, 50, 0, 2, 0, 0, 0])
    R = np.diag([2, 2])
    x_h_ref = ca.SX(x_h.shape[0], 1) # dummy entry, weighted by zero anyway
    x_ref = ca.veccat(x_r_ref, x_h_ref)
    cost_stage = (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u
    # cost_stage +=  100 * ca.sqrt(x[1]**2 + 1e-2)
    cost_terminal = (x - x_ref).T @ Q @ (x - x_ref)

    # constraints
    radius_sum = params.radius_robot + params.radius_human
    constr_collision = radius_sum - ca.norm_2(x_r[:2] - x_h)  # collision avoidance constraint
    constr_pymin = params.p_y_min + params.radius_robot - x[1] 
    constr_pymax = x[1]  - (params.p_y_max - params.radius_robot)
    constr_vrmin = params.v_r_min - x[3]
    constr_vrmax = x[3] - params.v_r_max
    constr_omegamin = params.omega_min - x[4]
    constr_omegamax = x[4] - params.omega_max

    # constr_state = [constr_collision, constr_pymin, constr_pymax, constr_vrmin, constr_vrmax]
    constr_state = [constr_collision, constr_pymin, constr_pymax, constr_vrmin, constr_vrmax, constr_omegamin, constr_omegamax]
    constr_stage = ca.veccat(*constr_state)
    constr_terminal = ca.veccat(*constr_state)

    # weight constraint uncertainty
    weight_stage_constr = np.ones(constr_stage.shape[0])
    weight_stage_constr[0] = 10.0  # collision avoidance
    weight_terminal_constr = np.ones(constr_terminal.shape[0])
    weight_terminal_constr[0] = 10.0  # collision avoidance

    # uncertainty cost hessian
    Q_ricc_k = Q.copy()
    R_ricc_k = R.copy()
    Q_ricc_N = Q.copy()

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
        weight_constr_unc_stage = weight_stage_constr,
        weight_constr_unc_terminal = weight_terminal_constr,
    )

    return ocp
