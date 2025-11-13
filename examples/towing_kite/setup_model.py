from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
import casadi as ca
from examples.diff_drive_robot.setup_robot import RobotOcpParams
from src.ocp_description import OcpDescription
from src.utils import integrate_RK4

rad_per_deg = np.pi / 180 # conversion factor from degrees to radians

@dataclass
class KiteParams:
    N: int = 80 # discrete time steps
    dt: float = 0.3 # time step duration (s)
    E0: float = 5 # glide ratio in absence of steering deflection (non-dimensional)
    v0: float = 10 # apparent windspeed (m/s)
    ctilde: float = 0.028 # coefficient for glide ratio in dependence of steering angle (non-dimensional)
    rho: float = 1 # air density (kg / m^3)
    L: float = 400 # tether length (m)
    A: float = 300 # kite area (m^2)
    beta: float = 0 # related to angle between wind and boat (rad)
    hmin: float = 100 # minimal height (m)
    umin: float = -10 # lower control constraint (non-dimensional)
    umax: float = 10 # upper control constraint (non-dimensional)
    x0: np.ndarray = field(default_factory=lambda: np.array([20 * rad_per_deg, 30 * rad_per_deg, 0])) # initial state
    std_procnoise: np.ndarray = field(default_factory=lambda: 1e-4 * np.ones(3))  # process noise standard deviation
    std_wind: float = 1

    @property
    def T(self) -> float:
        return self.N * self.dt  # continuous time horizon (s)

    def glide_ratio(self, u):
        return self.E0 - self.ctilde * u**2  # glide ratio without steering deflection

    def height(self, theta, phi):
        return self.L * np.sin(theta) * np.cos(phi)

    def thrust_force(self, theta, phi, u):
        PD = self.rho * self.v0 ** 2 / 2
        E = self.glide_ratio(u)
        TF = PD * self.A * np.cos(theta)**2 * (E + 1) * np.sqrt(E**2 + 1) * \
                ( np.cos(theta) * np.cos(self.beta) + np.sin(theta) * np.sin(self.beta) * np.sin(phi) )
        return TF   

def setup_model(params: KiteParams) -> OcpDescription:
    # states
    theta = ca.SX.sym("theta")  # angle between wind and boat (rad)
    phi = ca.SX.sym("phi")      # azimuth angle (rad)
    psi = ca.SX.sym("psi")      # heading angle (rad)
    x = ca.vertcat(theta, phi, psi)

    # controls
    u = ca.SX.sym("u")  # steering deflection (non-dimensional)

    # noise
    w_proc = ca.SX.sym("w_proc", x.shape[0])  # process noise
    w_wind = ca.SX.sym("w_wind") # unknown wind speed
    w = ca.veccat(w_proc, w_wind) # all uncertainty

    # dynamics
    L = params.L
    E = params.glide_ratio(u)
    va = (params.v0 + params.std_wind * w_wind) * E * ca.cos(theta)

    theta_dot = va /  L * (np.cos(psi) - np.tan(theta) / E )
    phi_dot = - va * np.sin(psi) / ( L * np.sin(theta))
    psi_dot = va / params.L * u + phi_dot * np.cos(theta)

    xdot = ca.vertcat(theta_dot, phi_dot, psi_dot)
    xdot += params.std_procnoise * w_proc  # add process noise

    f_discrete = integrate_RK4(xdot=xdot, dt=params.dt, x=x, u=u, w=w)

    # cost
    cost_stage = -params.thrust_force(theta, phi, u) / params.N
    # rization
    cost_terminal = 0  # no terminal cost

    # constraints
    constr_hmin = params.hmin - params.height(theta, phi)  # minimal height constraint
    constr_umin = -u + params.umin  # lower control constraint
    constr_umax = u - params.umax  # upper control constraint

    constr_stage = ca.vertcat(constr_hmin, constr_umin, constr_umax)
    constr_terminal = ca.vertcat(constr_hmin)

    Q_ricc_k = np.eye(x.shape[0])
    R_ricc_k = 1e-2 * np.eye(u.shape[0])
    Q_ricc_N = np.eye(x.shape[0])

    weight_constr_stage =  np.ones((constr_stage.shape[0],))
    weight_constr_stage[0] = 100
    weight_constr_terminal = 100 * np.ones((constr_terminal.shape[0],))

    ocp = OcpDescription(
        x = x,
        u = u,
        w = w,
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
        weight_constr_unc_stage = weight_constr_stage,
        weight_constr_unc_terminal = weight_constr_terminal,
    )

    return ocp
