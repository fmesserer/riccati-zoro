from dataclasses import dataclass, field, replace
from typing import List
import numpy as np
import casadi as ca


@dataclass
class OcpDescription:
    x: ca.SX  # state symbol
    u: ca.SX  # control symbol
    T: float # time horizon
    N: int  # OCP horizon
    w: ca.SX = ca.SX()  # noise symbol
    par: ca.SX = ca.SX()  # parameter symbol
    slack_stage: ca.SX = ca.SX()  # slack variable
    slack_terminal: ca.SX = ca.SX()  # slack variable
    dyn_discr_expr: ca.SX = ca.SX(0)  # dynamics expression
    cost_stage_expr: ca.SX = ca.SX(0)  # stage cost expression
    cost_terminal_expr: ca.SX = ca.SX(0)  # terminal cost expression
    constr_stage_expr: ca.SX = ca.SX()  # stage constraint expression
    constr_terminal_expr: ca.SX = ca.SX()  # terminal constraint expression

    # for tube robust OCP
    state_feedback_gain: np.ndarray = field(default_factory=lambda: np.array([]))
    cost_uncertainty_Q_stage: np.ndarray = field(default_factory=lambda: np.array([]))
    cost_uncertainty_R: np.ndarray = field(default_factory=lambda: np.array([]))
    cost_uncertainty_S: np.ndarray = field(default_factory=lambda: np.array([]))
    cost_uncertainty_Q_terminal: np.ndarray = field(default_factory=lambda: np.array([]))
    backoff_eps: float = 1e-6
    weight_constr_unc_stage: np.ndarray = field(default_factory=lambda: np.array([]))
    weight_constr_unc_terminal: np.ndarray = field(default_factory=lambda: np.array([]))

    # for tree OCP
    Wval: List[np.ndarray] = field(default_factory=lambda: [])  # disturbance values per stage
    N_rob: int = 1  # uncertainty horizon

    def __post_init__(self):
        if self.state_feedback_gain.size == 0:
            self.state_feedback_gain = np.zeros((self.nu, self.nx))
        if self.cost_uncertainty_Q_stage.shape[0] == 0:
            self.cost_uncertainty_Q_stage = np.zeros((self.nx, self.nx))
        if self.cost_uncertainty_R.shape[0] == 0:
            self.cost_uncertainty_R = 1e-6 * np.eye(self.nu)
        if self.cost_uncertainty_S.shape[0] == 0:
            self.cost_uncertainty_S = np.zeros((self.nu, self.nx))
        if self.cost_uncertainty_Q_terminal.shape[0] == 0:
            self.cost_uncertainty_Q_terminal = np.zeros((self.nx, self.nx))
        if self.weight_constr_unc_stage.size == 0:
            self.weight_constr_unc_stage = np.ones((self.n_stage_constr,))
        if self.weight_constr_unc_terminal.size == 0:
            self.weight_constr_unc_terminal = np.ones((self.n_terminal_constr,))

    @property
    def nx(self) -> int:
        return self.x.shape[0]
    
    @property
    def nu(self) -> int:
        return self.u.shape[0]

    @property
    def npar(self) -> int:
        return self.par.shape[0]

    @property
    def nw(self) -> int:
        return self.w.shape[0]
    
    @property
    def dt(self) -> float:
        return self.T / self.N
    
    @property
    def n_stage_constr(self) -> int:
        return self.constr_stage_expr.shape[0]

    @property
    def n_terminal_constr(self) -> int:
        return self.constr_terminal_expr.shape[0]
    
    @property
    def n_slack_stage(self) -> int:
        return self.slack_stage.shape[0]

    @property
    def n_slack_terminal(self) -> int:
        return self.slack_terminal.shape[0]

    @property
    def n_dist(self) -> int:           # number of disturbance values
        return len(self.Wval)
    
    @property
    def stage_constr_lb(self) -> np.ndarray:
        return -np.inf * np.ones((self.n_stage_constr,))
    
    @property
    def stage_constr_ub(self) -> np.ndarray:
        return np.zeros((self.n_stage_constr,))
    
    @property
    def terminal_constr_lb(self) -> np.ndarray:
        return -np.inf * np.ones((self.n_terminal_constr,))
    
    @property
    def terminal_constr_ub(self) -> np.ndarray:
        return np.zeros((self.n_terminal_constr,))
    


def slack_the_constraints(ocp: OcpDescription, weight_stage: np.ndarray=1e4, weight_terminal: np.ndarray=1e4) -> OcpDescription:
    """Add slack variables to the constraints in the OCP description.

    Args:
        ocp (OcpDescription): Original OCP description.

    Returns:
        OcpDescription: OCP description with slack variables added to constraints.
    """

    # stage constraints
    slack_stage = ca.SX.sym("s_stage", ocp.n_stage_constr)
    constr_stage_expr_slacked = ocp.constr_stage_expr - slack_stage

    # terminal constraints
    slack_terminal = ca.SX.sym("s_terminal", ocp.n_terminal_constr)
    constr_terminal_expr_slacked = ocp.constr_terminal_expr - slack_terminal

    # penalize slack variables in cost
    cost_stage_w_slacks = ocp.cost_stage_expr + ca.sum1(weight_stage * slack_stage)
    cost_terminal_w_slacks = ocp.cost_terminal_expr + ca.sum1(weight_terminal * slack_terminal)

    ocp_slacked = replace(ocp,
                          slack_stage=slack_stage,
                          slack_terminal=slack_terminal,
                          constr_stage_expr=constr_stage_expr_slacked,
                          constr_terminal_expr=constr_terminal_expr_slacked,
                          cost_stage_expr=cost_stage_w_slacks,
                          cost_terminal_expr=cost_terminal_w_slacks)

    return ocp_slacked 