import time
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
import casadi as ca
from src.ocp_description import OcpDescription


@dataclass
class OptionsSolver():
   opts_nlpsol: dict = field(default_factory=dict)  # options passed to casadi.nlpsol


class SolverNominalOcp():
    def __init__(self, ocp: OcpDescription, options: OptionsSolver = OptionsSolver()):
        self.ocp = ocp
        self.options = options
        self._build_solver()
        self.success = False
        self._sol = None
        self._init_guess = 0


    def _build_solver(self) -> None:

        # dynamics and cost functions
        dyn_expr = self.ocp.dyn_discr_expr
        dyn_expr = ca.substitute(dyn_expr, self.ocp.w, 0)
        dyn_func = ca.Function("dyn", [self.ocp.x, self.ocp.u, self.ocp.par], [dyn_expr])
        stage_cost_func = ca.Function("stage_cost", [self.ocp.x, self.ocp.u, self.ocp.slack_stage, self.ocp.par], [self.ocp.cost_stage_expr])
        terminal_cost_func = ca.Function("terminal_cost", [self.ocp.x, self.ocp.slack_terminal, self.ocp.par], [self.ocp.cost_terminal_expr])
        stage_constr_func = ca.Function("stage_constr", [self.ocp.x, self.ocp.u, self.ocp.slack_stage, self.ocp.par], [self.ocp.constr_stage_expr])
        terminal_constr_func = ca.Function("terminal_constr", [self.ocp.x, self.ocp.slack_terminal, self.ocp.par], [self.ocp.constr_terminal_expr])

        # build OCP
        # decision variables
        X = ca.SX.sym('X', self.ocp.nx, self.ocp.N+1)
        U = ca.SX.sym("U", self.ocp.nu, self.ocp.N)
        S_k = ca.SX.sym("S_k", self.ocp.n_slack_stage, self.ocp.N)
        S_N = ca.SX.sym("S_N", self.ocp.n_slack_terminal)

        decvar_xu = [ca.veccat(X[:,k], U[:,k]) for k in range(self.ocp.N)] + [X[:,self.ocp.N]]
        decvar_xu_vec = ca.veccat(*decvar_xu)
        decvar_ordered_list = [ ca.veccat(X[:,k], U[:,k], S_k[:,k]) for k in range(self.ocp.N)] + [X[:,self.ocp.N], S_N]

        # vector of all decision variables ordered as [x0, u0, s0, x1, u1, ..., xN, sN]
        decvar_vec = ca.veccat(*decvar_ordered_list)
        # to extract state and control trajectories in nice shape from decvar
        self._extract_traj = ca.Function("extract_traj", [decvar_vec], [X, U])
        self._traj_to_vec = ca.Function("traj_to_vec", [X, U], [decvar_xu_vec])
        self._extract_traj_xus = ca.Function("extract_traj_xus", [decvar_vec], [X, U, S_k, S_N])
        self._traj_to_vec_xus = ca.Function("traj_to_vec_xus", [X, U, S_k, S_N], [decvar_vec])

        # direct bound on decision variables
        self._lbx = self._traj_to_vec_xus(-ca.inf, -ca.inf, 0, 0).full().flatten()
        self._ubx = self._traj_to_vec_xus(ca.inf, ca.inf, ca.inf, ca.inf).full().flatten()

        # parameters of OCP definition
        Par_prob = ca.SX.sym("P", self.ocp.npar, self.ocp.N+1)
        par_prob_vec = ca.veccat(Par_prob)
        # to extract parameter trajectory in nice shape from par_var
        self._partraj_to_vec = ca.Function("par_to_vec", [Par_prob], [par_prob_vec])
        self._Par_prob_val = np.zeros(Par_prob.shape)  # default parameter value

        # algorithmic parameters (for algorithms build on top of nominalOCPsolver)
        grad_cor = ca.SX.sym("grad_cor", decvar_xu_vec.shape[0])
        self._grad_cor_val = np.zeros(grad_cor.shape[0])

        par_all_vec = ca.veccat(par_prob_vec, grad_cor)
        self._par_all_to_vec = ca.Function("par_all_to_vec", [par_prob_vec, grad_cor], [par_all_vec])

        # backoff variable, for parametric upper bounds (ubg)
        Backoff = ca.SX.sym("Backoff", self.ocp.n_stage_constr, self.ocp.N)
        Backoff_N = ca.SX.sym("Backoff_N", self.ocp.n_terminal_constr)
        backoff_var = ca.veccat(Backoff, Backoff_N)

        # objective
        obj = 0

        constr = []      # constraint expressions
        constr_lb = []    # lower bounds on constraints
        constr_ub = []    # upper bounds on constraints

        # iterate through stages
        for k in range(self.ocp.N):
            obj += stage_cost_func(X[:,k], U[:,k], S_k[:,k], Par_prob[:,k])

            x_next = dyn_func(X[:,k], U[:,k], Par_prob[:,k])

            # add dynamics constraint
            constr.append(X[:,k+1] - x_next)
            constr_lb.append(np.zeros((self.ocp.nx,)))
            constr_ub.append(np.zeros((self.ocp.nx,)))

          # add stage constraints
            constr.append(stage_constr_func(X[:,k], U[:,k], S_k[:,k], Par_prob[:,k]))
            constr_lb.append(self.ocp.stage_constr_lb)
            constr_ub.append(self.ocp.stage_constr_ub - Backoff[:,k])

        # terminal cost
        obj += terminal_cost_func(X[:,self.ocp.N], S_N, Par_prob[:,self.ocp.N])
        constr.append(terminal_constr_func(X[:,self.ocp.N], S_N, Par_prob[:,self.ocp.N]))
        constr_lb.append(self.ocp.terminal_constr_lb)
        constr_ub.append(self.ocp.terminal_constr_ub - Backoff_N)

        constr = ca.veccat(*constr)
        constr_ub = ca.veccat(*constr_ub)
        constr_ub_eval = ca.Function("constr_ub_eval", [Backoff, Backoff_N], [constr_ub])
        self._g_idx_backoff = ca.evalf(ca.sum2(ca.jacobian(constr_ub, backoff_var))).full().nonzero()[0]

        # gradient correction
        obj += grad_cor.T @ decvar_xu_vec   # gradient correction term

        nlp = {"x": decvar_vec, "f": obj, "g": constr, "p": par_all_vec}
        self._solver = ca.nlpsol("solver", "ipopt", nlp, self.options.opts_nlpsol)
        self._decvar_vec = decvar_vec
        self._decvar_xu_vec = decvar_xu_vec
        self._X = X
        self._U = U
        self._Par_prob = Par_prob
        self._constr_lb = np.concatenate(constr_lb)
        self._constr_ub_eval = constr_ub_eval
        self._constr_ub =  constr_ub_eval(0,0).full().flatten()

    def solve(self, x0: np.ndarray, silent=False) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Solve the OCP for initial state x0.
        Returns state and control trajectory as arrays of shape (nx, N+1) and (nu, N).
        '''

        # set initial state constraint
        self._lbx[:self.ocp.nx] = x0
        self._ubx[:self.ocp.nx] = x0

        if not silent:
            print("---------------------------------------")
            print("Nominal OCP with IPOPT")

        tic = time.perf_counter()
        self._sol = self._solver(x0=self._init_guess, lbx=self._lbx, ubx=self._ubx, lbg=self._constr_lb, ubg=self._constr_ub, p=self.par_all_val)
        toc = time.perf_counter()
        return_status = self._solver.stats()["return_status"]
        self.success = self._solver.stats()["success"]
        if not self.success:
            print("Warning: Solver nominal OCP returned status {}".format(return_status))
        self._iters = self._solver.stats()["iter_count"]
        if not silent:
            print("{}, {} iterations.".format(return_status, self._iters))
            print(f"Solve time: {toc - tic:.3f} seconds")

        Xopt, Uopt = self._extract_traj(self._sol["x"])
        return Xopt.full(), Uopt.full()
    

    def get_optimal_trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Get the optimal state and control trajectory after solve() has been called.
        Returns state and control trajectory as arrays of shape (nx, N+1) and (nu, N).
        '''
        if self._sol is None:
            raise RuntimeError("No solution available. Call solve() first.")
        Xopt, Uopt = self._extract_traj(self._sol["x"])
        return Xopt.full(), Uopt.full()
    
    def set_backoffs(self, backoff_stage: np.ndarray, backoff_terminal: np.ndarray) -> None:
        '''
        Set the backoff values for parametric upper bounds on constraints.
        backoff_stage: backoff for stage constraints of shape (n_stage_constr, N)
        backoff_terminal: backoff for terminal constraints of shape (n_terminal_constr,)
        '''
        self._constr_ub = self._constr_ub_eval(backoff_stage, backoff_terminal).full().flatten()

    def set_parameter_values(self, P: np.ndarray) -> None:
        '''
        Set the parameter trajectory for the optimization problem.
        P: parameter trajectory of shape (npar, N+1)
        '''
        self._Par_prob_val = P

    def set_gradient_correction(self, c: np.ndarray) -> None:
        '''
        Set the gradient correction term for the objective.
        c: gradient correction vector of shape (nx*(N+1) + nu*N,)
                sorted corresponding to [x0, u0, x1, u1, ..., xN]
        '''
        self._grad_cor_val[:] = c

    def get_inequality_multipliers_as_vec(self) -> np.ndarray:

        if self._sol is None:
            raise RuntimeError("No solution available. Call solve() first.")
        lam_g = self._sol["lam_g"].full().flatten()
        mu = lam_g[self._g_idx_backoff]
        return mu
    
    def get_inequality_multipliers(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Get the inequality multipliers for stage and terminal constraints.
        Returns:
            mu_stage: multipliers for stage constraints of shape (n_stage_constr, N)
            mu_terminal: multipliers for terminal constraints of shape (n_terminal_constr,)
        '''
        mu = self.get_inequality_multipliers_as_vec()
        mu_stage = mu[:self.ocp.n_stage_constr * self.ocp.N].reshape((self.ocp.n_stage_constr, self.ocp.N), order="F")
        mu_terminal = mu[self.ocp.n_stage_constr * self.ocp.N:]
        return mu_stage,  mu_terminal

    def par_prob_val_as_vec(self) -> np.ndarray:
        return self._partraj_to_vec(self._Par_prob_val).full().flatten()

    @property
    def par_all_val(self) -> np.ndarray:
        return self._par_all_to_vec(self.par_prob_val_as_vec(), self._grad_cor_val).full().flatten()

    def set_initial_guess(self, X: np.ndarray, U: np.ndarray, S_stage: np.ndarray=0, S_N: np.ndarray=0) -> None:
        '''
        Set the initial guess for the optimization problem.
        X: state trajectory of shape (nx, N+1)
        U: control trajectory of shape (nu, N)
        S_stage: slack variable trajectory of shape (n_slack_stage, N)
        S_N: terminal slack variable of shape (n_slack_terminal,)
        '''
        self._init_guess = self._traj_to_vec_xus(X, U, S_stage, S_N).full().flatten()
