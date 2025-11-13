import time
from typing import Tuple
import numpy as np
import casadi as ca
from src.ocp_description import OcpDescription
from src.solver_nominal_ocp import SolverNominalOcp, OptionsSolver
from src.utils import vecToSymm, symmToVec

class SolverEllipsoidOcpFixedStatefb(SolverNominalOcp):
    def __init__(self, ocp: OcpDescription, options: OptionsSolver = OptionsSolver()):
        self.backoff_eps = 1e-4  # small regularization to avoid numerical issues
        super().__init__(ocp=ocp, options=options)

    def _build_solver(self) -> None:

        # dynamics and cost functions
        dyn_discr = ca.Function("dyn", [self.ocp.x, self.ocp.u, self.ocp.w, self.ocp.par], [self.ocp.dyn_discr_expr])
        stage_cost_func = ca.Function("stage_cost", [self.ocp.x, self.ocp.u, self.ocp.par], [self.ocp.cost_stage_expr])
        terminal_cost_func = ca.Function("terminal_cost", [self.ocp.x, self.ocp.par], [self.ocp.cost_terminal_expr])

        # ellipsoid dynamics
        # due to symmetry of P, we only need the lower triangular part as decision variable
        pvec = ca.SX.sym("pvec", int(self.ocp.nx*(self.ocp.nx+1)/2))
        p = vecToSymm(pvec, self.ocp.nx)
        beta = ca.SX.sym("beta", self.ocp.n_stage_constr)  # slack var for backoff
        beta_N = ca.SX.sym("beta_N", self.ocp.n_terminal_constr)  # slack var for backoff

        A = ca.jacobian(self.ocp.dyn_discr_expr, self.ocp.x)
        B = ca.jacobian(self.ocp.dyn_discr_expr, self.ocp.u)
        A = A + B @ self.ocp.state_feedback_gain  # closed loop A matrix
        Gamma = ca.jacobian(self.ocp.dyn_discr_expr, self.ocp.w)
        p_next = A @ p @ A.T + Gamma @ Gamma.T
        p_next_vec = symmToVec(p_next)
        dyn_ellipsoid = ca.Function("dyn_ellipsoid", [self.ocp.x, self.ocp.u, self.ocp.w, pvec, self.ocp.par], [p_next_vec])

        # stage
        # robustified constraint
        # we lift the backoff using the slack variable beta, because it is easier to ensure non-negativity of beta throughout the iterations
        # This is relevant because of the square root.
        constr_stage_rob_expr = self.ocp.constr_stage_expr + ca.sqrt(beta + self.backoff_eps)
        stage_constr_func = ca.Function("stage_constr", [self.ocp.x, self.ocp.u, beta, self.ocp.par], [constr_stage_rob_expr])
        
        constr_jac = ca.jacobian(self.ocp.constr_stage_expr, self.ocp.x)
        # if we use feedback, u depends on x, so it is also uncertain and control constraints need to be tightened
        constr_jac += ca.jacobian(self.ocp.constr_stage_expr, self.ocp.u) @ self.ocp.state_feedback_gain
        # square of backoff ("variance in constraint direction"), computed for each scalar constraint separately
        backoff_sq_expr = ca.veccat(*[constr_jac[i,:] @ p @ constr_jac[i,:].T for i in range(self.ocp.n_stage_constr)])
        backoff_sq_func = ca.Function("backoff_sq", [self.ocp.x, self.ocp.u, pvec, self.ocp.par], [backoff_sq_expr])

        # terminal
        constr_terminal_rob_expr = self.ocp.constr_terminal_expr + ca.sqrt(beta_N + self.backoff_eps)
        terminal_constr_func = ca.Function("terminal_constr", [self.ocp.x, beta_N, self.ocp.par], [constr_terminal_rob_expr])
        constr_jac = ca.jacobian(self.ocp.constr_terminal_expr, self.ocp.x)
        backoff_sq_expr = ca.veccat(*[constr_jac[i,:] @ p @ constr_jac[i,:].T for i in range(self.ocp.n_terminal_constr)])
        backoff_sq_func_terminal = ca.Function("backoff_sq_terminal", [self.ocp.x, pvec, self.ocp.par], [backoff_sq_expr])

        # build OCP
        # decision variables
        X = ca.SX.sym("X", self.ocp.nx, self.ocp.N+1)
        Pvec = ca.SX.sym("Pvec", int(self.ocp.nx*(self.ocp.nx+1)/2), self.ocp.N+1)
        U = ca.SX.sym("U", self.ocp.nu, self.ocp.N)
        Beta = ca.SX.sym("Beta", self.ocp.n_stage_constr, self.ocp.N) # backoff slack var for stage constraints
        beta_N = ca.SX.sym("beta_N", self.ocp.n_terminal_constr)      # backoff slack var for terminal constraints

        # vector of all decision variables ordered as [x0, pvec0, beta0, u0, x1, u1, ..., xN, pvecN, betaN]
        decvar = ca.veccat(X[:,0], Pvec[:,0], ca.vertcat(Beta, U, X[:,1:], Pvec[:,1:], ), beta_N)
        # to extract trajectories in nice shape from decvar
        self._extract_traj = ca.Function("extract_traj", [decvar], [X, U, Pvec, Beta, beta_N])
        self._traj_to_vec = ca.Function("traj_to_vec", [X, U, Pvec, Beta, beta_N], [decvar])

        # parameter var
        Par = ca.SX.sym("P", self.ocp.npar, self.ocp.N+1)
        par_var = ca.veccat(Par)
        # to extract parameter trajectory in nice shape from par_var
        self._extract_partraj = ca.Function("extract_par_traj", [par_var], [Par])
        self._partraj_to_vec = ca.Function("par_to_vec", [Par], [par_var])
        self._par_val = np.zeros(par_var.shape[0])  # default parameter value

        # set general bounds of decvars (x0, P0) will be set later
        # also ocp constraints are set g
        lb_X = -np.inf * np.ones(X.shape)
        ub_X = np.inf * np.ones(X.shape)
        lb_Pvec = -np.inf * np.ones(Pvec.shape)
        ub_Pvec = np.inf * np.ones(Pvec.shape)
        lb_U = -np.inf * np.ones(U.shape)
        ub_U = np.inf * np.ones(U.shape)
        lb_Beta = np.zeros(Beta.shape)   # slack vars for backoff must be non-negative to avoid the square root of negative numbers
        ub_Beta = np.inf * np.ones(Beta.shape)
        lb_beta_N = np.zeros(beta_N.shape)
        ub_beta_N = np.inf * np.ones(beta_N.shape)

        decvar_lb = self._traj_to_vec(lb_X, lb_U, lb_Pvec, lb_Beta, lb_beta_N).full().flatten()
        decvar_ub = self._traj_to_vec(ub_X, ub_U, ub_Pvec, ub_Beta, ub_beta_N).full().flatten()
        self._lbx = decvar_lb
        self._ubx = decvar_ub


        # objective
        obj = 0

        constr = []      # constraint expressions
        constr_lb = []    # lower bounds on constraints
        constr_ub = []    # upper bounds on constraints

        # iterate through stages
        for k in range(self.ocp.N):
            obj += stage_cost_func(X[:,k], U[:,k], Par[:,k])
            obj += 1e-4 * ca.sum1(Beta[:,k])  # small penalty to push actively into constraint

            x_next = dyn_discr(X[:,k], U[:,k], 0, Par[:,k])
            p_next_vec = dyn_ellipsoid(X[:,k], U[:,k], 0, Pvec[:,k], Par[:,k])

            # add dynamics constraint
            constr.append(X[:,k+1] - x_next)
            constr_lb.append(np.zeros((self.ocp.nx,)))
            constr_ub.append(np.zeros((self.ocp.nx,)))
            constr.append(Pvec[:,k+1] - p_next_vec)
            constr_lb.append(np.zeros((Pvec.shape[0],)))
            constr_ub.append(np.zeros((Pvec.shape[0],)))

            # add stage constraints
            constr.append(stage_constr_func(X[:,k], U[:,k], Beta[:,k], Par[:,k]))
            constr_lb.append([-np.inf] * self.ocp.n_stage_constr)
            constr_ub.append([0] * self.ocp.n_stage_constr)
            # link slack var to backoff
            constr.append(Beta[:,k] - backoff_sq_func(X[:,k], U[:,k], Pvec[:,k], Par[:,k]))
            constr_lb.append(np.zeros((self.ocp.n_stage_constr,)))
            constr_ub.append(np.inf * np.ones((self.ocp.n_stage_constr,)))


        # terminal cost
        obj += terminal_cost_func(X[:,self.ocp.N], Par[:,self.ocp.N])
        obj += 1e-4 * ca.sum1(beta_N)  # small penalty to push actively into constraint
        # terminal constraint
        constr.append(terminal_constr_func(X[:,self.ocp.N], beta_N, Par[:,self.ocp.N]))
        constr_lb.append([-np.inf] * self.ocp.n_terminal_constr)
        constr_ub.append([0] * self.ocp.n_terminal_constr)
        # link slack var to backoff
        constr.append(beta_N - backoff_sq_func_terminal(X[:,self.ocp.N], Pvec[:,self.ocp.N], Par[:,self.ocp.N]))
        constr_lb.append(np.zeros((self.ocp.n_terminal_constr,)))
        constr_ub.append(np.inf * np.ones((self.ocp.n_terminal_constr,)))

        constr = ca.veccat(*constr)

        nlp = {"x": decvar, "f": obj, "g": constr, "p": par_var}
        self._solver = ca.nlpsol("solver", "ipopt", nlp, self.options.opts_nlpsol)
        self._decvar = decvar
        self._X = X
        self._U = U
        self._Pvec = Pvec
        self._constr_lb = np.concatenate(constr_lb)
        self._constr_ub = np.concatenate(constr_ub)

    def solve(self, x0: np.ndarray, P0: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Solve the OCP for initial state x0 (nx, ) and initial covariance P0 (nx, nx).
        Returns state and control trajectory as arrays of shape (nx, N+1) and (nu, N).
        '''
        if P0 is None:
            P0 = np.zeros((self.ocp.nx, self.ocp.nx))
        P0vec = symmToVec(P0).full().flatten()

        # set initial state constraint
        self._lbx[:self.ocp.nx] = x0
        self._ubx[:self.ocp.nx] = x0
        self._lbx[self.ocp.nx:self.ocp.nx+P0vec.shape[0]] = P0vec
        self._ubx[self.ocp.nx:self.ocp.nx+P0vec.shape[0]] = P0vec

        print("---------------------------------------")
        print("Fixed-feedback OCP with IPOPT")
        tic = time.perf_counter()
        self._sol = self._solver(x0=self._init_guess, lbx=self._lbx, ubx=self._ubx, lbg=self._constr_lb, ubg=self._constr_ub, p=self._par_val)
        toc = time.perf_counter()
        return_status = self._solver.stats()["return_status"]
        self.success = self._solver.stats()["success"]
        self._iters = self._solver.stats()["iter_count"]
        print("{}, {} iterations.".format(return_status, self._iters))
        print(f"Solve time: {toc - tic:.3f} seconds")

        Xopt, Uopt, P_opt, _, _ = self._extract_traj(self._sol["x"])
        return Xopt.full(), Uopt.full(), self.Pvec_traj_to_P_list(P_opt.full())

    def set_initial_guess(self, X: np.ndarray, U: np.ndarray, P: np.ndarray, Beta: np.ndarray = 1e-4, beta_N: np.ndarray = 1e-4) -> None:
        '''
        Set the initial guess for the optimization problem.
        X: state trajectory of shape (nx, N+1)
        U: control trajectory of shape (nu, N)
        P: ellipsoid trajectory in vector form of shape (int(nx*(nx+1)/2), N+1)
        Beta: slack variable trajectory for stage constraints of shape (n_stage_constr, N)
        beta_N: slack variable for terminal constraints of shape (n_terminal_constr,)
        '''
        self._init_guess = self._traj_to_vec(X, U, P, Beta, beta_N).full().flatten()


    def Pvec_traj_to_P_list(self, Pvec: np.ndarray) -> list[np.ndarray]:
        '''
        Convert trajectory of Pvec (int(nx*(nx+1)/2), N+1) to list of P matrices [P0, P1, ..., PN] with each Pi of shape (nx, nx).
        '''
        P_list = [vecToSymm(Pvec[:,k], self.ocp.nx) for k in range(Pvec.shape[1])]
        return P_list

    def set_parameter_values(self, P: np.ndarray) -> None:
        '''
        Set the parameter trajectory for the optimization problem.
        P: parameter trajectory of shape (npar, N+1)
        '''
        self._par_val = self._partraj_to_vec(P).full().flatten()