import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import casadi as ca

from src.solver_nominal_ocp import OptionsSolver, SolverNominalOcp
from src.ocp_description import OcpDescription
from src.utils import symmToVec, vecToSymm

class RiccatiConstraintCost(Enum):
    NONE = auto()
    HESSIAN_LOG = auto()
    LAGRANGE_MULTIPLIERS = auto()       # -> SIRO


@dataclass
class OptionsRiccatiZoro():
    max_iter: int = 50
    convtol: float = 1e-6 # convergence tolerance, on norm of step in nominal trajectory
    backoff_eps: float= 1e-4  # eval backoff squareroot as sqrt(beta + backoff_eps)
    gradient_correction: bool = False 
    compute_riccati: bool = True 
    riccati_use_const_cost: bool = True # use the constant cost matrices defined in OcpDescription
    riccati_constraint_cost: RiccatiConstraintCost = RiccatiConstraintCost.NONE # type of cost assigned to backoffs in Riccati recursion
    opts_nlpsol: dict = field(default_factory=dict)  # options passed to casadi.nlpsol


class SolverRiccatiZoro():
    """
    Riccati-ZORO solver for uncertain OCPs.
    This class also implements ZORO and SIRO as a special case.
    All methods can be run with or without gradient correction.
    """
    def __init__(self, ocp: OcpDescription, options: OptionsRiccatiZoro = OptionsRiccatiZoro()):
        self.ocp = ocp
        self.options = options
        self.success = False
        self._sol = None
        self._init_guess = 0
        self._kfb_vec_traj_val = 0
        self._build_solver()

    def _build_solver(self) -> None:

        # build Nominal OCP
        self._solver_nominal = SolverNominalOcp(self.ocp, options=OptionsSolver(opts_nlpsol=self.options.opts_nlpsol))

        X = self._solver_nominal._X
        U = self._solver_nominal._U
        traj_xu_vec = self._solver_nominal._decvar_xu_vec
        Par_prob = self._solver_nominal._Par_prob
        par_prob_vec = self._solver_nominal._partraj_to_vec(Par_prob)
        # build uncertainy propagation functions

        # ellipsoid dynamics
        # due to symmetry of P, we only need the lower triangular part
        pvec = ca.SX.sym("pvec", int(self.ocp.nx*(self.ocp.nx+1)/2))
        pvec_traj = ca.SX.sym("pvec_traj", int(self.ocp.nx*(self.ocp.nx+1)/2), self.ocp.N+1)
        p = vecToSymm(pvec, self.ocp.nx)
        kfb_vec = ca.SX.sym("kfb_vec", self.ocp.nu * self.ocp.nx)  # feedback gain as vector
        kfb = ca.reshape(kfb_vec, (self.ocp.nu, self.ocp.nx))

        A = ca.jacobian(self.ocp.dyn_discr_expr, self.ocp.x)
        B = ca.jacobian(self.ocp.dyn_discr_expr, self.ocp.u)
        Gamma = ca.jacobian(self.ocp.dyn_discr_expr, self.ocp.w)
        Acl = A + B @ kfb  # closed loop A matrix
        p_next = Acl @ p @ Acl.T + Gamma @ Gamma.T
        p_next_vec = symmToVec(p_next)
        dyn_ellipsoid = ca.Function("dyn_ellipsoid", [pvec, self.ocp.x, self.ocp.u, self.ocp.w, kfb_vec, self.ocp.par], [p_next_vec])
        
        kfb_vec_traj = ca.SX.sym("kfb_vec_traj", self.ocp.nu * self.ocp.nx, self.ocp.N)  # feedback gain as vector for all stages
        dyn_ellipsoid_mapaccum = dyn_ellipsoid.mapaccum(self.ocp.N)
        p_traj_elim = dyn_ellipsoid_mapaccum(pvec, X[:,:-1], U, 0, kfb_vec_traj, Par_prob[:,:-1])
        p_traj_elim = ca.horzcat(pvec, p_traj_elim)
        self.eval_dyn_ellipsoid = ca.Function("eval_dyn_ellipsoid", [pvec, X[:,:-1], U, kfb_vec_traj, Par_prob[:,:-1]], [p_traj_elim])

        # constraints and backoffs
        constr_stage_expr = ca.substitute(self.ocp.constr_stage_expr, self.ocp.slack_stage, 0)
        stage_constr_func = ca.Function("stage_constr", [self.ocp.x, self.ocp.u, self.ocp.par], [constr_stage_expr])
        self._stage_constr_map = stage_constr_func.map(self.ocp.N)
        constr_terminal_expr = ca.substitute(self.ocp.constr_terminal_expr, self.ocp.slack_terminal, 0)
        self._term_constr_func = ca.Function("terminal_constr", [self.ocp.x, self.ocp.par], [constr_terminal_expr])

        # stage backoffs
        constr_jac = ca.jacobian(constr_stage_expr, self.ocp.x)
        # if we use feedback, u depends on x, so it is also uncertain and control constraints need to be tightened
        constr_jac += ca.jacobian(constr_stage_expr, self.ocp.u) @ kfb
        # square of backoff ("variance in constraint direction"), computed for each scalar constraint separately
        beta = ca.veccat(*[constr_jac[i,:] @ p @ constr_jac[i,:].T for i in range(self.ocp.n_stage_constr)])
        backoff_stage_expr = ca.sqrt(beta + self.options.backoff_eps)
        backoff_stage_func = ca.Function("backoff_stage", [self.ocp.x, self.ocp.u, pvec, self.ocp.w, kfb_vec, self.ocp.par], [backoff_stage_expr])
        self.eval_backoff_stage = backoff_stage_func.map(self.ocp.N)

        # terminal
        constr_jac_N = ca.jacobian(self.ocp.constr_terminal_expr, self.ocp.x)
        beta_N = ca.veccat(*[constr_jac_N[i,:] @ p @ constr_jac_N[i,:].T for i in range(self.ocp.n_terminal_constr)])
        backoff_terminal_expr = ca.sqrt(beta_N + self.options.backoff_eps)
        self.eval_backoff_terminal = ca.Function("backoff_terminal", [self.ocp.x, pvec, self.ocp.w, self.ocp.par], [backoff_terminal_expr])

        # heuristic shadow cost assigned to constraints, informing the Riccati recursion about constraint vicinity
        z = ca.SX.sym('z')  # symbol for single constraint function value
        log_tau = 1
        if self.options.riccati_constraint_cost == RiccatiConstraintCost.HESSIAN_LOG:
            shadow_cost = ca.hessian(-log_tau * ca.log(-z), z)[0] # expectation of 2nd order taylor of log barrier
        else:
            shadow_cost = ca.SX(0)  # to make sure it's defined
        shadow_cost_heuristic = ca.Function('shadow_cost_heuristic', [z], [shadow_cost])

        mu_stage = ca.SX.sym('mu_stage', self.ocp.n_stage_constr)
        mu_terminal = ca.SX.sym('mu_terminal', self.ocp.n_terminal_constr)

        backoff_stage = ca.SX.sym('backoff_stage', self.ocp.n_stage_constr)
        backoff_terminal = ca.SX.sym('backoff_terminal', self.ocp.n_terminal_constr)

        if self.options.riccati_constraint_cost == RiccatiConstraintCost.HESSIAN_LOG:
            eta_stage_expr = shadow_cost_heuristic( constr_stage_expr) * self.ocp.weight_constr_unc_stage
            eta_terminal_expr = shadow_cost_heuristic( constr_terminal_expr) * self.ocp.weight_constr_unc_terminal
        elif self.options.riccati_constraint_cost == RiccatiConstraintCost.LAGRANGE_MULTIPLIERS:
            eta_stage_expr = self.mu_to_eta(mu_stage, backoff_stage)
            eta_terminal_expr = self.mu_to_eta(mu_terminal, backoff_terminal)
        else:
            eta_stage_expr = ca.SX.zeros(self.ocp.n_stage_constr)
            eta_terminal_expr = ca.SX.zeros(self.ocp.n_terminal_constr)

        self.eval_eta_stage = ca.Function("eval_eta_stage", [self.ocp.x, self.ocp.u, backoff_stage, mu_stage, self.ocp.par], [eta_stage_expr])
        self.eval_eta_terminal = ca.Function("eval_eta_terminal", [self.ocp.x, backoff_terminal, mu_terminal, self.ocp.par], [eta_terminal_expr])

        # Riccati recursion
        self.set_constant_feedback_gain(self.ocp.state_feedback_gain)
        eta_k = ca.SX.sym('eta', self.ocp.n_stage_constr)
        eta_N = ca.SX.sym('eta_N', self.ocp.n_terminal_constr)

        # Cost matrices for riccati
        C_k = ca.SX(self.ocp.nx + self.ocp.nu, self.ocp.nx + self.ocp.nu)
        C_N = ca.SX(self.ocp.nx, self.ocp.nx)

        # C_k[:self.ocp.nx, :self.ocp.nx] += 1e-6 * ca.SX.eye(self.ocp.nx)
        C_k[self.ocp.nx:, self.ocp.nx:] += 1e-6 * ca.SX.eye(self.ocp.nu)
        # C_N += 1e-6 * ca.SX.eye(self.ocp.nx)
        
        if self.options.riccati_use_const_cost:
            C_k[:self.ocp.nx, :self.ocp.nx] += self.ocp.cost_uncertainty_Q_stage
            C_k[self.ocp.nx:, self.ocp.nx:] += self.ocp.cost_uncertainty_R
            C_k[self.ocp.nx:, :self.ocp.nx] += self.ocp.cost_uncertainty_S
            C_k[:self.ocp.nx, self.ocp.nx:] += self.ocp.cost_uncertainty_S.T
            C_N += self.ocp.cost_uncertainty_Q_terminal

        # components of C_k that are directly on the objective, i.e., without constraints
        C_k_obj = C_k + 0
        C_N_obj = C_N + 0

        for i in range(self.ocp.n_stage_constr):
            h_i = self.ocp.constr_stage_expr[i]
            grad_h_i = ca.gradient(h_i, ca.veccat(self.ocp.x, self.ocp.u))
            C_k += eta_k[i] * grad_h_i @ grad_h_i.T

        for i in range(self.ocp.n_terminal_constr):
            h_i = self.ocp.constr_terminal_expr[i]
            grad_h_i = ca.gradient(h_i, self.ocp.x)
            C_N += eta_N[i] * grad_h_i @ grad_h_i.T

        self.C_k_func = ca.Function("C_k_func", [self.ocp.x, self.ocp.u, eta_k, self.ocp.par], [C_k])
        self.C_N_func = ca.Function("C_N_func", [self.ocp.x, eta_N, self.ocp.par], [C_N])

        Q = C_k[:self.ocp.nx, :self.ocp.nx]
        S = C_k[self.ocp.nx:, :self.ocp.nx]
        R = C_k[self.ocp.nx:, self.ocp.nx:]
        Q_N = self.C_N_func(X[:,-1], eta_N, Par_prob[:,-1])

        Eta_stages = ca.SX.sym('Eta', self.ocp.n_stage_constr, self.ocp.N)

        V_vec = ca.SX.sym("V_vec", int(self.ocp.nx*(self.ocp.nx+1)/2))  # value function hessian
        V = vecToSymm(V_vec, self.ocp.nx)
        K_ricc = -ca.solve(R + B.T @ V @ B, S + B.T @ V @ A)  # riccatti feedback gain
        V_next = Q + A.T @ V @ A + (S.T + A.T @ V @ B) @ K_ricc
        V_next_vec = symmToVec(V_next)
        K_ricc_vec = ca.reshape(K_ricc, (self.ocp.nu * self.ocp.nx, 1))
        self.riccati_step = ca.Function("riccati_step", [V_vec, self.ocp.x, self.ocp.u, self.ocp.w, eta_k, self.ocp.par], [V_next_vec, K_ricc_vec])

        riccati_step_mapaccum = self.riccati_step.mapaccum(self.ocp.N)
        _, K_ricc_vec_traj = riccati_step_mapaccum(symmToVec(Q_N), X[:,-2::-1], U[:,::-1], 0, Eta_stages[:,::-1], Par_prob[:,-2::-1])
        K_ricc_vec_traj = K_ricc_vec_traj[:,::-1]   # reverse because we solved backwards


        self.eval_riccati = ca.Function("eval_riccati", [X, U, Eta_stages, eta_N, Par_prob], [K_ricc_vec_traj])

        # ******************************
        # construct gradient correction
        # ******************************

        # some variables for the inequality constraint multipliers
        mu_stages = ca.SX.sym("mu_stages", Eta_stages.shape[0], Eta_stages.shape[1])  # adjoint variable for backoffs
        mu_N = ca.SX.sym("mu_N", eta_N.shape[0])  # adjoint variable for backoffs
        mu_vec = ca.veccat(mu_stages, mu_N)

        # build direct uncertainty objective F(y, M) (without constraint contribution) 
        obj_unc_stage_expr = ca.trace(C_k_obj @ ca.blockcat([[p, p @ kfb.T], [kfb @ p, kfb @ p @ kfb.T]]))
        obj_unc_stage_func = ca.Function("obj_unc_stage", [self.ocp.x, self.ocp.u, self.ocp.w, pvec, kfb_vec, self.ocp.par], [obj_unc_stage_expr])
        obj_unc_stage_map = obj_unc_stage_func.map(self.ocp.N)
        obj_unc_terminal_expr = ca.trace(C_N_obj @ vecToSymm(pvec_traj[:,-1], self.ocp.nx))
        obj_unc = ca.sum2(obj_unc_stage_map(X[:,:-1], U, 0, pvec_traj[:,:-1], kfb_vec_traj, Par_prob[:,:-1])) + obj_unc_terminal_expr
        obj_unc_func = ca.Function("obj_unc", [traj_xu_vec, pvec_traj, kfb_vec_traj, par_prob_vec], [obj_unc])

        # backoffs as a large vector
        backoffs_stage_expr = self.eval_backoff_stage(X[:,:-1], U, pvec_traj[:,:-1], 0, kfb_vec_traj, Par_prob[:,:-1])
        backoffs_terminal_expr = self.eval_backoff_terminal(X[:, -1], pvec_traj[:,-1], 0, Par_prob[:, -1])
        backoffs_vec_expr = ca.veccat(backoffs_stage_expr, backoffs_terminal_expr)
        backoffs_vec_func = ca.Function("eval_backoffs", [traj_xu_vec, pvec_traj, kfb_vec_traj, par_prob_vec], [backoffs_vec_expr])

        # Feedback gains as function of nominal trajectory etc.
        # In the Riccati for SIRO, we have eta = mu_to_eta(mu, backoff), such that we need a variable for the current backoff
        backoffs_stage_var = ca.SX.sym("backoff_stage_var", backoffs_stage_expr.shape[0], backoffs_stage_expr.shape[1])
        backoffs_terminal_var = ca.SX.sym("backoff_terminal_var", backoffs_terminal_expr.shape[0])
        backoffs_vec_var = ca.veccat(backoffs_stage_var, backoffs_terminal_var)

        # Constraint weights in the Riccati recursion 
        Eta_stages_elim = self.eval_eta_stage( X[:,:-1], U, backoffs_stage_var, mu_stages, Par_prob[:,:-1] )
        eta_N_elim = self.eval_eta_terminal( X[:,-1], backoffs_terminal_var,  mu_N, Par_prob[:,-1] )

        if self.options.compute_riccati:
            # feedback gains as a function of nominal trajectory etc via Riccati
            kfb_vec_traj_eliminated = self.eval_riccati(X, U, Eta_stages_elim, eta_N_elim, Par_prob)
        else:
            kfb_vec_traj_eliminated = self._kfb_vec_traj_val

        # ellipsoids eliminated as a function of nominal trajectory etc and *Feedback gains*
        pvec_traj_of_kfb= self.eval_dyn_ellipsoid(pvec_traj[:,0], X[:,:-1], U, kfb_vec_traj, Par_prob[:,:-1])
        # as above, but feedback gains are also eliminated
        pvec_traj_eliminated = self.eval_dyn_ellipsoid(pvec_traj[:,0], X[:,:-1], U, kfb_vec_traj_eliminated, Par_prob[:,:-1])

        # Build Lagrangian contribution
        if self.options.compute_riccati and self.options.riccati_constraint_cost == RiccatiConstraintCost.LAGRANGE_MULTIPLIERS \
            or not self.options.compute_riccati:
            # for SIRO and ZORO we only need to differentiate through the ellipsoid propagation
            # Lagr = F(y, M) + \mu^\top b(y, M)
            lagrange_term = obj_unc_func(traj_xu_vec, pvec_traj_of_kfb, kfb_vec_traj, par_prob_vec)
            lagrange_term += mu_vec.T @ backoffs_vec_func(traj_xu_vec, pvec_traj_of_kfb, kfb_vec_traj, par_prob_vec)
        else:
            # for Riccati-ZORO, we need to differentiate through the Riccati recursion
            # Lagr = F(y, m(y)) + \mu^\top b(y, m(y))
            lagrange_term = obj_unc_func(traj_xu_vec, pvec_traj_eliminated, kfb_vec_traj_eliminated, par_prob_vec)
            lagrange_term += mu_vec.T @ backoffs_vec_func(traj_xu_vec, pvec_traj_eliminated, kfb_vec_traj_eliminated, par_prob_vec)

        # compute gradient
        gradcor_prelim = ca.Function("dLag_dy", [traj_xu_vec, pvec_traj[:,0], kfb_vec_traj, mu_vec, backoffs_vec_var, par_prob_vec], [ca.gradient(lagrange_term, traj_xu_vec)])
        # substitute feedback gains, M = m(y) (necessary for ZORO, SIRO)
        gradient_correction = gradcor_prelim(traj_xu_vec, pvec_traj[:,0], kfb_vec_traj_eliminated, mu_vec, backoffs_vec_var, par_prob_vec)
        self.eval_gradient_correction = ca.Function("eval_gradient_correction", [traj_xu_vec, pvec_traj[:,0], mu_vec, backoffs_vec_var, par_prob_vec], [gradient_correction])


    def solve(self, x0: np.ndarray, P0:np.ndarray = None ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        print("---------------------------------------")
        print(self.solver_info_str())
        tic = time.perf_counter()

        backoff_stage = np.sqrt(self.options.backoff_eps) * np.ones((self.ocp.n_stage_constr, self.ocp.N))
        backoff_terminal = np.sqrt(self.options.backoff_eps) * np.ones(self.ocp.n_terminal_constr)
        # call nominal ocp solver once, to make sure everything is properly initialized (e.g. constraint multpliers)
        # (actually not necessary for (Riccati-)ZORO without gradient correction, but for some other variants)
        Xnom, Unom = self.initialize_to_nominal(x0=x0, backoff_stage=backoff_stage, backoff_terminal=backoff_terminal, silent=True)

        if P0 is None:
            P0 = np.zeros((self.ocp.nx, self.ocp.nx))
        p0_vec = symmToVec(P0)
        kfb_vec_traj = self._kfb_vec_traj_val

        for i in range(self.options.max_iter):
            # print('Riccati-ZORO iteration {}'.format(i+1))

            # multipliers only needed for gradient correction or LAGRANGE_MULTIPLIERS riccati cost
            mu_stages, mu_N = self._solver_nominal.get_inequality_multipliers()

            # compute feedback gain for current nominal trajectory
            if self.options.compute_riccati:
                # compute backoff weights for Riccati
                eta_stages = self.eval_eta_stage( Xnom[:,:-1], Unom, backoff_stage, mu_stages, self._solver_nominal._Par_prob_val[:,:-1] ).full()
                eta_N = self.eval_eta_terminal( Xnom[:,-1], backoff_terminal, mu_N, self._solver_nominal._Par_prob_val[:,-1] ).full()
                kfb_vec_traj = self.eval_riccati(Xnom, Unom, eta_stages, eta_N, self._solver_nominal._Par_prob_val)
            
            # propagate ellipsoids
            p_traj = self.eval_dyn_ellipsoid(p0_vec, Xnom[:,:-1], Unom, kfb_vec_traj, self._solver_nominal._Par_prob_val[:,:-1])

            # compute backoffs 
            backoff_stage = self.eval_backoff_stage(Xnom[:,:-1], Unom, p_traj[:,:-1], 0, kfb_vec_traj, self._solver_nominal._Par_prob_val[:,:-1])
            backoff_terminal = self.eval_backoff_terminal(Xnom[:, -1], p_traj[:,-1], 0, self._solver_nominal._Par_prob_val[:,-1])
           
            # set backoffs
            self._solver_nominal.set_backoffs(backoff_stage.full(), backoff_terminal.full())
            
            # set gradient correction
            mu = self._solver_nominal.get_inequality_multipliers_as_vec()
            if self.options.gradient_correction:
                bo_vec = ca.veccat(backoff_stage, backoff_terminal).full().flatten()
                grad_corr = self.eval_gradient_correction(self._solver_nominal._traj_to_vec(Xnom, Unom), p0_vec, mu, bo_vec, self._solver_nominal.par_prob_val_as_vec())
                self._solver_nominal.set_gradient_correction(grad_corr.full().flatten())
            
            # solve nominal OCP with fixed backoffs
            self._solver_nominal.set_initial_guess(Xnom, Unom)
            Xnom_new, Unom_new = self._solver_nominal.solve(x0, silent=True)

            # check convergence
            dXnom_norm = np.linalg.norm(Xnom_new - Xnom, ord=np.inf)
            dUnom_norm = np.linalg.norm(Unom_new - Unom, ord=np.inf)
            converged = (dXnom_norm < self.options.convtol) and (dUnom_norm < self.options.convtol)
            if converged:
                print(f"Converged in {i+1} iterations.")
                break

            Xnom = Xnom_new + 0
            Unom = Unom_new + 0

        if not converged:
            print("Did not converge within {} iterations.".format(self.options.max_iter))

        toc = time.perf_counter()
        print(f"Solve time: {toc - tic:.3f} seconds")

        self._kfb_vec_traj_val = kfb_vec_traj + 0

        # extract P trajectory
        P = self.Pvec_traj_to_P_list(p_traj.full())
        K = self.Kvec_traj_to_K_list(self._kfb_vec_traj_val.full())
        return Xnom, Unom, P, K
    
    def initialize_to_nominal(self, x0: np.ndarray, backoff_stage: np.ndarray = 0, backoff_terminal: np.ndarray = 0, silent:bool=True) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Initialize the solver to the nominal solution.
        '''
        self._solver_nominal.set_backoffs(backoff_stage, backoff_terminal)
        self._solver_nominal.set_gradient_correction(0)
        self._solver_nominal.solve(x0, silent=silent)
        Xopt, Uopt = self._solver_nominal.get_optimal_trajectory()
        return Xopt, Uopt
 
    def set_initial_guess(self, X: np.ndarray, U: np.ndarray) -> None:
        '''
        Set the initial guess for the optimization problem.
        X: state trajectory of shape (nx, N+1)
        U: control trajectory of shape (nu, N)
        '''
        self._solver_nominal.set_initial_guess(X, U)

    def set_parameter_values(self, P: np.ndarray) -> None:
        '''
        Set the parameter trajectory for the optimization problem.
        P: parameter trajectory of shape (npar, N+1)
        '''
        self._solver_nominal.set_parameter_values(P)

    def set_constant_feedback_gain(self, K: np.ndarray) -> None:
        '''
        Set a constant state feedback gain for the uncertainty propagation and backoff computation.
        K: feedback gain of shape (nu, nx)
        '''
        self._kfb_vec_traj_val = self.K_list_to_Kvec_traj([K]*self.ocp.N)

    def Pvec_traj_to_P_list(self, Pvec: np.ndarray) -> list[np.ndarray]:
        '''
        Convert trajectory of Pvec (int(nx*(nx+1)/2), N+1) to list of P matrices [P0, P1, ..., PN] with each Pi of shape (nx, nx).
        '''
        P_list = [vecToSymm(Pvec[:,k], self.ocp.nx) for k in range(Pvec.shape[1])]
        return P_list
    
    def Kvec_traj_to_K_list(self, Kvec: np.ndarray) -> list[np.ndarray]:
        '''
        Convert trajectory of Kvec (nu*nx, N) to list of K matrices [K0, K1, ..., K(N-1)] with each Ki of shape (nu, nx).
        '''
        K_list = [np.reshape(Kvec[:,k], (self.ocp.nu, self.ocp.nx), order='F') for k in range(Kvec.shape[1])]
        return K_list

    def K_list_to_Kvec_traj(self, K_list: List[np.ndarray]) -> ca.DM:
        '''
        Convert list of K matrices [K0, K1, ..., K(N-1)] with each Ki of shape (nu, nx) to trajectory of Kvec (nu*nx, N).
        '''
        Kvec = ca.DM([K_list[k].flatten(order='F') for k in range(len(K_list))]).T
        return Kvec

    def mu_to_eta(self, mu: np.ndarray, backoff: np.ndarray) -> np.ndarray:
        return .5 * mu / backoff

    def solver_info_str(self) -> str:
        solver_str = ""
        if self.options.compute_riccati:
            solver_str += "Riccati-"

        solver_str += "ZORO"
        if self.options.compute_riccati:
            if self.options.riccati_use_const_cost:
                solver_str += " (constant cost)"
            if self.options.riccati_constraint_cost == RiccatiConstraintCost.HESSIAN_LOG:
                solver_str += " (log-barrier hessian)"
        if self.options.compute_riccati and self.options.riccati_constraint_cost == RiccatiConstraintCost.LAGRANGE_MULTIPLIERS:
            solver_str = "SIRO"
        if self.options.gradient_correction:
            solver_str += " with gradient correction"
        return solver_str
    