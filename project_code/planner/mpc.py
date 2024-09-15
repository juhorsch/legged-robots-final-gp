import numpy as np
import scipy as sp
import osqp
import time
from typing import Optional

class MPC:
    def __init__(self,
                 xk: np.ndarray,
                 Xd: np.ndarray,
                 N: int, 
                 Ak: np.ndarray, 
                 Bk: np.ndarray, 
                 Q: np.ndarray, 
                 R: np.ndarray, 
                 QT: Optional[np.ndarray] = None,  
                 umin: Optional[np.ndarray] = None, 
                 umax: Optional[np.ndarray] = None, 
                 xmin: Optional[np.ndarray] = None, 
                 xmax: Optional[np.ndarray] = None, 
                 C_ext: Optional[np.ndarray] = None, 
                 l_ext: Optional[np.ndarray] = None, 
                 h_ext: Optional[np.ndarray] = None,
                 verbose: Optional[bool] = False,
                 measure_runtime: Optional[bool] = False) -> None:
        """
        Initialize the MPC Solver with system dynamics, constraints, and cost matrices.
        External constraints can be given for the entire prediction horizon.
        System dynamics can either be given as a single matrix or as a 3D array of matrices 
        for each time step. In either case, operation mode cannot be changed after initialization.
        The cost function is assumed to be J = sum(x'Qx + u'Ru) over the prediction horizon.
        Usage: init -> solve -> update -> solve -> ...
        
        :param xk: current system state (1xnx)
        :param Xd: reference trajectory (Nxnx)
        :param N: prediction horizon
        :param Ak: discrete time system matrix (nx x nx) or (N x nx x nx)
        :param Bk: discrete time input matrix (nx x nu)
        :param Q: stage state cost matrix (nx x nx)
        :param R: stage input cost matrix (nu x nu)
        :param QT: terminal state cost matrix
        :param umin: minimum input constraint (nu x 1)
        :param umax: maximum input constraint (nu x 1)
        :param xmin: minimum state constraint (nx x 1)
        :param xmax: maximum state constraint (nx x 1)
        :param C_ext: External constraint matrix (. x nu) or (. x N*nu)
        :param l_ext: lower bound of external constraints (should match C_ext)
        :param h_ext: upper bound of external constraints (should match C_ext)
        :param verbose: print OSQP solver output
        :param measure_runtime: measure the runtime of MPC from update to solve
        """

        self.xk = xk
        self.Xd = Xd.flatten()
        self.N = N
        self.Ak = Ak
        self.Bk = Bk
        self.Q = Q
        self.R = R
        self.QT = QT
        self.umin = umin
        self.umax = umax
        self.xmin = xmin
        self.xmax = xmax
        self.C_ext = C_ext
        self.l_ext = l_ext
        self.h_ext = h_ext
        self.verbose = verbose
        self.measure_runtime = measure_runtime
        self.runtime_ns = None
        self.timestamp = None
        self.Qbar = None
        self.Rbar = None

        self.time_variant = (len(self.Ak.shape) == 3)

        self.nx = Ak.shape[1]
        self.nu = Bk.shape[1]

        if self.measure_runtime:
            self.timestamp = time.time_ns()
        self._evaluate_constraint_types()
        self._formulate_QP()
        self._compute_constraints()
        self._setup_optimizer()

    def _evaluate_constraint_types(self) -> None:
        """Evaluate the types of constraints present in the problem."""
        self.has_input_constraints = self.umin is not None or self.umax is not None
        self.has_state_constraints = self.xmin is not None or self.xmax is not None
        self.has_external_constraints = self.C_ext is not None
        
    def _formulate_QP(self) -> None:
        """Compute the prediction matrices M, C, Qbar, and Rbar for the entire prediction horizon."""

        # Construct prediction matrices X = M@xk + C@U
        self.M = np.zeros((self.nx*self.N, self.nx))
        self.C = np.zeros((self.nx*self.N, self.nu*self.N))
        for i in range(self.N, 0, -1):
            if not self.time_variant:
                self.M[(i-1)*self.nx:i*self.nx, :] = np.linalg.matrix_power(self.Ak, i)
            else:
                temp = np.eye(self.nx)
                for k in range(i):
                    temp = self.Ak[k] @ temp
                self.M[(i-1)*self.nx:i*self.nx, :] = temp
            if i == self.N:
                for j in range(self.N):
                    if not self.time_variant:
                        self.C[(self.N-1)*self.nx:self.N*self.nx, j*self.nu:(j+1)*self.nu] = (
                               np.linalg.matrix_power(self.Ak, self.N-j-1)@self.Bk)
                    else:
                        temp2 = np.eye(self.nx)
                        for z in range(self.N-j-1):
                            temp2 = self.Ak[z] @ temp2
                        self.C[(self.N-1)*self.nx:self.N*self.nx, j*self.nu:(j+1)*self.nu] = temp2 @ self.Bk
                        
            else:
                # reuse lower block row
                self.C[(i-1)*self.nx:i*self.nx, 0:i*self.nu] = (
                    self.C[i*self.nx:(i+1)*self.nx, self.nu:(i+1)*self.nu])
        
        if self.Qbar is None:
            # Concatenate cost matrices
            self.Qbar = np.kron(np.eye(self.N), self.Q)
            self.Rbar = np.kron(np.eye(self.N), self.R)
            
            # Replace last Q with QT
            if self.QT is not None:
                self.Qbar[-self.nx:, -self.nx:] = self.QT
        
        # to minimize 1/2 * Up.T @ P @ Up + q.T @ Up
        self.P = 2*self.C.T @ self.Qbar @ self.C + self.Rbar
        self.M_xk = self.M @ self.xk
        self.q = 2 * ((self.M_xk - self.Xd).T @ self.Qbar @ self.C).T

        # Convert to sparse matrix for OSQP
        self.Ps = sp.sparse.csc_matrix(self.P)

    def _compute_input_constraints(self) -> None:
        umin = -np.inf*np.ones(self.nu) if self.umin is None else self.umin
        umax = np.inf*np.ones(self.nu) if self.umax is None else self.umax
        self.l_u = np.tile(umin, self.N)
        self.h_u = np.tile(umax, self.N)
        self.A_u = np.eye(self.N * self.nu)

    def _compute_state_constraints(self) -> None:
        xmin = -np.inf*np.ones(self.nx) if self.xmin is None else self.xmin
        xmax = np.inf*np.ones(self.nx) if self.xmax is None else self.xmax
        self.l_x = np.tile(xmin, self.N) - self.M_xk
        self.h_x = np.tile(xmax, self.N) - self.M_xk
        self.A_x = self.C.copy()

    def _compute_external_constraints(self) -> None:
        # repeat if not whole matrix is given
        if self.C_ext.shape[1] != self.N*self.nu:
            temp = np.zeros((self.C_ext.shape[0]*self.N, self.C_ext.shape[1])*self.N)
            for i in range(self.N):
                temp[i*self.C_ext.shape[0]:(i+1)*self.C_ext.shape[0], 
                     i*self.C_ext.shape[1]:(i+1)*self.C_ext.shape[1]] = self.C_ext
            self.C_ext = temp
            self.l_ext = np.tile(self.l_ext, self.N)
            self.h_ext = np.tile(self.h_ext, self.N)

    def _compute_constraints(self) -> None:
        """Compute the constraint matrices A, l, and h for the entire prediction horizon.
        Up is subject to l <= A@Up <= h."""

        self.l = []
        self.h = []
        self.A = []

        if self.has_input_constraints:
            self._compute_input_constraints()
            self.A.append(self.A_u)
            self.l.append(self.l_u)
            self.h.append(self.h_u)
        
        if self.has_state_constraints:
            self._compute_state_constraints()
            self.A.append(self.A_x)
            self.l.append(self.l_x)
            self.h.append(self.h_x)

        if self.has_external_constraints:
            self._compute_external_constraints()
            self.A.append(self.C_ext)
            self.l.append(self.l_ext)
            self.h.append(self.h_ext)
            
        if self.A != []:
            self.A = np.vstack(self.A)
            self.l = np.hstack(self.l)
            self.h = np.hstack(self.h)
            self.As = sp.sparse.csc_matrix(self.A)
        else:
            self.l = None
            self.h = None
            self.As = None
    
    def update(self, xk:Optional[np.ndarray] = None,
                Xd: Optional[np.ndarray] = None,
                Ak: Optional[np.ndarray] = None, 
                Bk: Optional[np.ndarray] = None, 
                umin: Optional[np.ndarray] = None, 
                umax: Optional[np.ndarray] = None, 
                xmin: Optional[np.ndarray] = None, 
                xmax: Optional[np.ndarray] = None, 
                C_ext: Optional[np.ndarray] = None, 
                l_ext: Optional[np.ndarray] = None, 
                h_ext: Optional[np.ndarray] = None) -> None:
        """
        efficiently update only necessary params of the MPC solver.
        Important: setting a param to None will keep the previous 
        value and will not actually set it to None. In the case of 
        constraint bounds, set them to float -np.inf or np.inf to remove them,
        and set them to None to keep previous constraints.
        """

        if self.measure_runtime:
            self.timestamp = time.time_ns()
        
        self.xk = xk if xk is not None else self.xk
        self.Xd = Xd.flatten() if Xd is not None else self.Xd
        self.umin = umin if umin is not None else self.umin
        self.umin = None if np.all(umin == -np.inf) else self.umin
        self.umax = umax if umax is not None else self.umax
        self.umax = None if np.all(umax == np.inf) else self.umax
        self.xmin = xmin if xmin is not None else self.xmin
        self.xmin = None if np.all(xmin == -np.inf) else self.xmin
        self.xmax = xmax if xmax is not None else self.xmax
        self.xmax = None if np.all(xmax == np.inf) else self.xmax
        self.C_ext = C_ext if C_ext is not None else self.C_ext
        self.l_ext = l_ext if l_ext is not None else self.l_ext
        self.h_ext = h_ext if h_ext is not None else self.h_ext

        self._evaluate_constraint_types()

        if (Ak is not None and Bk is not None):
            self.Ak = Ak
            self.Bk = Bk
            self._formulate_QP()
            if (xmin is None and xmax is None and umin is None and umax is None 
                and C_ext is None and l_ext is None and h_ext is None and not self.has_state_constraints):
                self._setup_optimizer()
            else:
                self._compute_constraints()
                # self.optimizer.update(Px=self.Ps.data)
                self._setup_optimizer()

        elif (xmin is not None or xmax is not None):
            self.M_xk = self.M@self.xk
            self.q = 2 * ((self.M_xk - self.Xd).T @ self.Qbar @ self.C).T
            self._compute_constraints()
            # self.optimizer.update(q=self.q, Ax=self.As.data, l=self.l, u=self.h)
            self._setup_optimizer()
        elif (umin is not None or umax is not None or C_ext is not None
               or l_ext is not None or h_ext is not None):
            self._compute_constraints()
            # self.optimizer.update(Ax=self.As.data, l=self.l, u=self.h)
            self._setup_optimizer()

        elif (xk is not None or Xd is not None):
            self.M_xk = self.M@self.xk
            self.q = 2 * ((self.M_xk - self.Xd).T @ self.Qbar @ self.C).T
            if self.has_state_constraints:
                self._compute_constraints()
                # self.optimizer.update(q=self.q, Ax=self.As.data, l=self.l, u=self.h)
                self._setup_optimizer()

            else:
                # self.optimizer.update(q=self.q)
                self._setup_optimizer()
        else:
            print("[MPC] Invalid update arguments!")

    def _setup_optimizer(self) -> None:
        """Setup the OSQP optimizer once during initialization."""
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(self.Ps, self.q, A=self.As, l=self.l, u=self.h, verbose=self.verbose)
    
    def solve(self) -> np.ndarray:
        """
        Solve the MPC problem to get the optimal control input trajectory.
        
        :return: Up: predicted optimal control input trajectory (N x nu)
        """
        self.optimization_result = self.optimizer.solve()
        self.Up = self.optimization_result.x

        if (self.optimization_result.info.status != "solved"):
            print("[MPC] OSQP could not reach optimal solution!")
            print("[MPC] OSQP Status: ", self.optimization_result.info.status)
            return None
        
        if self.measure_runtime:
            self.runtime_ns = (time.time_ns() - self.timestamp)

        return self.Up.reshape((self.N, self.nu))
    
    def get_predicted_state(self) -> np.ndarray:
        """
        Get the predicted state trajectory. (N x nx)
        """
        self.Xp = self.M @ self.xk + self.C @ self.Up
        return self.Xp.reshape((self.N, self.nx))
    
    def get_predicted_cost(self) -> float:
        """
        Get the predicted cost of the current state and input trajectory.
        """
        return (self.Xp - self.Xd).T @ self.Qbar @ (self.Xp - self.Xd) + self.Up.T @ self.Rbar @ self.Up
    
    def get_runtime(self) -> float:
        """
        Get nanoseconds between update and solve calls.
        """
        return self.runtime_ns