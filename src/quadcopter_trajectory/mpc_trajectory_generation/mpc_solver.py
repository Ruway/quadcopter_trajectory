from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import math as ma
import casadi as ca
import casadi.tools as ctools
from dlt_tmpc.tools.utils import *
from scipy.stats import norm
import scipy.linalg
from quadcopter_trajectory.mpc_trajectory_generation.trajectory_elements import *


class MPCTrajectory(object):

    def __init__(self, dynamics, obstacles = [], h = 0.05,
                 horizon=10, Q=None, P=None, R=None, S=None,Sl=None,
                 ulb=None, uub=None, xlb=None, xub=None, terminal_constraint=None,
                 solver_opts=None,
                 x_d=[0]*12,
                ):

        """ Initialize and build the MPC solver
        # Arguments:
            horizon: Prediction horizon in seconds
            model: System model
        # Optional Argumants:
            Q: State penalty matrix, default=diag(1,...,1)
            P: Termial penalty matrix, default=diag(1,...,1)
            R: Input penalty matrix, default=diag(1,...,1)*0.01
            ulb: Lower boundry input
            uub: Upper boundry input
            xlb: Lower boundry state
            xub: Upper boundry state
            terminal_constraint: Terminal condition on the state
                    * if None: No terminal constraint is used
                    * if zero: Terminal state is equal to zero
                    * if nonzero: Terminal state is bounded within +/- the constraint
            solver_opts: Additional options to pass to the NLP solver
                    e.g.: solver_opts['print_time'] = False
                          solver_opts['ipopt.tol'] = 1e-8
        """


        self.horizon = horizon

        build_solver_time = -time.time()
        self.dt = h
        self.Nx, self.Nu = 12, 4 # TODO
        Nopt = self.Nu + self.Nx
        self.Nt = int(horizon)
        self.dynamics = dynamics

        # Initialize variables
        self.set_cost_functions()

        # Cost function weights
        if P is None:
            P = np.eye(self.Nx) * 10
        if Q is None:
            Q = np.eye(self.Nx)
        if R is None:
            R = np.eye(self.Nu) * 0.01
        if S is None:
            S = np.eye(self.Nx) * 1000
        if Sl is None:
            Sl = np.full((self.Nx,), 1000)

        self.Q = ca.MX(Q)
        self.P = ca.MX(P)
        self.R = ca.MX(R)
        self.S = ca.MX(S)
        self.Sl = ca.MX(Sl)

        if xub is None:
            xub = np.full((self.Nx), np.inf)
        if xlb is None:
            xlb = np.full((self.Nx), -np.inf)
        if uub is None:
            uub = np.full((self.Nu), np.inf)
        if ulb is None:
            ulb = np.full((self.Nu), -np.inf)

        self.ulb = ulb
        self.uub = uub
        self.terminal_constraint = terminal_constraint
        # Starting state parameters - add slack here
        x0       = ca.MX.sym('x0', self.Nx)
        x_ref    = ca.MX.sym('x_ref', self.Nx*(self.Nt + 1))
        u0       = ca.MX.sym('u0', self.Nu)
        param_s  = ca.vertcat(x0, x_ref, u0)


        # Create optimization variables
        opt_var = ctools.struct_symMX([(
                ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                ctools.entry('x', shape=(self.Nx,), repeat=self.Nt+1),
                ctools.entry('s', shape=(self.Nx,), repeat=1)
        )])

        self.opt_var = opt_var
        self.num_var = opt_var.size


        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)


        """ Set initial values """
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)

        # Generate MPC Problem
        for t in range(self.Nt):
            
            # Get variables
            x_t = opt_var['x', t]
            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t)
            con_eq.append(x_t_next - opt_var['x',t+1])

            # Input constraints
            if uub is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(uub)
                con_ineq_lb.append(np.full((self.Nu,), -ca.inf))
            if ulb is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(np.full((self.Nu,), ca.inf))
                con_ineq_lb.append(ulb)

            # State constraints
            if xub is not None:
                con_ineq.append(x_t)
                con_ineq_ub.append(xub)
                con_ineq_lb.append(np.full((self.Nx,), -ca.inf))
            if xlb is not None:
                con_ineq.append(x_t)
                con_ineq_ub.append(np.full((self.Nx,), ca.inf))
                con_ineq_lb.append(xlb)

            if obstacles is not None:
                for obs in obstacles:
                    con_ineq.append(self.distance_obs(x_t, obs))
                    con_ineq_ub.append(ca.inf)
                    con_ineq_lb.append(ca.DM((obs.eps + obs.radius)**2))
            
            obj += self.running_cost((x_t-x_ref[t*12:12+t*12]), self.Q, u_t, self.R)

        # Terminal Cost
        obj += self.terminal_cost((opt_var['x', self.Nt] - x_ref[(self.Nt)*12:12+(self.Nt)*12]), self.P)
        
        
        # Terminal contraint

        s_t = opt_var['s', 0]
        con_ineq.append((opt_var['x', self.Nt] - x_ref[(self.Nt)*12:12+(self.Nt)*12])**2 - s_t)
        con_ineq_lb.append(np.full((self.Nx,), 0.0 ))
        con_ineq_ub.append(np.full((self.Nx,),  terminal_constraint**2 ))
        
        obj +=  self.slack_cost(s_t, self.S, self.Sl)

        # Equality constraints bounds are 0 (they are equality constraints),
        # -> Refer to CasADi documentation
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con, 1))
        con_eq_ub = np.zeros((num_eq_con, 1))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver (can also solve QP)
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)

        options = {
            'ipopt.print_level' : 0,
            'ipopt.mu_init' : 0.01,
            'ipopt.tol' : 1e-4,
            'ipopt.sb' : 'yes',
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-3,
            'ipopt.warm_start_bound_frac': 1e-3,
            'ipopt.warm_start_slack_bound_frac': 1e-3,
            'ipopt.warm_start_slack_bound_push': 1e-3,
            'ipopt.warm_start_mult_bound_push': 1e-3,
            'ipopt.mu_strategy' : 'adaptive',
            'print_time' : False,
            'verbose' : False,
            'expand' : True,
            'ipopt.max_iter' : 100
        }


        if solver_opts is not None:
            options.update(solver_opts)

        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, options)
        # self.solver = ca.nlpsol('mpc_solver', 'sqpmethod', nlp, self.sol_options_sqp)

        build_solver_time += time.time()
        
        print('\n________________________________________')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        pass

    def set_cost_functions(self):

        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nx, self.Nx)
        S = ca.MX.sym('S', self.Nx, self.Nx)
        Sl = ca.MX.sym('S', self.Nx)
        R = ca.MX.sym('R', self.Nu, self.Nu)
        P = ca.MX.sym('P', self.Nx, self.Nx)

        x = ca.MX.sym('x', self.Nx)
        s = ca.MX.sym('s', self.Nx)
        u = ca.MX.sym('q', self.Nu)
                                        
        self.running_cost = ca.Function('Jstage', [x, Q, u, R], \
                                      [ca.mtimes(ca.mtimes(x.T, Q), x) + ca.mtimes(ca.mtimes(u.T, R), u)] )

        self.terminal_cost = ca.Function('Jtogo', [x, P], \
                                  [ca.mtimes(ca.mtimes(x.T, P), x)] )
        
        self.slack_cost = ca.Function('Jslack', [s, S, Sl], \
                                    [ca.mtimes(ca.mtimes(s.T, S), s) + ca.mtimes(s.T, Sl)])
                                  

    def distance_obs(self, x, obs):
        return (x[0]-obs.x)**2 + (x[1]-obs.y)**2


    def solve_mpc(self, x0, x_ref, u0=None):
        """ Solve the optimal control problem
        # Arguments:
            x0: Initial state vector.
            sim_time: Simulation length.
        # Optional Arguments:
            x_sp: State set point, default is zero.
            u0: Initial input vector.
            debug: If True, print debug information at each solve iteration.
            noise: If True, add gaussian noise to the simulation.
            con_par_func: Function to calculate the parameters to pass to the
                          inequality function, inputs the current state.
        # Returns:
            mean: Simulated output using the optimal control inputs
            u: Optimal control inputs
        """

        # Initial state
        if u0 is None:
            u0 = np.zeros(self.Nu)

        # Initialize variables
        self.optvar_x0          = np.full((1, self.Nx), x0.T)
        self.optvar_s0          = np.full((1, self.Nx), 0)

        # Initial guess of the warm start variables
        self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        solve_time = -time.time()
        param  = ca.vertcat(x0, x_ref, u0)

        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)

        # Solve NLP
        sol             = self.solver(**args)
        status          = self.solver.stats()['return_status']
        optvar          = self.opt_var(sol['x'])
        solve_time+=time.time()

        return optvar['x'], optvar['u'], solve_time

    def mpc_controller(self, x0, x_ref, u0=None):
        
        x_pred, u_pred, t_solve = self.solve_mpc(x0, x_ref)
        
        return x_pred, u_pred, t_solve



    
