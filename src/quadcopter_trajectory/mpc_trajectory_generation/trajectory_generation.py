import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import rc

from quadcopter_trajectory.mpc_trajectory_generation.poly_solver import Polytrajectory
from quadcopter_trajectory.mpc_trajectory_generation.mpc_solver import MPCTrajectory
from quadcopter_trajectory.mpc_trajectory_generation.trajectory_elements import *
from quadcopter_trajectory.mpc_trajectory_generation.utils import *

POSITION = 0 
VELOCITY = 1 
ACCELERATION = 2
JERK = 3
SNAP = 4

class Trajectory_generation(object):
    def __init__(self, setpoint, dt, obstacles = [], arena_lb = [-2, -2, 0.0], arena_ub = [2, 2, 2]):

        self._setpoint = setpoint
        self._dt = dt
        self.waypoints = []
        self.obstacles = []
        self.arena_ub = arena_ub
        self.arena_lb = arena_lb

        for obs in obstacles:
            self.obstacles.append(Obstacle(obs[0:2], obs[2], arena_ub[2]))

    def build_polytrajectory_solver(self):

        self._poly_dimensions = 3
        self._polynom_degree = 10
        self._max_derivative_to_optimize = 3
        self._weights = np.array([1,1,1])
        self._v_max = 1
        self._eps = 1.0/3.0

        self._poly_trajectory = Polytrajectory(self._polynom_degree, self._poly_dimensions)
        self.vertices = self._build_vertices()

        self.time_vect = self.vertices.estimate_segment_time(self._v_max, self._eps)

        self._poly_trajectory.setup_from_vertice(self.vertices, self.time_vect, ACCELERATION, self._max_derivative_to_optimize, self._weights)

    def generate_polytrajectory(self, display = False):

        self._poly_trajectory.solve()
        pos_fine, vel_fine, self.horizon = self._sample_polytrajectory()
        
        for i in range(self.horizon - 1):
            wp_pos = np.array([pos_fine[0][i], pos_fine[1][i], pos_fine[2][i]])
            wp_npos = np.array([pos_fine[0][i+1], pos_fine[1][i+1], pos_fine[2][i+1]])
            wp_vel = np.array([vel_fine[0][i], vel_fine[1][i], vel_fine[2][i]])
                
            dif_ahead = wp_npos - wp_pos
            wp_psi = np.arctan2(dif_ahead[1], dif_ahead[0])

            self.waypoints.append(Waypoints(wp_pos, wp_vel, wp_psi))

        if display:
            self.ax_3d = self._build_polytrajectory_axes()
    
    def build_mpc_solver(self, default_params = True, dynamics = None, Q = None, P = None, R = None, 
                                                      ulb = None, uub = None, xlb = None, xub = None, xt = None):
        if default_params:
            quadcopter, Q, P, R, ulb, uub, xlb, xub, xt = self._get_default_mpc_params()


        self._mpc_trajectory = MPCTrajectory(dynamics = quadcopter.discrete_nl_dynamics_rk4, h = self._dt, obstacles = self.obstacles, horizon = self.horizon - 1,
                                              Q=Q, P=P, R=R,
                                              ulb=ulb, uub=uub, xlb=xlb, xub=xub, terminal_constraint=xt)

    def generate_mpc_trajectory(self, display = False):
        x0, reference = self._build_reference()
        
        x_solve, u, t_solve = self._mpc_trajectory.mpc_controller(x0, reference)
        self.ax_3d.plot([pt[0] for pt in x_solve], [pt[1] for pt in x_solve], [pt[2] for pt in x_solve])
        
        for obs in self.obstacles:
            obs.show(self.ax_3d)
        
        plt.figure()
        plt.plot([pt[3] for pt in x_solve])
        
        plt.show()

    def _build_reference(self):
        mpc_ref = np.array([])

        for wp in self.waypoints:
            wp_state = np.concatenate((wp.pos,wp.vel))
            mpc_ref = np.concatenate((mpc_ref, np.array(wp_state), np.zeros(6)))

        mpc_ref = np.concatenate((mpc_ref, mpc_ref[-12:]))

        return mpc_ref[:12], mpc_ref        


    def _build_vertices(self):
        vertices = Vertices()

        v = Vertex(self._poly_dimensions)
        v.start(self._setpoint[0], self._max_derivative_to_optimize)
        vertices.add_start(v)
        v = Vertex(self._poly_dimensions)
        v.end(self._setpoint[len(self._setpoint)-1], self._max_derivative_to_optimize)
        vertices.add_end(v)

        for pts in range(1, len(self._setpoint)-1):
            v = Vertex(self._poly_dimensions)
            v.add_constrain(POSITION, self._setpoint[pts])
            vertices.append(v)

        return vertices

    def _sample_polytrajectory(self):

        time_stamp = [sum(self.time_vect[:i]) for i in range(len(self.time_vect) + 1)]
        self.t_fine = np.array([self._dt * i for i in range(int(ma.ceil(time_stamp[-1]/self._dt)))])

        traj_fine = self._poly_trajectory.eval(self.t_fine, POSITION)
        vel_fine = self._poly_trajectory.eval(self.t_fine, VELOCITY)

        return traj_fine, vel_fine, len(self.t_fine)

    def _build_polytrajectory_axes(self):
        rc('text', usetex=True)
        fig = plt.figure()
        
        ax = fig.gca(projection='3d')
        x_ = []
        y_ = []
        z_ = []

        for vertex in self.vertices:
            pt = vertex.get_constraint(symbols.POSITION)
            ax.scatter(pt[0], pt[1], pt[2], color='r', marker='o',)

        for wp in self.waypoints:
            x_.append(wp.pos[0])
            y_.append(wp.pos[1])
            z_.append(wp.pos[2])

        ax.plot(x_,y_,z_, '--', color='orange') 

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title('PolyTrajectory')

        ax.set_zlim3d(0, 2)                   
        ax.set_ylim3d(-2, 2)          
        ax.set_xlim3d(-2, 2)

        return ax

    def _get_default_mpc_params(self):
        quadcopter, mpc_params = read_default_params(self._dt)
        
        Q = np.diag(mpc_params['Q'])
        P = np.diag(mpc_params['Q']) * int(mpc_params['Qn_mult'])
        R = np.diag(mpc_params['R'])
        ulb = np.array(mpc_params['ulb'])
        uub = np.array(mpc_params['uub'])
        xlb = np.array(mpc_params['xlb'])
        xub = np.array(mpc_params['xub'])
        xt = np.array(mpc_params['xt'])

        return quadcopter, Q, P, R, ulb, uub, xlb, xub, xt

