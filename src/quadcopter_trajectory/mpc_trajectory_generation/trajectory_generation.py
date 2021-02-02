import numpy as np 
from matplotlib import pyplot as plt
from matplotlib import rc

from quadcopter_trajectory.mpc_trajectory_generation.poly_solver import Polytrajectory
from quadcopter_trajectory.mpc_trajectory_generation.mpc_solver import MPCTrajectory
from quadcopter_trajectory.mpc_trajectory_generation.trajectory_elements import *
from quadcopter_trajectory.mpc_trajectory_generation.utils import *

from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
import std_srvs.srv
from quadcopter_trajectory_msg.msg import QuadcopterTrajectory, QuadcopterState

POSITION = 0 
VELOCITY = 1 
ACCELERATION = 2
JERK = 3
SNAP = 4

rc('text', usetex=True)
state_label = ['x [m]','y [m]','z [m]','vx [m.s-1]','vy [m.s-1]','vz [m.s-1]','wx [rad.s-1]','wy [rad.s-1]','wz [rad.s-1]','x [m]','y [rad]','z [rad]']
#Ajouter des fonction de verifiaction de la trajectoire X
#Ajouter des securites
#Ajouter un fonction de get msg ros X
#Ajouter une fonction de get ref 


class Trajectory_generation(object):
    def __init__(self, setpoint, dt, obstacles = [], arena_lb = [-2, -2, 0.0], arena_ub = [2, 2, 2]):

        self._dt = dt
        self.arena_ub = arena_ub
        self.arena_lb = arena_lb

        self.setpoint = setpoint
        self.waypoints = []
        self.obstacles = []
        self.trajectory = []

        self.t_fine = []
        self.t_total = 0.0

        self.ax_states = None
        self.ax_3d = None

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
        self.t_total = sum(self.time_vect)

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
            self.ax_3d = self._build_polytrajectory_3Daxes()
            self.ax_states = self._build_polytrajectory_state_axes(self.ax_states)

            self.ax_3d.legend()
            plt.draw()
            plt.pause(0.01)
    
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
        for x in x_solve : 
            state = np.array([x_i[0] for x_i in np.array(x)])
            self.trajectory.append(state)

        if display == True:
            self.ax_3d = self._build_mpctrejectory_3DAxes(x_solve, self.ax_3d)

            self.ax_3d.legend()
            self.ax_states = self._build_mpctrejectory_state_axes(x_solve, self.ax_states)
            plt.draw()
            plt.show()

    def get_reference_mpc(self, id_0, horizon, states = 'Full'):
        output_reference = np.array([])
        for idx in range(id_0, id_0 + horizon +1):
            if idx < len(self.trajectory):
                output_reference = np.concatenate((output_reference, np.array(self.trajectory[idx])))
            else : 
                output_reference = np.concatenate((output_reference, np.array(self.trajectory[-1])))

        return output_reference

    def build_ros_message(self):

        trajectory = QuadcopterTrajectory()
    
        trajectory.size = len(self.trajectory)
        pathHead = Header()
        trajectory.header = pathHead

        for idx, pt in enumerate(self.trajectory):
            position = Point()
            position.x, position.y, position.z = pt[:3]

            velocity = Point()
            velocity.x, velocity.y, velocity.z = pt[3:6]

            rate = Point()
            rate.x, rate.y, rate.z = pt[6:9]

            quaternion = Quaternion()
            quaternion.x, quaternion.y, quaternion.z, quaternion.w = euler_to_quaternion(pt[9], pt[10], pt[11])

            state = QuadcopterState()
            state.position = position
            state.attitude = quaternion
            state.velocity = velocity
            state.rates = rate
            
            trajectory.states.append(state)
        
        return trajectory
    
    def build_viz_message(self):
        trajectory_viz = MarkerArray()

        for idx, pt in enumerate(self.trajectory):
            pts_marker = Marker()

            pathHead = Header()
            pathHead.frame_id = 'map'
            pts_marker.header = pathHead

            pts_marker.type = 2
            pts_marker.action = 0
            pts_marker.id = idx

            pt_viz = Pose()
            
            position = Point()
            position.x, position.y, position.z = pt[:3]

            quaternion = Quaternion()
            quaternion.x, quaternion.y, quaternion.z, quaternion.w = euler_to_quaternion(pt[9], pt[10], pt[11])

            pt_viz.orientation = quaternion
            pt_viz.position = position
            pts_marker.pose = pt_viz

            pts_marker.scale.x = 0.05
            pts_marker.scale.y = 0.05
            pts_marker.scale.z = 0.05
            pts_marker.color.a = 1.0
            pts_marker.color.r = 0.0
            pts_marker.color.g = 1.0
            pts_marker.color.b = 0.0

            trajectory_viz.markers.append(pts_marker)

        return trajectory_viz

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
        v.start(self.setpoint[0], self._max_derivative_to_optimize)
        vertices.add_start(v)
        v = Vertex(self._poly_dimensions)
        v.end(self.setpoint[len(self.setpoint)-1], self._max_derivative_to_optimize)
        vertices.add_end(v)

        for pts in range(1, len(self.setpoint)-1):
            v = Vertex(self._poly_dimensions)
            v.add_constrain(POSITION, self.setpoint[pts])
            vertices.append(v)

        return vertices

    def _sample_polytrajectory(self):

        time_stamp = [sum(self.time_vect[:i]) for i in range(len(self.time_vect) + 1)]
        self.t_fine = np.array([self._dt * i for i in range(int(ma.ceil(time_stamp[-1]/self._dt)))])

        traj_fine = self._poly_trajectory.eval(self.t_fine, POSITION)
        vel_fine = self._poly_trajectory.eval(self.t_fine, VELOCITY)

        return traj_fine, vel_fine, len(self.t_fine)

    def _build_polytrajectory_3Daxes(self):

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        x_ = []
        y_ = []
        z_ = []

        for obs in self.obstacles:
            ax = obs.show_3d(ax)

        for vertex in self.vertices:
            pt = vertex.get_constraint(symbols.POSITION)
            ax.scatter(pt[0], pt[1], pt[2], color='r', marker='o', label='Setpoint' if vertex.id == 0 else '')

        for wp in self.waypoints:
            x_.append(wp.pos[0])
            y_.append(wp.pos[1])
            z_.append(wp.pos[2])

        ax.plot(x_,y_,z_, '--', color='orange', label='Polytrajectory') 

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title('Trajectory')

        ax.set_zlim3d(self.arena_lb[2], self.arena_ub[2])                   
        ax.set_ylim3d(self.arena_lb[1], self.arena_ub[1])          
        ax.set_xlim3d(self.arena_lb[0], self.arena_ub[0]) 

        return ax

    def _build_polytrajectory_state_axes(self, gs = None):
        if gs is None :
            fig = plt.figure(figsize=(15,15))
            fig.suptitle('States', fontsize=20) 
            gs = fig.subplots(nrows=3, ncols=4)

        for col in range(2):
            for row in range(3):
                title = r'$x^{('+str(col%2)+')}$' if row == 0 else ""
                x_label = r'$T [s]$' if row == 2 else ""
                sub_ax = gs[row, col]
                sub_ax.set_xlabel(x_label)
                sub_ax.set_title(title)

                if col == 0 : 
                    sub_ax.plot(self.t_fine[:-1], [wp.pos[row] for wp in self.waypoints],'--', color='orange', label=state_label[(col*3 + row)])
                else :
                    sub_ax.plot(self.t_fine[:-1], [wp.vel[row] for wp in self.waypoints],'--', color='orange', label=state_label[(col*3 + row)])
                
                lines = sub_ax.lines
                max_val = []
                min_val = []

                for idx, line in enumerate(lines):
                    max_val.append(np.max(line.get_ydata()))
                    min_val.append(np.min(line.get_ydata()))
                    
                if lines != []:
                    sub_ax.set_ylim(np.clip(min(min_val)-0.5, -100, 0.5),np.clip(max(max_val)+0.5, 0.5, 100))
                    sub_ax.legend(prop={'size': 6})           
    
        return gs

    def _build_mpctrejectory_3DAxes(self, x_solve, ax = None):
        if ax is None :
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            for vertex in self.vertices:
                pt = vertex.get_constraint(symbols.POSITION)
                ax.scatter(pt[0], pt[1], pt[2], color='r', marker='o', label='Setpoint' if vertex.id == 0 else '')  

            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_zlabel(r'$z$')
            ax.set_title('Trajectory')

            ax.set_zlim3d(self.arena_lb[2], self.arena_ub[2])                   
            ax.set_ylim3d(self.arena_lb[1], self.arena_ub[1])          
            ax.set_xlim3d(self.arena_lb[0], self.arena_ub[0])            

        ax.plot([pt[0] for pt in x_solve], [pt[1] for pt in x_solve], [pt[2] for pt in x_solve], 'b', label = 'MPC Trajectory')
    
        return ax

    def _build_mpctrejectory_state_axes(self, x_solve, gs = None):
        if gs is None :
            fig = plt.figure(figsize=(15,15))
            fig.suptitle('States', fontsize=20) 
            gs = fig.subplots(nrows=3, ncols=4)


        for col in range(4):
            for row in range(3):
                title = r'$x^{('+str(col%2)+')}$' if row == 0 else ""
                x_label = r'$T [s]$' if row == 2 else ""
                sub_ax = gs[row, col]
                sub_ax.set_xlabel(x_label)
                sub_ax.set_title(title)
                sub_ax.plot(self.t_fine, [pt[(col*3 + row)] for pt in x_solve], label=state_label[(col*3 + row)])

                lines = sub_ax.lines
                max_val = []
                min_val = []

                for idx, line in enumerate(lines):
                    max_val.append(np.max(line.get_ydata()))
                    min_val.append(np.min(line.get_ydata()))
                    
                if lines != []:
                    sub_ax.set_ylim(np.clip(min(min_val)-0.5, -100, 0.5),np.clip(max(max_val)+0.5, 0.5, 100))
                    sub_ax.legend(prop={'size': 6})           
    
        return gs
    
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

