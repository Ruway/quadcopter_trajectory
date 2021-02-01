#!/usr/bin/env python
# coding=utf-8
from __future__ import division
import math as ma
import numpy as np
from abc import abstractmethod
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from quadcopter_trajectory.poly_trajectory_generation import symbols
from quadcopter_trajectory.poly_trajectory_generation.vertex import Vertex, Vertices
from quadcopter_trajectory import utils
# from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply

class TrajectoryTools(object):
    def __init__(self, dim_,):
        # dimension of curve
        self.dim = dim_
        self.pinSet = None
        # is solved?
        self.isSolved = False
        # time nkots (length = M for polyTrajGen and length 2 for optimalTraj)
        self.weight_mask = None
        # pin sets
        ## fixed
        self.fixPinSet = {}
        self.loosePinSet = {}
        self.fixPinOrder = {}

    @abstractmethod
    def setDerivativeObj(self, weight):
        pass

    @abstractmethod
    def solve(self,):
        pass

    @abstractmethod
    def eval(self, t, d):
        pass

    def sample_trajectory(self, dt):

        self.dt = dt
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]
        self.t_fine = np.array([dt * i for i in range(int(ma.ceil(time_stamp[-1]/dt)))])

        self.traj_fine = self.eval(self.t_fine, symbols.POSITION)
        self.vel_fine = self.eval(self.t_fine, symbols.VELOCITY)
        self.acc_fine = self.eval(self.t_fine, symbols.ACCELERATION)
        return self.traj_fine, self.vel_fine
       
    def generate_quad_trajectory(self):

        quad_trajectory = []

        for idx in range(len(self.t_fine)):

            z = [self.traj_fine[0][idx], self.traj_fine[1][idx], self.traj_fine[2][idx], ma.tan(self.traj_fine[3][idx]/2.0)]
            zd = [self.vel_fine[0][idx], self.vel_fine[1][idx], self.vel_fine[2][idx], self.vel_fine[3][idx]]
            zdd = [self.acc_fine[0][idx], self.acc_fine[1][idx], self.acc_fine[2][idx], self.acc_fine[3][idx]]

            r, p, y = self.flat_representation(z, zdd)

            qx, qy, qz, qw = utils.euler_to_quaternion(r, p, y)
            # qx2, qy2, qz2, qw2 = quaternion_from_euler(r, p, y)
            # r2,p2,y2 = euler_from_quaternion([qx2, qy2, qz2, qw2])
            # r3,p3,y3 = utils.quaternion_to_euler([qx2, qy2, qz2, qw2])

            # print(r,p,y)
            # print(r2,p2,y2)
            # print(r3,p3,y3)
            # print(qx,qy,qz,qw)
            # print(qx2,qy2,qz2,qw2)

            quad_trajectory.append([self.traj_fine[0][idx], self.traj_fine[1][idx], self.traj_fine[2][idx], \
                                    self.vel_fine[0][idx], self.vel_fine[1][idx], self.vel_fine[2][idx], \
                                    0, 0, self.vel_fine[3][idx], \
                                     qx, qy, qz, qw])
        
        return quad_trajectory

    def get_time_stamp(self):
        return self.t_fine

    def showPath(self, fig_title="Path"):

        rc('text', usetex=True)
        fig = plt.figure()
        if self.dim == 3 or self.dim == 4:
            ax = fig.gca(projection='3d')
        else:
            ax = plt.gca(projection='2d')

        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]

        # draw pin set
        for vertex in self.vertices:
            X = vertex.get_constraint(symbols.POSITION)
            if len(X.shape) == 2:
                ## loose pin
                x_ = X_[0, 0]
                y_ = X_[1, 0]
                x_size_ = X_[0, 1] - X_[0, 0]
                y_size_ = X_[1, 1] - X_[1, 0]
                ### 2D
                if self.dim == 2:
                    ax.add_patch(patches.Rectangle((x_, y_), x_size_, y_size_), facecolor='r', alpha=0.1)
                ### 3D cube/bar
                else:
                    z_ = X_[2, 0]
                    z_size_ = X_[2, 1] - X_[2, 0]
                    ax.bar3d(x_, y_, z_, x_size_, y_size_, z_size_, color='r', alpha=0.1)
            else:
                ## fix pin
                ### 2D
                if self.dim == 2:
                    ax.scatter(X[0], X[1], color='b', marker='o',)
                ### 3D
                else:
                    ax.scatter(X[0], X[1], X[2], color='b', marker='o',)
        # draw curve
        if self.isSolved:
            N_plot = 100
            ts = np.linspace(time_stamp[0], time_stamp[-1], N_plot)
            Xs = self.eval(ts, 0)
            if self.dim == 2:
                ax.plot(Xs[0], Xs[1], 'k-')
            else:
                ax.plot(Xs[0], Xs[1], Xs[2], 'k-')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title(fig_title)
        plt.show()

    def showAttitude(self, model):

        ax_dict = {}
        fig_dict = {}
        rc('text', usetex=True)
        fig, axs = plt.subplots(3, 1)
        N_plot = 50
                
        # print pins
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]
        
        ts = np.linspace(time_stamp[0], time_stamp[-1], N_plot)
        if self.isSolved:
            Xangles = self.get_angles(ts, N_plot, model)

            for d in range(3):
                # axs[d, 0].set_title(title_list[d])
                axs[d].plot(ts, Xangles[d], 'k-')
                axs[d].set_xlim(time_stamp[0]-0.1, time_stamp[-1]+0.1)
                axs[d].set_ylim(np.min(Xangles[d]), np.max(Xangles[d]))
                axs[d].set_xlabel('t')
        # plt.show()

    def showTraj(self, plotOrder):
        assert plotOrder>=0, 'Invalid plot order'
        # plt.figure()
        ax_dict = {}
        fig_dict = {}
        rc('text', usetex=True)
        # rc('font', family='serif')
        # create subfigures in (dim, order+1)
        fig, axs = plt.subplots(self.dim, plotOrder+1)
        # print pins
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]
        
        for dd in range(self.dim):
            for vertex in self.vertices:
                constraints = vertex.get_constraints_dict()

                for d in constraints.keys():
                    axs[dd, d].vlines(time_stamp[vertex.id], -10.0, 10, color='k', linestyle='dashed', linewidth=.5)

                    axs[dd, d].scatter(x=time_stamp[vertex.id], y=vertex.get_constraint(d)[dd], color='r', marker='.', )

        # print curves
        title_list = [r'$x^{('+str(i)+')}$' for i in range(plotOrder+1)]

        if self.isSolved:
            for d in range(plotOrder+1):
                N_plot = 50
                ts = np.linspace(time_stamp[0], time_stamp[-1], N_plot)
                Xs = self.eval(ts, d)
                axs[0, d].set_title(title_list[d])
                for dd in range(self.dim):
                    if d > 0:
                        axs[dd, d].hlines(y=0.0, xmin=time_stamp[0]-0.5, xmax=time_stamp[-1]+0.5, colors='r', linestyles='dashed')
                    axs[dd, d].plot(ts, Xs[dd], 'k-')
                    for t_ in time_stamp:
                        axs[dd, d].vlines(x=t_, ymin=np.min(Xs[dd])-0.5, ymax=np.max(Xs[dd])+0.5, color='k', linestyle='dashed', linewidth=0.3)
                    axs[dd, d].set_xlim(time_stamp[0]-0.5, time_stamp[-1]+0.5)
                    axs[dd, d].set_ylim(np.min(Xs[dd])-0.5, np.max(Xs[dd])+0.5)
                    axs[dd, d].set_xlabel('t')
        # plt.show()

    def get_MPC_traj(self, t0, dt, horizon):

        x_ref = []
        xd_ref = []
        errors = []
        Xangles = []
        x_ref0 = self.vertices[-1].get_constraint(symbols.POSITION)
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)]
       
        t0 = int(round(t0/dt))

        tsX = []
        tsV = []
        
        for i in range(t0, t0 + horizon + 1):
            if i >= round(time_stamp[-1]/self.dt):
                tsX.append(int(round(time_stamp[-1]/self.dt)-1))
            else:
                tsX.append(i)
        for i in range(t0, t0 + horizon + 1):
            if i >= round(time_stamp[-1]/self.dt):
                tsV.append(int(round(time_stamp[-1]/self.dt)-1))
            else:
                tsV.append(i)

        roll = []
        pitch = []
        for idx in tsX:

            z = [self.traj_fine[0][idx], self.traj_fine[1][idx], self.traj_fine[2][idx], ma.tan(self.traj_fine[3][idx]/2.0)]
            zdd = [self.acc_fine[0][idx], self.acc_fine[1][idx], self.acc_fine[2][idx], self.acc_fine[3][idx]]

            r, p, y = self.flat_representation(z, zdd)
            roll.append(r)
            pitch.append(p)

        Xs = [list(a) for a in zip(self.traj_fine[0][tsX], self.traj_fine[1][tsX], self.traj_fine[2][tsX], roll, pitch, self.traj_fine[3][tsX])] 

        Xds =[list(a) for a in zip([(self.vel_fine[0][i+1] + self.vel_fine[0][i])/2.0 if i < len(self.t_fine) -1 else 0 for i in (tsV)] , [(self.vel_fine[1][i+1] + self.vel_fine[1][i])/2.0 if i < len(self.t_fine)-1 else 0 for i in (tsV)],\
                                   [(self.vel_fine[2][i+1] + self.vel_fine[2][i])/2.0 if i < len(self.t_fine) -1 else 0 for i in (tsV)], [(self.vel_fine[3][i+1] + self.vel_fine[3][i])/2.0 if i < len(self.t_fine)-1 else 0 for i in (tsV)])] 

        return Xs, Xds


    def get_closest(self, x):

        err = [ma.sqrt(ma.pow(x[0] - self.traj_fine[0][j], 2)+ma.pow(x[1] - self.traj_fine[1][j], 2)+ma.pow(x[2] - self.traj_fine[2][j], 2)) for j in range(len(self.traj_fine[0]))]
        idx_min = np.argmin(err)

        return idx_min 

    def check_eot(self, x, eps):
        time_stamp = [sum(self.time_segment[:i]) for i in range(len(self.time_segment) + 1)][-1]
        x_end = self.eval_one(time_stamp, symbols.POSITION)
        err = ma.sqrt(ma.pow(x[0] - x_end[0], 2)+ma.pow(x[1] - x_end[1], 2)+ma.pow(x[2] - x_end[2], 2))

        return err < eps

    def get_angles(self, ts, N_plot, model):
        Xs = {}
        for d in range(4):
            Xs[d] = self.eval(ts, d)
    
        angles = np.zeros((3,N_plot))
         
        for t in range(N_plot):
            z = [Xs[symbols.POSITION][0][t], Xs[symbols.POSITION][1][t], Xs[symbols.POSITION][2][t], ma.tan(Xs[symbols.POSITION][3][t]/2)]
            zdd = [Xs[symbols.ACCELERATION][0][t], Xs[symbols.ACCELERATION][1][t], Xs[symbols.ACCELERATION][2][t], Xs[symbols.ACCELERATION][3][t]]

            r, p, y = self.flat_representation(z, zdd)

            angles[0,t] = r
            angles[1,t] = p
            angles[2,t] = y

        return angles

    def setup_trajectory(self, setPoints, dimension, max_derivative_to_optimize):
        vertices = Vertices()

        v = Vertex(dimension)
        v.start(setPoints[0], max_derivative_to_optimize)
        vertices.add_start(v)
        v = Vertex(dimension)
        v.end(setPoints[len(setPoints)-1], max_derivative_to_optimize)
        vertices.add_end(v)

        for pts in range(1, len(setPoints)-1):
            v = Vertex(dimension)
            v.add_constrain(symbols.POSITION, setPoints[pts])
            vertices.append(v)

        vertices.set_yaw_constraints()

        for v in vertices: 
            print(v)

        return vertices

    def flat_representation(self, v_z, vdd_z):
        """
        Compute equivalent equivalent roll, pitch, yaw angles based on flat representation.
        from Nguyen's paper, Reliable nonlinear control for quadcopter trajectory tracking through differential flatness.

        :param v_z: flat state [z1, z2, z3, z4] = [x, y ,z tan(yaw/2)]
        :type v_z: np.array or list
        :param vdd_z: double derivative flat state [z1'', z2'', z3'', z4'']
        :type vdd_z: np.array or list
        :return: roll, pitch, yaw
        :rtype: float, float, float
        """
        
        z1, z2, z3, z4 = v_z
        zdd1, zdd2, zdd3, _ = vdd_z 

        roll_num = 2*z4*zdd1 - (1-z4**2)*zdd2
        roll_den = (1+z4**2) * ma.sqrt(zdd1**2 + zdd2**2 + (zdd3 + 9.81)**2)
        roll = ma.asin(roll_num/roll_den)

        pitch_num = (1-z4**2)*zdd1 + 2*z4*zdd2
        pitch_den = (1+z4**2) * (zdd3 + 9.81)
        pitch = ma.atan(pitch_num/pitch_den)

        yaw = 2 * ma.atan(z4)
        
        return roll, pitch, yaw

    
          



