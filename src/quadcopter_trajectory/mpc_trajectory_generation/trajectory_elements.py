from __future__ import division
import casadi as ca
import numpy as np
from numpy import linalg as LA
from quadcopter_trajectory.poly_trajectory_generation import symbols
import glog as log
import math as ma

class Vertex():
    def __init__(self, dimension):
        self._dimension = dimension
        self._deriveToOptimize = 0
        self._constraints = {}
        self.id = 0

    def __str__(self):
        return "Vertex "+str(self.id)+" in R^" + str(self._dimension) + ".\nConstraints on " + str(len(list(self._constraints.keys()))) + \
               " derivatives.\n" + str(list(self._constraints.values())) + "\n"

    def add_constrain(self, derivative, constraints):
        self._constraints[derivative] = np.array(constraints)

    def start(self, constraint, up_to_derivative, constraint_v = []):
        if constraint_v == [] :
            constraint_v = [0 for i in range(self._dimension)]
        
        self.add_constrain(symbols.POSITION, constraint)
        self.add_constrain(symbols.VELOCITY, constraint_v)
        for i in range(symbols.ACCELERATION, up_to_derivative):
            self._constraints[i] = np.zeros(self._dimension)
    
    def end(self, constraint, up_to_derivative):
        self.add_constrain(symbols.POSITION, constraint)
        for i in range(symbols.VELOCITY, up_to_derivative):
            self._constraints[i] = np.zeros(self._dimension)
    
    def get_constraints_dict(self):
        return self._constraints
    
    def get_constraint(self, derivative):
        if not derivative in self._constraints.keys():
            return None
        return self._constraints[derivative]

    def get_constraints_number(self):
        return len(list(self._constraints.values()))

    def set_id(self, idx):
        self.id = int(idx)


class Vertices(): 

    def __init__(self):
        self._data = {}

    def __iter__(self):
        for k in self._data.keys():
            yield self._data[k]

    def __getitem__(self, id):
        if id == -1:
            return self._data[len(self._data.keys())-1]
        return self._data[id]

    def size(self):
        return len(self._data.keys())

    def add_start(self, vertex):
        for i,val in enumerate(self._data):
            self._data[i+1] = val

        self._data[0] = vertex

    def append(self, vertex):
        used_keys = list(self._data.keys())

        if len(used_keys) >= 2:
            last = self._data[used_keys[-1]]

            last.set_id(used_keys[-1] + 1)
            vertex.set_id(used_keys[-1])

            self._data[used_keys[-1]] = vertex
            self._data[used_keys[-1] + 1] = last
        else:
            self._data[used_keys[-1] + 1] = vertex
            vertex.set_id(used_keys[-1] + 1)

    
    def add_end(self, vertex):
        used_keys = list(self._data.keys())
        log.check_ne(len(used_keys), 0)

        vertex.set_id(used_keys[-1] + 1)

        self._data[used_keys[-1] + 1] = vertex

    def estimate_segment_time(self, v_max, eps):
        if len(self._data.keys()) < 2 : 
            return None

        estimate_time = []
        for i in range(0,len(self._data.keys()) - 1):

            start = self._data[i].get_constraint(symbols.POSITION)[:]
            stop = self._data[i+1].get_constraint(symbols.POSITION)[:]
            dist = LA.norm(stop -start)
            print(stop, start)
            print(dist)
            estimate_time.append(dist / (eps * v_max))

        return np.array(estimate_time)
    
class Waypoints(object):
    def __init__(self, position, velocity, psi = None):
        self.pos = np.array(position)
        self.vel = np.array(velocity)
        self.psi = psi

        self.uub = np.array([])
        self.ulb = np.array([])
        self.xub = np.array([]) 
        self.xlb = np.array([])

    def init_constraints(self, ulb, uub, xlb, xub):
        self.uub = np.array(uub)
        self.ulb = np.array(ulb)
        self.xub = np.array(xub) 
        self.xlb = np.array(xlb)
        
    def __sub__(self, other):
        return ((self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2)**0.5
        
class Obstacle(object):
    def __init__(self, position, r, h, eps = 0.0):
        x, y = position 
        self.x = x
        self.y = y 
        self.eps = eps
        self.radius = r
        self.height = h

    def show(self, ax):
        height = np.linspace(0, self.height, 100)
        theta = np.linspace(0, 2*3.1459, 100)
        x_fine = []
        y_fine = []
        z_fine = []

        for t in theta:
            for h in height:
                x_fine.append(self.x + self.radius * np.cos(t))
                y_fine.append(self.y + self.radius * np.sin(t))
                z_fine.append(h)
        
        ax.plot(x_fine, y_fine, z_fine, 'k.', markersize = 0.1)

        return ax









    