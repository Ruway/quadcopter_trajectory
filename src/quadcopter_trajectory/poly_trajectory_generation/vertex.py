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
            
            start = self._data[i].get_constraint(symbols.POSITION)[:-1]
            stop = self._data[i+1].get_constraint(symbols.POSITION)[:-1]
            dist = LA.norm(stop -start)

            estimate_time.append(dist / (eps * v_max))

        return np.array(estimate_time)
    
    def set_warm_start(self, v_start, w_start, yaw_0 = 0):
        v_0 = self._data[0]
        v_1 = self._data[1]
        
        p_0 = np.array(v_0.get_constraint(symbols.POSITION)[0:3])
        p_1 = np.array(v_1.get_constraint(symbols.POSITION)[0:3])
        
        start_vel = ((p_1 - p_0) / np.linalg.norm((p_1 - p_0))*v_start).tolist()

        p_0 = (v_0.get_constraint(symbols.POSITION)[3:])[0]
        p_1 = (v_1.get_constraint(symbols.POSITION)[3:])[0]
        
        start_yaw_vel = 0#(p_1 - p_0)/2

        v_0.add_constrain(symbols.VELOCITY, start_vel + [start_yaw_vel])

    def set_yaw_constraints(self, yaw_0 = 0):
        start_set = 0 
        # v_1 = self._data[0].get_constraint(symbols.POSITION)
        # self._data[0].add_constrain(symbols.POSITION, np.append(v_1, yaw_0))
        # start_set = 1

        for v in range(start_set, len(self._data)-1):
            v_1 = self._data[v].get_constraint(symbols.POSITION)
            v_2 = self._data[v+1].get_constraint(symbols.POSITION)
            yaw = self._get_yaw(v_1, v_2)
            self._data[v].add_constrain(symbols.POSITION, np.append(v_1, yaw))

        v_0 = self._data[len(self._data)-2].get_constraint(symbols.POSITION)
        v_1 = self._data[len(self._data)-1].get_constraint(symbols.POSITION)

        self._data[len(self._data)- 1].add_constrain(symbols.POSITION, np.append(v_1, v_0[-1]))
    
    def _get_yaw(self, v_0, v_1):

        vect = ((v_1 - v_0) / np.linalg.norm((v_1 - v_0))).tolist()
        vect[2] = 0
        v_x = np.array([1,0,0])

        unit_vector_1 = v_x / np.linalg.norm(v_x)
        unit_vector_2 = vect / np.linalg.norm(vect)
        angle = np.arccos(np.clip(np.dot(unit_vector_1, unit_vector_2), -1.0, 1.0))

        return angle









    