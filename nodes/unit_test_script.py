from __future__ import division

from dlt_tmpc_trajectory.poly_trajectory_generation.vertex import Vertices, Vertex
from dlt_tmpc_trajectory.poly_trajectory_generation.trajectoryGen import Trajectory
from dlt_tmpc_trajectory.poly_trajectory_generation.symbols import *
import numpy as np
import matplotlib.pyplot as plt

# setPoints = [[0, 0, 0], [2.0, -1.0, 2.0], [5.0, 3.0, 4.0], [6.0, 5.0, 5.5], [7.0, -5, 5]]
setPoints = [[0.0, -2.0, 0.0], [0, 0, 1], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],  [1.0, 2.0, 1.0],  [1.0, 3.0, 2.0]]
# setPoints = [[0.0, 0.0, 0.0], [0, 0, 1.5], [0.0, 0.0, 3.0]]
# setPoints = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]
setPoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
# setPoints = [[0.0, 0.0, 0.0], [1.0, 0, 0], [1.5, 0.5, 0.0], [2.0, 1.0, 0.0]]

dimensions = 4
polynom_degree = 10
max_derivative_to_optimize = 3
weights = np.array([1,1,1])
v_max = 4
eps = 1.0/3.0

trajectory = Trajectory(polynom_degree, dimensions)
vertices = trajectory.setup_trajectory(setPoints, dimensions, max_derivative_to_optimize)

time_vect = vertices.estimate_segment_time(v_max, eps)
time_stamp = [sum(time_vect[:i]) for i in range(len(time_vect) + 1)]

trajectory.setup_from_vertice(vertices, time_vect, ACCELERATION, 4, weights)
trajectory.solve()
trajectory.sample_trajectory(0.05)

print("\nTrajectory ready. Press enter to continue.")

trajectory.showTraj(4)
fig_title ='Title'
trajectory.showPath(fig_title)

plt.show()