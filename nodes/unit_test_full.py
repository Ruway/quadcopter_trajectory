from quadcopter_trajectory.mpc_trajectory_generation.trajectory_generation import Trajectory_generation
import numpy as np

# setPoints = [[0, 0, 0], [2.0, -1.0, 2.0], [5.0, 3.0, 4.0], [6.0, 5.0, 5.5], [7.0, -5, 5]]
# setPoints = [[0.0, -2.0, 0.0], [0, 0, 1], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],  [1.0, 2.0, 1.0],  [1.0, 3.0, 2.0]]
setPoints = [[0.0, 0.0, 0.0], [0, 0, 1.0], [0, 0, 2.0], [0.0, 0.0, 3.0]]
setPoints = [[0.0, 0.0, 1.0], [1.0, 0, 2.0], [2.0, 0, 3.0]]
# setPoints = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]]
# setPoints = [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
# setPoints = [[0.0, 0.0, 0.0], [1.0, 0, 0], [1.5, 0.5, 0.0], [2.0, 1.0, 0.0]]
setPoints = [[-0.5, 0.0, 0.5], [0.0, 0.0, 0.75], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.5, 0.0, 1.0]]
# setPoints = [[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]]
# setPoints = [[-1.5, -0.5, 0.5],[-0.75, -0.5, 0.5], [0.0, -0.5, 0.5], [0.3, -0.5, 0.83], [0.5, -0.5, 1.2], [1.0, -0.3, 1.2], [1.5, 0.2, 1.2], [1.8, 0.2, 1.2]]
x_0 = np.array(setPoints[0] + [0,0,0,0,0,0] + [0,0,0])

obs = [[0.0, -0.1, 0.2], [1.0, 0.1, 0.2]]


ref_trajectory = Trajectory_generation(setPoints, 0.02, obstacles=obs)
ref_trajectory.build_polytrajectory_solver()
ref_trajectory.generate_polytrajectory(display = False)
ref_trajectory.build_mpc_solver()

ref_trajectory.generate_mpc_trajectory(display = True)