from quadcopter_trajectory.models.quadcopter_CA import Quadcopter
from quadcopter_trajectory.mpc_trajectory_generation.traj_mpc import MPC_EULER
from quadcopter_trajectory.poly_trajectory_generation import symbols
from quadcopter_trajectory.poly_trajectory_generation.trajectoryGen import Trajectory
from quadcopter_trajectory.utils import *
import math
import matplotlib.pyplot as plt

dt = 0.02
g, m, Ixx, Iyy, Izz, dynamics = read_default_params_quad(MODELS_PATH + '/config/srd370_2.yaml')
quadcopter = Quadcopter(h = dt, g = g, Ix = Ixx, Iy = Iyy, Iz = Izz, m = m)

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

dimensions = 4
polynom_degree = 10
max_derivative_to_optimize = 3
weights = np.array([1,1,1])
v_max = 1
eps = 1.0/3.0

trajectory = Trajectory(polynom_degree, dimensions)
vertices = trajectory.setup_trajectory(setPoints, dimensions, max_derivative_to_optimize)

time_vect = vertices.estimate_segment_time(v_max, eps)
time_stamp = [sum(time_vect[:i]) for i in range(len(time_vect) + 1)]

trajectory.setup_from_vertice(vertices, time_vect, symbols.ACCELERATION, 3, weights)
trajectory.solve()
pos_fine, vel_fine = trajectory.sample_trajectory(dt)

horizon = math.ceil(time_stamp[-1]/0.02)
#----------MPC matrix----------#
Q = np.eye(12)

Q[0:3,0:3] = np.diag([1,1,1])*100  # position
Q[3:6,3:6] = np.diag([1,1,1])*10 # velocity
Q[6:9,6:9] = np.diag([1,1,1])*0 # angular velocity
Q[8,8] = 0.0  # angular velocity
Q[9:12,9:12] = np.diag([1,1,1])*0 # angles

R = np.eye(4)*0.1
R[1:4,1:4] = np.diag([1,1,1])*1

P = Q*100
#----------------------------------#

#----------MPC constraints----------#

ulb = np.array([0]+[-1.5e-0]*2+[-7.5e-1])
uub = np.array([2*quadcopter.m*quadcopter.g]+[1.5e-0]*2+[7.5e-1])

xlb = np.array([-10]*3+[-10]*3+[-3]*3+[-np.deg2rad(30)]*2+[-np.deg2rad(180)])
xub = np.array([ 10]*3+[ 10]*3+[ 3]*3+[ np.deg2rad(30)]*2+[ np.deg2rad(180)])
xt  = np.array([0.1]*3+[10]*3+[5]*3+[ np.deg2rad(5)]*2+[ np.deg2rad(10)])
xt = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


#----------------------------------#

ctl = MPC_EULER(model=quadcopter, 
          dynamics=quadcopter.discrete_nl_dynamics_rk4,
          horizon=horizon,
          Q = Q , R = R, P = P,
          ulb=ulb, uub=uub, 
          xlb=xlb, 
          xub=xub,
          terminal_constraint=xt,
          )

x_ref, xd_ref = trajectory.get_MPC_traj(0.0, 0.02, horizon)
x_ref = np.array([])
for i in range(horizon + 1):
    if i == horizon:
        x_ref = np.concatenate((x_ref, np.array([pos_fine[0][-1], pos_fine[1][-1], pos_fine[2][-1], vel_fine[0][i-1], vel_fine[1][-1], vel_fine[2][-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))
    else :
        x_ref = np.concatenate((x_ref, np.array([pos_fine[0][i], pos_fine[1][i], pos_fine[2][i], vel_fine[0][i], vel_fine[1][i], vel_fine[2][i], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))

print(len(x_ref))
# x_ref = quadcopter.trajectory_to_reference(x_ref, xd_ref)

x_solve, u, t_solve = ctl.mpc_controller(x_0, x_ref)
print(x_0)
# plt.plot([x_ref[i*12] for i in range(horizon)])
# plt.plot([x_ref[i*12+1] for i in range(horizon)])
# plt.plot([x_ref[i*12+2] for i in range(horizon)])
# plt.plot([x_ref[i*12+3] for i in range(horizon)])
# plt.plot([x_ref[i*12+4] for i in range(horizon)])
plt.plot([x_ref[i*12+5] for i in range(horizon)])
# plt.plot([x_ref[i*12+6] for i in range(horizon)])
# plt.plot([x_ref[i*12+7] for i in range(horizon)])
# plt.plot([x_ref[i*12+8] for i in range(horizon)])
# plt.plot([x_ref[i*12+9] for i in range(horizon)])
# plt.plot([x_ref[i*12+10] for i in range(horizon)])
# plt.plot([x_ref[i*12+11] for i in range(horizon)])
print(x_ref[-12:])
plt.plot([pt[0] for pt in x_solve])
plt.plot([pt[1] for pt in x_solve])
plt.plot([pt[2] for pt in x_solve])
plt.show()

input()
print("\nTrajectory ready. Press enter to continue.")
