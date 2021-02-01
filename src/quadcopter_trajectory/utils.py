from __future__ import division
import math
import numpy as np
from numpy import linalg as LA
import casadi as ca
import os
import yaml

X = 0
Y = 1
Z = 2

VX = 3
VY = 4
VZ = 5

QX = 9
QY = 10
QZ = 11
QW = 12

WX = 6
WY = 7
WZ = 8

Roll = 9
Pitch = 10
Yaw = 11

DLT_TMPC_PATH = os.path.dirname(os.path.abspath(__file__))
CONTROLLERS_PATH = DLT_TMPC_PATH + '/controllers/'
MODELS_PATH = DLT_TMPC_PATH + '/models/'

def euler_to_quaternion(roll, pitch, yaw):
    # """
    # Perform a conversion from euler angle to quaternion

    # :param roll: roll angle
    # :type roll: float
    # :param pitch: pitch angle
    # :type pitch: float
    # :param yaw: yaw angle 
    # :type yaw: float

    # :return: Quaternion
    # :rtype: list

    # """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    # qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    # qy = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    # qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    # qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def quaternion_to_euler(quaternion):
    # """
    # Perform a conversion from quaternion to euler angle 

    # :param quaternion: quaternion
    # :type quaternion: ca.DM, list, np.array
    # :return: Euler angles [roll, pitch, yaw]
    # :rtype: list of float

    # """
    [x, y, z, w] = quaternion
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return [roll, pitch, yaw]

def q_err(q_t, q_r):
    # """
    # Compute angular error between two unit quaternions.

    # :param q_t: New quaternion
    # :type q_t: ca.DM, list, np.array
    # :param q_r: Reference quaternion
    # :type q_r: ca.DM, list, np.array
    # :return: vector corresponding to SK matrix
    # :rtype: ca.DM

    # """
    q_upper_t = [q_r[3],-q_r[0],-q_r[1],-q_r[2]]
    q_lower_t = [q_t[3],q_t[0],q_t[1],q_t[2]]

    qd_t = [q_upper_t[0]*q_lower_t[0] - q_upper_t[1]*q_lower_t[1] - q_upper_t[2]*q_lower_t[2] - q_upper_t[3]*q_lower_t[3],
            q_upper_t[1]*q_lower_t[0] + q_upper_t[0]*q_lower_t[1] + q_upper_t[2]*q_lower_t[3] - q_upper_t[3]*q_lower_t[2],
            q_upper_t[0]*q_lower_t[2] - q_upper_t[1]*q_lower_t[3] + q_upper_t[2]*q_lower_t[0] + q_upper_t[3]*q_lower_t[1],
            q_upper_t[0]*q_lower_t[3] + q_upper_t[1]*q_lower_t[2] - q_upper_t[2]*q_lower_t[1] + q_upper_t[3]*q_lower_t[0]]

    phi_t   = ca.atan2( 2 * (qd_t[0] * qd_t[1] + qd_t[2] * qd_t[3]), 1 - 2 * (qd_t[1]**2 + qd_t[2]**2) )
    theta_t = ca.asin ( 2 * (qd_t[0] * qd_t[2] - qd_t[3] * qd_t[1]) )
    psi_t   = ca.atan2( 2 * (qd_t[0] * qd_t[3] + qd_t[1] * qd_t[2]), 1 - 2 * (qd_t[2]**2 + qd_t[3]**2) )

    return ca.vertcat(phi_t,theta_t,psi_t)


def read_default_params_mpc(path):
    data = {}
    params = {}

    for i in MPC_PARAMS :
        params[i] = None

    with open(path) as file_d:
        data = yaml.load(file_d, Loader=yaml.FullLoader)
    file_d.close()

    for i in MPC_PARAMS:
        if i in data.keys():
            params[i] = np.array(data[i])

    return params

def read_default_params_quad(path):
    data = {}

    with open(path) as file_d:
        data = yaml.load(file_d, Loader=yaml.FullLoader)
    file_d.close()

    g = data['g']
    m = data['mass']
    Ixx = data['inertia']['xx']
    Iyy = data['inertia']['yy']
    Izz = data['inertia']['zz']
    dynamics = data['dynamic']

    return g, m, Ixx, Iyy, Izz, dynamics


 