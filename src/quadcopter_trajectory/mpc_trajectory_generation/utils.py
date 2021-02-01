from __future__ import division
import math
import os
import yaml
import numpy as np
from quadcopter_trajectory.models.quadcopter_CA import Quadcopter

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

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = PACKAGE_PATH + '/config/'

def read_default_params(dt):
    data = {}

    with open(CONFIG_PATH + 'default.yaml' ) as file_d:
        data = yaml.load(file_d, Loader=yaml.FullLoader)
    file_d.close()

    g = data['quadcopter']['g']
    mass = data['quadcopter']['mass']
    Ixx = data['quadcopter']['Ixx']
    Iyy = data['quadcopter']['Iyy']
    Izz = data['quadcopter']['Izz']
    quadcopter = Quadcopter(h=dt, g = g, Ix = Ixx, Iy = Iyy, Iz = Izz, m = mass)

    return quadcopter, data['mpc']

def q2w(wp_prev, wp, wp_next, ub, lb):
    dub = np.zeros(2)
    dlb = np.zeros(2)
    
    x, y, _ = wp.pos
    psi = wp.psi

    dub[0] = x - np.sin(psi) * ub if x - np.sin(psi) * ub > wp_next.pos[0] else wp_next.pos[0]
    dub[1] = y + np.cos(psi) * ub if y + np.cos(psi) * ub > wp_next.pos[1] else wp_next.pos[1]

    dlb[0] = x - np.sin(psi) * lb if x - np.sin(psi) * lb < wp_prev.pos[0] else wp_prev.pos[0]
    dlb[1] = y + np.cos(psi) * lb if y + np.cos(psi) * lb < wp_prev.pos[1] else wp_prev.pos[1]

    return dub, dlb

