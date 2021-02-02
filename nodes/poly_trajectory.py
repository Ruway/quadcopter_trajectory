#!/usr/bin/env python
from __future__ import division
import numpy as np
import time as t
import scipy
import rospy
from quadcopter_trajectory.mpc_trajectory_generation.trajectory_generation import Trajectory_generation

from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
import std_srvs.srv
from quadcopter_trajectory_msg.msg import QuadcopterTrajectory, QuadcopterState


class Poly_trajectory(object):

    def __init__(self):
        target = rospy.get_param('~target_path')
        target_obs = rospy.get_param('~target_obs')
        self.dt = rospy.get_param('~time_step')

        self._setPoints = rospy.get_param('~trajectory/setPoints/'+target)
        self._obstacles = rospy.get_param('~trajectory/obstacles/'+target_obs)
        self._eps = float(rospy.get_param('~trajectory/eps'))
        self._v_max = float(rospy.get_param('~trajectory/v_max'))
        
        # self._setPoints = [[-0.5, 0.0, 0.5], [0.0, 0.0, 0.75], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0], [1.5, 0.0, 1.0]]
        # self._obstacles = [[0.0, -0.1, 0.2], [1.0, 0.1, 0.2]]
        self.build_trajectory()

        pass

    # ---------------------------------
    # BEGIN: Callbacks Section
    # ---------------------------------

    def start_srv_callback(self, req=std_srvs.srv.SetBoolRequest()):
        self.start = req.data

        ans = std_srvs.srv.SetBoolResponse()
        ans.success = True
        ans.message = "Node started!"

        return ans

    # ---------------------------------
    # END: Callbacks Section
    # ---------------------------------

    def set_subscribers_publishers(self):

        # # Publishers
        self.trajectory_pub = rospy.Publisher("~Trajectory",
                                           QuadcopterTrajectory,
                                           queue_size=1,
                                           latch = True)

        self.trajectory_viz_pub = rospy.Publisher("~Trajectory_viz",
                                           MarkerArray,
                                           queue_size=1,
                                           latch = True)

        pass    
    
    def build_trajectory(self):
        self.ref_trajectory = Trajectory_generation(self._setPoints, self.dt, self._obstacles)
        self.ref_trajectory.build_polytrajectory_solver()
        self.ref_trajectory.generate_polytrajectory()
        self.ref_trajectory.build_mpc_solver()

        self.ref_trajectory.generate_mpc_trajectory()

    def run(self):
        # while not rospy.is_shutdown():
        self.trajectory_pub.publish(self.ref_trajectory.build_ros_message())
        self.trajectory_viz_pub.publish(self.ref_trajectory.build_viz_message())
        print("End")

    pass

if __name__ == "__main__":
    rospy.init_node('poly_trajectory_node')
    
    poly = Poly_trajectory()
    poly.set_subscribers_publishers()
    poly.run()

    rospy.spin()
    pass


