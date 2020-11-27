#!/usr/bin/env python
from __future__ import division
import numpy as np
import time as t
import scipy
import rospy
from dlt_tmpc_trajectory.trajectoryGen import Trajectory
from dlt_tmpc_trajectory.symbols import *

from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
import std_srvs.srv


class Poly_trajectory(object):

    def __init__(self):

        self._dimension = int(rospy.get_param('~/settings/dimension'))
        self._weights = np.array(rospy.get_param('~/settings/weights')) 
        self._polynom_degree = int(rospy.get_param('~/settings/polynom_degree'))
        self._max_derivative_to_optimize = int(rospy.get_param('~/settings/max_derivative_to_optimize'))

        self._setPoints = rospy.get_param('~/trajectory/setPoints/straight_z')
        self._eps = float(rospy.get_param('~/trajectory/eps'))
        self._v_max = float(rospy.get_param('~/trajectory/v_max'))
        
        self.build_trajectory()
        self._trajectory.sample_trajectory(0.05)

        raw_traj = self._trajectory.generate_quad_trajectory()
        time_stamp = self._trajectory.get_time_stamp()
        
        self.build_msg(time_stamp, raw_traj)

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
        self.trajectory_pub = rospy.Publisher("~Path",
                                           Path,
                                           queue_size=1,
                                           latch = True)
        pass    
    
    def build_trajectory(self):
        self._trajectory = Trajectory(self._polynom_degree, self._dimension)
        
        vertices = self._trajectory.setup_trajectory(self._setPoints, self._dimension, self._max_derivative_to_optimize)

        self._time_vect = vertices.estimate_segment_time(self._v_max, self._eps)

        self._trajectory.setup_from_vertice(vertices, self._time_vect, ACCELERATION, self._dimension, self._weights)
        self._trajectory.solve()

    def build_msg(self, time, raw):
        path = Path()
        pathHead = Header()
        pathHead.frame_id = "map"
        path.header = pathHead

        for idx, pt in enumerate(raw):
            point = Point()
            point.x, point.y, point.z = pt[:3]

            quaternion = Quaternion()
            quaternion.x, quaternion.y, quaternion.z, quaternion.w = pt[3:]

            pose = Pose()
            pose.position = point
            pose.orientation = quaternion

            pose_stamped = PoseStamped()
            head = Header()
            head.seq = idx

            pose_stamped.header = head
            pose_stamped.pose = pose
            
            path.poses.append(pose_stamped)
        
        self.path = path

    def run(self):
        # while not rospy.is_shutdown():
        self.trajectory_pub.publish(self.path)
        print("End")

    pass



if __name__ == "__main__":
    rospy.init_node('poly_trajectory_node')
    
    poly = Poly_trajectory()
    poly.set_subscribers_publishers()
    poly.run()

    rospy.spin()
    pass


