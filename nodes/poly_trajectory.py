#!/usr/bin/env python
from __future__ import division
import numpy as np
import time as t
import scipy
import rospy
from quadcopter_trajectory.poly_trajectory_generation.trajectoryGen import Trajectory
from quadcopter_trajectory.poly_trajectory_generation.symbols import *

from geometry_msgs.msg import Quaternion, Point, Pose, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
import std_srvs.srv
from quadcopter_trajectory_msg.msg import QuadcopterTrajectory, QuadcopterState


class Poly_trajectory(object):

    def __init__(self):

        self._dimension = int(rospy.get_param('~settings/dimension'))
        self._weights = np.array(rospy.get_param('~settings/weights')) 
        self._polynom_degree = int(rospy.get_param('~settings/polynom_degree'))
        self._max_derivative_to_optimize = int(rospy.get_param('~settings/max_derivative_to_optimize'))

        self._setPoints = rospy.get_param('~trajectory/setPoints/first')
        self._eps = float(rospy.get_param('~trajectory/eps'))
        self._v_max = float(rospy.get_param('~trajectory/v_max'))
        dt = rospy.get_param('~time_step')
        
        self.build_trajectory()
        self._trajectory.sample_trajectory(dt)

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
        self._trajectory = Trajectory(self._polynom_degree, self._dimension)
        
        vertices = self._trajectory.setup_trajectory(self._setPoints, self._dimension, self._max_derivative_to_optimize)

        self._time_vect = vertices.estimate_segment_time(self._v_max, self._eps)

        self._trajectory.setup_from_vertice(vertices, self._time_vect, ACCELERATION, self._dimension, self._weights)
        self._trajectory.solve()


    def build_msg(self, time, raw):
        trajectory = QuadcopterTrajectory()
    
        trajectory.size = len(raw)
        pathHead = Header()
        pathHead.frame_id = "map"
        trajectory.header = pathHead

        for idx, pt in enumerate(raw):
            position = Point()
            position.x, position.y, position.z = pt[:3]

            velocity = Point()
            velocity.x, velocity.y, velocity.z = pt[3:6]

            rate = Point()
            rate.x, rate.y, rate.z = pt[6:9]

            quaternion = Quaternion()
            quaternion.x, quaternion.y, quaternion.z, quaternion.w = pt[9:]

            state = QuadcopterState()
            state.position = position
            state.attitude = quaternion
            state.velocity = velocity
            state.rates = rate
            
            trajectory.states.append(state)
        
        self.trajectory = trajectory
    
    def build_viz(self, trajectory):
        trajectory_viz = MarkerArray()

        for idx, pt in enumerate(trajectory.states):
            pts_marker = Marker()

            pathHead = Header()
            pathHead.frame_id = "map"
            pts_marker.header = pathHead

            pts_marker.type = 2
            pts_marker.action = 0
            pts_marker.id = idx

            pt_viz = Pose()
            pt_viz.orientation = pt.attitude
            pt_viz.position = pt.position

            pts_marker.pose = pt_viz

            pts_marker.scale.x = 0.05
            pts_marker.scale.y = 0.05
            pts_marker.scale.z = 0.05
            pts_marker.color.a = 1.0
            pts_marker.color.r = 0.0
            pts_marker.color.g = 1.0
            pts_marker.color.b = 0.0

            trajectory_viz.markers.append(pts_marker)

        return trajectory_viz

    def run(self):
        # while not rospy.is_shutdown():
        self.trajectory_pub.publish(self.trajectory)
        self.trajectory_viz_pub.publish(self.build_viz(self.trajectory))
        print("End")

    pass

if __name__ == "__main__":
    rospy.init_node('poly_trajectory_node')
    
    poly = Poly_trajectory()
    poly.set_subscribers_publishers()
    poly.run()

    rospy.spin()
    pass


