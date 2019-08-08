#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:47:07 2019

@author: jing
"""

import rospy
from std_msgs.msg import String

import message_filters

from sensor_msgs.msg import LaserScan, CameraInfo, Image
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from naviswarm.msg import ScanObs, Scan, Velocities, Velocity, ActionObs,Action, CameraImage
from kobuki_msgs.msg import BumperEvent, CliffEvent
from geometry_msgs.msg import Twist

from gazebo_msgs.msg import ContactState

import rosservice




def pub():
    vel = Twist()
    vel.linear.x = 1
    vel.angular.z = 0.1
    topicname = "/turtlebot0/cmd_vel_mux/input/navi"
    publish =rospy.Publisher(topicname, Twist, queue_size=10)
    #rate = rospy.Rate(10) # 10hz
    if (publish.get_num_connections()>0):
        publish.publish(vel)
        #rate.sleep()

if __name__ == '__main__':
    rospy.init_node('ser', anonymous=True)
    pub()
    #RobotNumber=5

    #data = EnvData(RobotNumber)
    #listener(RobotNumber,data)
    #testcompressed()


