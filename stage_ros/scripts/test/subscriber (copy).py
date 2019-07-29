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
from gazebo_msgs.msg import ModelState 


class EnvData(object):
    def __init__(self,RobotNumber):
        #==============================================
        #
        # for all values:   index   content
        #                   0       now
        #                   1       prevalue
        #                   2       pre-prevalue
        #                   :       :
        #
        #==============================================
        self.robotnumber = RobotNumber


        self.RGB    = []
        self.Depth  = []
        self.scan   = []
        for i in range(self.robotnumber):
            self.RGB.append([Image(),Image(),Image(),Image(),Image()])
            self.Depth.append([Image(),Image(),Image(),Image(),Image()])
            self.scan.append([LaserScan(),LaserScan(),LaserScan(),LaserScan(),LaserScan()])
            
            self.pose       = [0,0,0,0]*self.robotnumber   #float64 x,y,z,a
        self.collision = [False]*self.robotnumber

        self.timeinterval = 100
        self.lastfetchingtime = 0


def listener(robotnumber,env):

    for i in range(robotnumber):

        topicname = '/turtlebot%d/scan'%i
        scan= message_filters.Subscriber(topicname, LaserScan)

        topicname = '/turtlebot%d/camera/image_raw'%i
        cam= message_filters.Subscriber(topicname, Image)
        
        topicname = '/turtlebot%d/camera/depth/image_raw'%i
        dcam = message_filters.Subscriber(topicname, Image)



        ts = message_filters.TimeSynchronizer([scan,cam,dcam], 1)
        ts.registerCallback(callbacksync,(i,env))

    rospy.Subscriber("/clock", Clock, callbackunsync,callback_args=(env),queue_size=1)
    
    rospy.spin()

def callbacksync(scan, cam, dcam,args):
    i = args[0]
    env = args[1]

    env.RGB[i][4] = env.RGB[i][3]
    env.RGB[i][3] = env.RGB[i][2]
    env.RGB[i][2] = env.RGB[i][1]
    env.RGB[i][1] = env.RGB[i][0]
    env.RGB[i][0] = cam

    env.Depth[i][4] = env.Depth[i][3]
    env.Depth[i][3] = env.Depth[i][2]
    env.Depth[i][2] = env.Depth[i][1]
    env.Depth[i][1] = env.Depth[i][0]
    env.Depth[i][0] = dcam

    env.scan[i][4] = env.scan[i][3]
    env.scan[i][3] = env.scan[i][2]
    env.scan[i][2] = env.scan[i][1]
    env.scan[i][1] = env.scan[i][1]
    env.scan[i][0] = scan


def callbackunsync(data,env):
    print(data)






if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    RobotNumber=5

    data = EnvData(RobotNumber)
    listener(RobotNumber,data)