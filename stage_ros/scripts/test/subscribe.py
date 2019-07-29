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
#from navisim.msg import Scan



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

        self.RGB       = [ LaserScan() ]*self.robotnumber
        self.PRGB      = [ LaserScan() ]*self.robotnumber
        self.PPRGB     = [ LaserScan() ]*self.robotnumber

        self.Depth     = [ Image()     ]*self.robotnumber
        self.PDepth    = [ Image()     ]*self.robotnumber
        self.PPDepth   = [ Image()     ]*self.robotnumber

        self.scan      = [ Image()     ]*self.robotnumber
        self.Pscan     = [ Image()     ]*self.robotnumber
        self.PPscan    = [ Image()     ]*self.robotnumber

        self.pose      = [0,0,0,0]*self.robotnumber   #float64 x,y,z,a

        self.collision = [False]*self.robotnumber

        self.timeinterval = 100
        self.lastfetchingtime = 0

        TS = []
        for i in range(self.robotnumber):

            topicname = '/jackal%d/scan'%i
            scan= message_filters.Subscriber(topicname, LaserScan)

            topicname = '/jackal%d/depth/image_raw'%i
            cam= message_filters.Subscriber(topicname, Image)
            
            topicname = '/jackal%d/depth/depth/image_raw'%i
            dcam = message_filters.Subscriber(topicname, Image)

            ts = message_filters.TimeSynchronizer([scan,cam,dcam], 1)
            ts.registerCallback(self.callbacksync,i)
            TS.append(ts)

    def listener(self):

        for i in range(self.robotnumber):

            topicname = '/jackal%d/scan'%i
            scan= message_filters.Subscriber(topicname, LaserScan)

            topicname = '/jackal%d/depth/image_raw'%i
            cam= message_filters.Subscriber(topicname, Image)
            
            topicname = '/jackal%d/depth/depth/image_raw'%i
            dcam = message_filters.Subscriber(topicname, Image)

            ts = message_filters.TimeSynchronizer([scan,cam,dcam], 1)
            ts.registerCallback(self.callbacksync,i)

        clock = message_filters.Subscriber("/clock", Clock)
        rospy.Subscriber("/clock", Clock, self.callbackunsync,('clock',0))
        
        rospy.spin()

    def callbacksync(self,scan, cam, dcam,i):
        #rospy.loginfo(rospy.get_caller_id() + data.data)
        self.scans[i]   = scan.ranges
        self.RGBs[i]    = cam.data
        self.Depths[i]  = cam.data

        print(scan.header)
        print(cam.header)
        print(i)

    def callbackunsync(self,data,type,i):
        if type == 'clock':
            print(data)
        if type == 'pose':
            self.pose
        if type =='velocity':
            self.velocities[i] = self.actions[i].ac_now






if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    RobotNumber=2

    data = EnvData(RobotNumber)
    data.listener()