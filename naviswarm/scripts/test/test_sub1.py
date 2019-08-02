#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:47:07 2019

@author: jing
"""

import rospy
from std_msgs.msg import String, uint8

import message_filters

from sensor_msgs.msg import LaserScan, CameraInfo, Image
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from naviswarm.msg import ScanObs, Scan, Velocities, Velocity, ActionObs,Action, CameraImage
from kobuki_msgs.msg import BumperEvent, CliffEvent

from gazebo_msgs.msg import ContactState

import rosservice
import csv
import numpy as np


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


        #self.RGB       = [ Image() ]*self.robotnumber
        #self.PRGB      = [ Image() ]*self.robotnumber
        #self.PPRGB     = [ Image() ]*self.robotnumber

        #self.Depth     = [ Image()     ]*self.robotnumber
        #self.PDepth    = [ Image()     ]*self.robotnumber
        #self.PPDepth   = [ Image()     ]*self.robotnumber

        #self.scan      = [ LaserScan()     ]*self.robotnumber
        #self.Pscan     = [ LaserScan()     ]*self.robotnumber
        #self.PPscan    = [ LaserScan()     ]*self.robotnumber

        self.pose      = [0,0,0,0]*self.robotnumber   #float64 x,y,z,a

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

    rospy.Subscriber("/clock", Clock, callbackunsync,callback_args=('clock',env),queue_size=1)
    
    rospy.Subscriber("/turtlebot0/mobile_base/events/bumper", BumperEvent, callbackfortest,queue_size=1)
    
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

    #rospy.loginfo(rospy.get_caller_id() + data.data)
    #env.scan[i]   = scan
    #env.RGB[i]    = cam
    #env.Depth[i]  = dcam

    #print('----------------%d--------------------'%i)
    #print(env.scan[i][1].header)
    #print(env.scan[i][0].header)



def callbackunsync(data,args):
    msg = args[0]
    env = args[1]

    if msg == 'clock':
        '''
        print(data)
        print("++++++++++++++++++RGB0+++++++++++++++++++")
        print(env.RGB[0][0].header)
        print("-----")
        print(env.RGB[0][1].header)
        print("-----")
        print(env.RGB[0][2].header)
        print("========================================")
        
        print("++++++++++++++++++RGB1+++++++++++++++++++")
        print(env.RGB[1][0].header)
        print("-----")
        print(env.RGB[1][1].header)
        print("========================================")
        print("++++++++++++++++++scan++++++++++++++++++")
        print(env.scan[0][0].header)
        print("-----")
        print(env.scan[0][1].header)
        print("========================================")
        
        print(env.RGB[0][0].header)
        '''
    if msg == 'pose':
        env.pose
    if msg =='velocity':
        env.velocities[i] = env.actions[i].ac_now






from cv_bridge import CvBridge, CvBridgeError

def testcompressed():
    #data = CameraImage()
    #print(type(data.data))
    #data.data+='1'
    #data.data.append(1)
    rospy.Subscriber("/turtlebot0/camera/image_raw", Image, callbackfortest, queue_size=1)
    
    rospy.spin()
    #print(data)
    '''
    callbackstatus = False
    while(not callbackstatus):
        if (rosservice.get_service_node("update_goals")!=none):
            callbackstatus=True
            value = [geometry_msgs.Point()]
    '''  


def callbackfortest(data):
    image = data.data
    bridge = CvBridge()
    try:
      cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    a = np.array(cv_image)
    print(a.shape)
    #print(data.data)

    #print("--------------------------------")
    #print(image.cams)
    #image.cams.append(data.data)
    #print(len(data.ranges))
    #print(data.header)
    '''
    with open('sales.csv', 'w') as csv_file:
        print("writing")
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(data.data)
    
    
    '''
if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    #RobotNumber=5

    #data = EnvData(RobotNumber)
    #listener(RobotNumber,data)
    testcompressed()