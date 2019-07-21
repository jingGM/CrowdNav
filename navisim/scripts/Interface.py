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
from std_srvs.srv import Empty


def resetenvironment():
    rospy.wait_for_service('/gazebo/reset_world')
    reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    reset_simulation()


def talker(cmd):
    pub = rospy.Publisher('test_hahahaha', String, queue_size=10)
    
    rate = rospy.Rate(10) # 10hz
    
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

def listener(RobotNumber,SubData):

    for i in range(RobotNumber):

        #SubData['clock']
        #clock = message_filters.Subscriber("/clock", Clock)

        topicname = '/jackal%d/scan'%i
        #SubData['scan'] 
        scan= message_filters.Subscriber(topicname, LaserScan)

        topicname = '/jackal%d/depth/image_raw'%i
        #SubData['cam'] 
        cam= message_filters.Subscriber(topicname, Image)
        
        topicname = '/jackal%d/depth/depth/image_raw'%i
        #SubData['dcam'] 
        dcam = message_filters.Subscriber(topicname, Image)

        ts = message_filters.TimeSynchronizer([scan,cam,dcam], 1)
        ts.registerCallback(callback,i)

    rospy.spin()

def callback(scan, cam, dcam,i):
    #rospy.loginfo(rospy.get_caller_id() + data.data)
    print(scan.header)
    print(cam.header)
    print(i)


def cleardata(SubData):
    SubData = {'clock':[], 'scan':[], 'cam':[], 'dcam':[]}

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    RobotNumber=4

    SubData = {'clock':[], 'scan':[], 'cam':[], 'dcam':[]}

    listener(RobotNumber,SubData)

    #try:
        #talker()
    #except rospy.ROSInterruptException:
    #    pass