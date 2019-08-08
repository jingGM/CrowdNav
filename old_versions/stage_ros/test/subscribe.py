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
#from navisim.msg import Scan



def talker():
    pub = rospy.Publisher('test_hahahaha', String, queue_size=10)
    
    
    
    rate = rospy.Rate(10) # 10hz
    
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()
'''
def listener(RobotNumber,SubData):

    for i in range(RobotNumber):

        #SubData['clock']
        clock = message_filters.Subscriber("/clock", Clock)

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
'''



def listener(RobotNumber,SubData):

    

    for i in range(RobotNumber):

        #rospy.Subscriber("/clock", Clock, callback,(i), queue_size=1)
        
        topicname = '/robot_0/base_scan'
        rospy.Subscriber(topicname, LaserScan ,callback,(i), queue_size=1)

    rospy.spin()

def callback(data,i):
    #rospy.loginfo(rospy.get_caller_id() + data.data)
    print(i)
    print(data.header)


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