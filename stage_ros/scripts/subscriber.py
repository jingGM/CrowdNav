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
from gazebo_msgs.msg import ModelStates 
from stage_ros.msg import ScanObs, Scan, CamImage, ImageObs, Velocities, Velocity, ActionObs,Action, GoalObs, GazeboData
from stage_ros.srv import UpdateGazebo, UpdateGazeboResponse
import tf


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

        self.RGBs       = [ ImageObs()  ]*self.robotnumber
        self.Depths     = [ ImageObs()  ]*self.robotnumber
        self.scans      = [ ScanObs()   ]*self.robotnumber
            
        self.poses       = [0,0,0,0]*self.robotnumber   #float64 x,y,z,a
        self.collisions = [False]*self.robotnumber

        self.timeinterval = 100
        self.lastfetchingtime = 0

        for i in range(self.robotnumber):

            topicname = '/turtlebot%d/scan'%i
            scan= message_filters.Subscriber(topicname, LaserScan)

            topicname = '/turtlebot%d/camera/image_raw'%i
            cam= message_filters.Subscriber(topicname, Image)
            
            topicname = '/turtlebot%d/camera/depth/image_raw'%i
            dcam = message_filters.Subscriber(topicname, Image)

            ts = message_filters.TimeSynchronizer([scan,cam,dcam], 1)
            ts.registerCallback(self.callbacksync,(i))

        rospy.Subscriber("/gazebo/model_states", ModelStates, self.callbackpose,queue_size=1)
        #rospy.Subscriber("/clock", Clock, callbackunsync,callback_args=(env),queue_size=1)

        Updategazebodata = rospy.Service('updategazebodata', UpdateGazebo,self.handleservice)

    def handleservice(self,resp):
        print(self)
        print(resp)
        output              = UpdateGazeboResponse()
        output.RGBs         = self.RGBs
        output.Depths       = self.Depths
        output.scans        = self.scans
        output.poses        = self.poses
        output.collisions   = self.collisions

        return output


    def listener(self):
        rospy.spin()


    def callbacksync(self,scan, cam, dcam,i):

        self.RGBs[i].image_pprev = self.RGBs[i].image_prev
        self.RGBs[i].image_prev  = self.RGBs[i].image_now
        self.RGBs[i].image_now   = cam.data

        self.Depths[i].image_pprev = self.Depths[i].image_prev
        self.Depths[i].image_prev  = self.Depths[i].image_now
        self.Depths[i].image_now   = dcam.data

        self.scans[i].scan_pprev = self.scans[i].scan_prev
        self.scans[i].scan_prev = self.scans[i].scan_now
        self.scans[i].scan_now = scan.ranges

    def callbackpose(self,data):
        for i in range(len(data.name)):
            for j in range(self.robotnumber):
                if data.name[i] == 'turtlebot%d'%j:
                    tquaternion = [ data.pose[i].orientation.x,
                                   data.pose[i].orientation.y,
                                   data.pose[i].orientation.z,
                                   data.pose[i].orientation.w ]
                    tposition   = [ data.pose[i].position.x,
                                   data.pose[i].position.y,
                                   data.pose[i].position.z, ]
                    euler = tf.transformations.euler_from_quaternion(tquaternion)
                    tposition.append(euler[2])
                    self.poses[j] = tposition




def handleservices(resp,env):
        print(resp)
        print(env)
        output              = UpdateGazeboResponse()
        return output

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    RobotNumber=5

    data = EnvData(RobotNumber)
    data.listener()
