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

def handleservice(requst,env):
        output              = UpdateGazeboResponse()
        output.RGBs         = env.RGBs
        output.Depths       = env.Depths
        output.scans        = env.scans
        output.poses        = env.poses
        output.collisions   = env.collisions

        return output


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

    rospy.Subscriber("/gazebo/model_states", ModelStates, callbackpose,callback_args=(env),queue_size=1)
    #rospy.Subscriber("/clock", Clock, callbackunsync,callback_args=(env),queue_size=1)

    Updategazebodata = rospy.Service('updategazebodata', UpdateGazebo, lambda msg:handleservice(msg,env))

    rospy.spin()


def callbacksync(scan, cam, dcam,args):
    i = args[0]
    env = args[1]

    env.RGBs[i].image_pprev = env.RGBs[i].image_prev
    env.RGBs[i].image_prev  = env.RGBs[i].image_now
    env.RGBs[i].image_now   = cam.data

    env.Depths[i].image_pprev = env.Depths[i].image_prev
    env.Depths[i].image_prev  = env.Depths[i].image_now
    env.Depths[i].image_now   = dcam.data

    env.scans[i].scan_pprev = env.scans[i].scan_prev
    env.scans[i].scan_prev = env.scans[i].scan_now
    env.scans[i].scan_now = scan.ranges


def callbackpose(data,env):
    for i in range(len(data.name)):
        for j in range(env.robotnumber):
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
                env.poses[j] = tposition



#def callbackunsync(data,env):
#    print(data)






if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    RobotNumber=5

    data = EnvData(RobotNumber)
    listener(RobotNumber,data)