#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:47:07 2019

@author: jing
"""

import rospy
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel

from geometry_msgs.msg import Twist, Pose

class EmergentScenarios(object):
    '''
    need to set ros.init_node in other places
    put resetScenario in reseting part of environment
    use spawnped several times to spawn obstacle
    '''
    def __init__(self,
                 file_type="ped",
                 robotName="turtlebot0"):

        self.file_type = file_type
        self.filename = "./"+file_type+".sdf"
        self.numberObstacle = 0
        self.robotName = robotName

    def resetScenario(self):
        for i in range(self.numberObstacle):
            print(self.file_type+"_%d"%i)
            self.deletemodel(self.file_type+"_%d"%i)


    def spawnped(self,spawnpose):
        rospy.wait_for_service("gazebo/spawn_sdf_model")
        spawnservice = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)

        f = open(self.filename,'r')
        sdff = f.read()

        spawnrobot = SpawnModelRequest()
        spawnrobot.model_name = self.file_type+"_%d"%self.numberObstacle
        spawnrobot.model_xml  = sdff
        spawnrobot.robot_namespace = "emergent"
        spawnrobot.initial_pose = spawnpose
        spawnrobot.reference_frame = self.robotName
        success = spawnservice(spawnrobot)
        print(success.status_message)

        self.numberObstacle += 1

    def deletemodel(self, modelname):
        rospy.wait_for_service("gazebo/delete_model")
        deleteservice = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        success = deleteservice(modelname)
        print(success.status_message)

if __name__ == '__main__':
    rospy.init_node('ser', anonymous=True)        
    spawnpose = Pose()
    
    env = EmergentScenarios(file_type="corner")

    spawnpose.position.x = 1
    env.spawnped(spawnpose)

    # spawnpose.position.x = 5
    # env.spawnped(spawnpose)

    # env.resetScenario()
    



