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
from geometry_msgs.msg import Twist
from stage_ros.msg import ScanObs, Scan, CamImage, ImageObs, Velocities, Velocity, ActionObs,Action, GoalObs
from stage_ros.srv import UpdateGazebo, UpdateGazeboResponse

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
        self.actions    = [ ActionObs() ]*self.robotnumber
        self.velocities = [ Action()    ]*self.robotnumber
        self.pose       = [0,0,0,0]*self.robotnumber   #float64 x,y,z,a

        self.goals      = [ GoalObs() ]*self.robotnumber      #relative distance and relative angle, from robot to goal
        self.rewards    = [0]*self.robotnumber
        self.terminals  = [False]*self.robotnumber

        self.timeinterval = 100
        self.lastfetchingtime = 0

    def cleardata(self):
        self.RGBs       = [ ImageObs()  ]*self.robotnumber
        self.Depths     = [ ImageObs()  ]*self.robotnumber
        self.scans      = [ ScanObs()   ]*self.robotnumber
        self.actions    = [ ActionObs() ]*self.robotnumber
        self.velocities = [ ActionObs() ]*self.robotnumber
        self.goals      = [ GoalObs() ]*self.robotnumber      #relative distance and relative angle, from robot to goal
        
        self.pose       = [0,0,0,0]*self.robotnumber   #float64 x,y,z,a
        self.rewards    = [0]*self.robotnumber
        self.terminals  = [False]*self.robotnumber

        self.lastfetchingtime = 0
    
    def resetenvironment(self):
        rospy.wait_for_service('/gazebo/reset_world')
        reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_simulation()
        #d = rospy.Duration(10, 0)
        #rospy.sleep(d)   

    def setrobotposition(self,starts,goals):
        self.resetenvironment()
        time.sleep(1)

        robot_index = 0
        for s in starts:
            start = ModelState()
            setpose.model_name = "turtlebot%d",robot_index
            start.pose.position.x = s[0]
            start.pose.position.y = s[1]
            start.pose.orientation.w = np.cos(s[2] / 2)
            start.pose.orientation.z = np.sin(s[2] / 2)
            robot_index += 1

            rospy.wait_for_service('/gazebo/set_model_state')
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( setpose )

        #################################################################################################################################################
        ### need a function to transfer goals to relative goals
        ###    self.goals
        #######################################################################################################################################################        
        self.goals = goals


        self.cleardata()
        time.sleep(1)
        
        return resp

    def calculategoal(self,start,goal):
        print()


    def setvelocities(self, cmd):
        
        for i in range(self.robotnumber):
            self.actions[i].ac_pprev = self.actions[i].ac_prev
            self.actions[i].ac_prev = self.actions[i].ac_now
            self.actions[i].ac_now = cmd[i]

        pub = []
        NoR = len(cmd)
        for i in range(NoR):
            topicname = '/turtlebot%d/cmd_vel_mux/input/navi'%i
            pub.append(rospy.Publisher(topicname, Twist, queue_size=10))

        
        counter = [0]*NoR
        while (sum(counter)!=NoR):
            for i in range(NoR):
                if ((pub[i].get_num_connections()>0) & (counter[i]==0)):
                    pub[i].publish(cmd[i])
                    counter[i]=1
        '''
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            pub.publish(cmd)
            r.sleep()
        '''

    def _read_data(self):
        
        rospy.wait_for_service('updategazebodata')
        updategazebo = rospy.ServiceProxy('updategazebodata', UpdateGazebo)
        
        try:
            request = UpdateGazeboResponse()
            request = updategazebo(True)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
        #data = self.getgazebodata()
        self.RGBs       = request.RGBs
        self.Depths     = request.Depths
        self.scans      = request.scans 
        self.pose       = request.poses
        print(request)

        # need data.collisions to determin self.terminals




        






if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    data = EnvData(5)

    data._read_data()
    #listener(2)
'''
    cmd = [Twist()]*2
    cmd[0].linear.x = 10
    cmd[1].linear.x = 1
    
    talker(cmd)
    
    


    TimeStart = rospy.get_time()
    TimeElapsed = 0

    RobotNumber=4

    SubData = {'clock':[], 'scan':[], 'cam':[], 'dcam':[]}

    
'''
