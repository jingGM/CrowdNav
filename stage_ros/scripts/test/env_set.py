#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:47:07 2019

@author: jing
"""

import rospy

from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

def resetenvironment():
    rospy.wait_for_service('/gazebo/reset_world')
    reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    reset_simulation()
    #d = rospy.Duration(10, 0)
    #rospy.sleep(d)   


def setrobotposition(starts):

    resetenvironment()
    time.sleep(1.5)

    robot_index = 0
    for s in starts:
        start = ModelState()
        setpose.model_name = "jackal%d",robot_index
        start.pose.position.x = s[0]
        start.pose.position.y = s[1]
        start.pose.orientation.w = np.cos(s[2] / 2)
        start.pose.orientation.z = np.sin(s[2] / 2)
        robot_index += 1

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( setpose )
            print(resp)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    time.sleep(1)
    
    return resp.success


if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    RobotNumber=4
    
    try:
        setrobotposition()
    except rospy.ROSInterruptException:
        pass
    

    '''
    setpose = PoseWithCovarianceStamped()

    setpose.header.seq=1
    #setpose.header.frame_id=0
    setpose.pose.pose.position.x =1
    setpose.pose.pose.position.y =0
    setpose.pose.pose.position.z =0
    setpose.pose.pose.orientation.w =0
    setpose.pose.pose.orientation.z =1

    rospy.wait_for_service('/jackal0/SetPose')
    try:
        add_two_ints = rospy.ServiceProxy('SetPose', SetPose)
        resp1 = add_two_ints(setpose)
        print(resp1) 
        return
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
    '''