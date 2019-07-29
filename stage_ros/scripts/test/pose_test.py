#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 19:47:07 2019

@author: jing
"""

import rospy
import tf


if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)

    RobotNumber=5

    data = EnvData(RobotNumber)
    listener(RobotNumber,data)