import rospy
import numpy as np
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import math
from tf.transformations import quaternion_from_euler

def reset_to_initial_position(start):
    s = np.array(start)
    state_msg = ModelState()
    for i in range(s.shape[0]):
    	quaternion = quaternion_from_euler(0,0,start[i][2])
        state_msg.model_name = 'tb{}'.format(i)
        state_msg.pose.position.x = start[i][0]
        state_msg.pose.position.y = start[i][1]
        # state_msg.pose.position.z = 0
        # state_msg.pose.orientation.x = quaternion[0]
        # state_msg.pose.orientation.y = quaternion[1]
        # state_msg.pose.orientation.z = quaternion[2]
        # state_msg.pose.orientation.w = quaternion[3]
        state_msg.pose.orientation.z = np.cos(start[i][2] / 2)
        state_msg.pose.orientation.w = np.sin(start[i][2] / 2)
        state_msg.reference_frame = 'ground_plane'
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            reset_robots = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            reset_robots(state_msg)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
if __name__ == "__main__":
    reset_to_initial_position([[1,1,0.75],[1,2,0.75]])
