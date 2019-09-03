import rospy
import time
import math

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge, CvBridgeError
import cv2

# Hyper parameters
MAX_LINEAR_VELOCITY = 0.5
MAX_ANGULAR_VELOCITY = 0.4
CAMMAXDISTANCE = 5
STOPDISTANCE = 0.5


class MLSHAgent(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
        
        self._build_ph()
        
        self.subpolicies = []
        self.subpolicies.append(PPOAgent(self.sess, self.goal_ph, self.vel_ph,self.image_ph))
        self.subpolicies.append(ClassController())
        self.subpolicies.append(FailedSafeController(self.subpolicies[0]))
        
        self.sess.run(tf.global_variables_initializer())
        
    def _build_ph(self):
        self.image_ph= tf.placeholder(tf.float32,[None, 120,150, 3],'image_ph')
        # self.scan_ph = tf.placeholder(tf.float32,[None, 512, 3],     'scan_ph')
        self.goal_ph = tf.placeholder(tf.float32,[None, 2],          'goal_ph')
        self.vel_ph  = tf.placeholder(tf.float32,[None, 2],          'vel_ph')
            
    def act(self, obs, terminated):
        if terminated:
            return np.zeros(2)
        
        # scan = obs[0]
        goal = obs[0]
        # if np.min(scan) > 0.4 or (np.min(scan) * 4. > goal[0]):
        #     action = self.subpolicies[1].act(obs)
        # #elif np.min(scan) < 0.2:
        # #    action = self.subpolicies[0].act(obs)
        # else:
        action = self.subpolicies[0].act(obs)
        
        #action = self.subpolicies[1].act(obs)
            
        if action[0] < 0:
            action[0] = 0
        elif action[0] > MAX_LINEAR_VELOCITY:
            action[0] = MAX_LINEAR_VELOCITY
        
        if action[1] > MAX_ANGULAR_VELOCITY:
            action[1] = MAX_ANGULAR_VELOCITY
        elif action[1] < -MAX_ANGULAR_VELOCITY:
            action[1] = -MAX_ANGULAR_VELOCITY            
        return action
        
    def load(self):
        self.subpolicies[0].load()
        
        
class ClassController(object):
    def __init__(self):
        self.kp_x = 0.1
        self.kd_x = 0.0
        self.kp_z = 1.
        self.kd_z = 0.0
        
    def act(self, obs):
        goal = obs[0]
        vels = obs[1]
        action = np.zeros(2)
        action[0] = self.kp_x * goal[0]
        action[1] = self.kp_z * goal[1]
        if abs(goal[1]) < np.pi / 6.:
            action[0] = 0
        elif abs(goal[1]) > np.pi / 2.:
            action[0] = 1.
            
        return action
        
class FailedSafeController(object):
    def __init__(self, policy):
        self.policy = policy
        
    def act(self, obs):
        scan = obs[0]
        obs[0] = obs[0] / 1.25
        action = self.policy.act(obs)
        return action
        
        
class PPOAgent(object):
    def __init__(self, sess, goal, vel,image):
        self.sess = sess
        # self.scan_ph = scan
        self.image_ph = image
        self.goal_ph = goal
        self.vel_ph = vel
        
        self.model, self.means = self._build_net()
        
    def _build_net(self):

        imagenet = tl.layers.InputLayer(self.image_ph, name='image_input')
        imagenet = tl.layers.Conv2dLayer(imagenet,act=tf.nn.relu,shape=(5, 5, 3, 64), strides=(1,2,2,1),use_cudnn_on_gpu=True,name='Icnn1')
        imagenet = tl.layers.Conv2dLayer(imagenet,act=tf.nn.relu,shape=(5, 5, 64, 64), strides=(1,1,1,1),use_cudnn_on_gpu=True,name='Icnn2')
        imagenet = tl.layers.Conv2dLayer(imagenet,act=tf.nn.relu,shape=(3, 3, 64, 32), strides=(1,2,2,1),use_cudnn_on_gpu=True,name='Icnn3')
        #imagenet = tl.layers.Conv2dLayer(imagenet,act=tf.nn.relu,shape=(3, 3, 128, 128), strides=(1,1,1,1),use_cudnn_on_gpu=True,name='Icnn4')
        imagenet = tl.layers.FlattenLayer(imagenet, name='imagefl')
        imagenet = tl.layers.DenseLayer(imagenet, n_units=512, act=tf.nn.relu, name='Idense1')
        imagenet = tl.layers.DenseLayer(imagenet, n_units=256, act=tf.math.sigmoid, name='Idense2')
        image_output = imagenet.outputs
        #print(image_output.shape)

        act_net = tl.layers.InputLayer(tf.concat([self.goal_ph, self.vel_ph, image_output], axis=1), name='goal_input')

        #act_net = tl.layers.DenseLayer(act_net, n_units=64, act=tf.nn.tanh, name='act1')
        act_net = tl.layers.DenseLayer(act_net, n_units=128, act=tf.nn.relu, name='act2')
        linear  = tl.layers.DenseLayer(act_net, n_units=1, act=tf.nn.sigmoid, name='linear')
        angular = tl.layers.DenseLayer(act_net, n_units=1, act=tf.nn.tanh, name='angular')
        with tf.variable_scope('means'):
            action_mean = tf.concat([linear.outputs, angular.outputs], axis=1)
            # means = tf.concat([MAX_LINEAR_VELOCITY *linear.outputs, MAX_ANGULAR_VELOCITY * angular.outputs], axis=1)

        return [ act_net,imagenet, linear, angular], action_mean

    def act(self, obs):
        feed_dict = {
            self.goal_ph: [obs[0]],
            self.vel_ph: [obs[1]],
            self.image_ph:[obs[2]]
        }
        
        action = self.sess.run(self.means, feed_dict=feed_dict)[0]
        return action
    
    def load(self):
        model_name = 'best'
        for i in range(len(self.model)):
            params = tl.files.load_npz(name='./model/{}_act_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, self.model[i])
            
class Environment(object):
    def __init__(self, agent):
        import tf
        import tf.transformations
        rospy.init_node('ppo_agent')
        self.control_rate = rospy.Rate(10)
        self.agent = agent
        self.agent.load()
        # self.scan = np.zeros([512, 3])
        self.image = np.zeros([120,150,3])
        self.uwb_pos = np.zeros(4)
        self.uwb_pos[3] = np.pi / 2.
        self.uwb_vel = np.zeros(3)
        self.vel = np.zeros(2)
        self.target_pos = np.zeros(3)
        # self.target_pos[0] = 1
        self.odom_x = 0.
        self.odom_y = 0.
        self.odom_heading = 0.

        self.imagebuffer = []
        self.image_counter = 0
        
        # rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/target/position', Twist, self.target_callback)
        rospy.Subscriber('/camera/depth_registered/image_raw', Image, self.image_callback)
        
        self.command_pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)
        # self.command_pub = rospy.Publisher('/turtlebot0/cmd_vel_mux/input/navi', Twist, queue_size=1)
        
        time.sleep(1)
        
    def run(self):
        while not rospy.is_shutdown():
            goal, terminated = self.preprocess_nav()
            act = self.agent.act([goal, self.vel, self.image], terminated)
            
            act[1] = -act[1]

            if (self.target_pos[0]<STOPDISTANCE and self.target_pos[1]<STOPDISTANCE):
            	act[0] = 0
            	act[1] = 0

            print 'linear: ', act[0], ' --- angular: ', act[1]
            self.send_command(act)
            self.control_rate.sleep()
            if terminated:
                #direct += 1
                time.sleep(1)

            
    def send_command(self, act):
        command = Twist()
        command.linear.x = act[0]
        command.angular.z = act[1]
        # print(command)
        self.command_pub.publish(command)
    
    def preprocess_nav(self):
        goal = [self.compute_distance(), self.compute_angle()]
        print 'goal: ', goal[0], ' --- ', goal[1] * 57.3
        if goal[0] < 0.65:
            terminated = True
        else:
            terminated = False
        return goal, terminated
        
    def compute_distance(self):
        return np.hypot(self.uwb_pos[0] - self.target_pos[0], self.uwb_pos[1] - self.target_pos[1])
        
    def compute_angle(self):
        angle = self.uwb_pos[3] - math.atan2(-self.uwb_pos[1]+self.target_pos[1], -self.uwb_pos[0]+self.target_pos[0])
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        # print 'angle: ', angle * 57.3
        print 'p_heading: ', 57.3 * math.atan2(-self.uwb_pos[1]+self.target_pos[1], -self.uwb_pos[0]+self.target_pos[0]), '  r_heading: ', -57.3 * self.uwb_pos[3]
        return angle
        

    def process_depth_image(self,imagedata):
        bridge = CvBridge()
        try:
          image_now_cv = bridge.imgmsg_to_cv2(imagedata, "32FC1")
        except CvBridgeError as e:
          print(e)
        image_now = np.array(image_now_cv)
        r,c = np.shape(image_now)
        for k in range(r):
            for j in range(c):
                if math.isnan(image_now[k][j]) or image_now[k][j]>CAMMAXDISTANCE:
                    image_now[k][j] = CAMMAXDISTANCE

        #print(image_now.shape)
        image_now = np.array(cv2.resize(image_now,(150,120)),dtype=float)
        image_now = np.expand_dims(image_now, axis=2)        
        return image_now

    def rgb2gray(self,rgb):
        #print(rgb.shape)
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def process_rgb_image(self,imagedata):
        bridge = CvBridge()
        try:
          image_now_cv = bridge.imgmsg_to_cv2(imagedata.data, "bgr8")
          # print("0")
        except CvBridgeError as e:
          print(e)
        image_now = self.rgb2gray(np.array(image_now_cv))

        image_now = np.array(cv2.resize(image_now,(150,120)),dtype=float)
        image_now = np.expand_dims(image_now, axis=2)
        return image_now

    def image_callback(self,data):
        if self.image_counter>2:
        	for imageindex in range(3):
	        	image_now = self.process_depth_image(self.imagebuffer[imageindex])
	        	self.image = np.concatenate((self.image[:,:, 1:], image_now), axis=2)
        	self.imagebuffer = []
        	self.image_counter = 0
        else:
        	rospy.loginfo("==image==")
        	self.imagebuffer.append(data)
        	self.image_counter += 1
        
                
    def odom_callback(self, data):
    	# print("======velocity==========")
        import tf
        import tf.transformations
        self.vel[0] = data.twist.twist.linear.x
        self.vel[1] = data.twist.twist.angular.z/4.0 
        # print(self.vel)
        
        
    def uwb_vel_callback(self, data):
        self.uwb_vel[0] = data.angular.x
        
    def target_callback(self, data):
    	# print("========target==========")
        self.target_pos[0] = data.linear.x
        self.target_pos[1] = data.linear.y
        self.target_pos[2] = data.linear.z
        # print(self.target_pos)
        
        
    
if __name__ == '__main__':
    agent = MLSHAgent()
    env = Environment(agent)
    env.run()
            
