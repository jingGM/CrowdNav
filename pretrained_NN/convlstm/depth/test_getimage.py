import rospy
from std_msgs.msg import String

import message_filters

from sensor_msgs.msg import LaserScan, CameraInfo, Image
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry
from naviswarm.msg import ScanObs, Scan, Velocities, Velocity, ActionObs,Action, CameraImage
from kobuki_msgs.msg import BumperEvent, CliffEvent

from gazebo_msgs.msg import ContactState
from std_srvs.srv import Empty

import rosservice
import csv
import cv2
import numpy as np
import pylab as plt
import os
import math
#from ppo.RealTimeTracking.predict import Memory
from cv_bridge import CvBridge, CvBridgeError
import time
# import depthNN.ImageNN as depthN 
# from depthNN import ImageNN
#import rgbNN.rgbNN.ImageNN as rgbN

class Images(object):
    def __init__(self):
        self.rgbs = []
        self.depths = []
        self.frames = 7

        for i in range(self.frames):
            rgb = np.zeros((60,80,3))
            self.rgbs.append(rgb)
            self.depths.append(np.zeros((60,80,1)))

        #self.depthN = ImageNN()
        #self.rgbN = rgbN()
        self.counter = [0,0]

    #def storeimages(self):

    def predictdepth(self):
        #graph = tf.get_default_graph()
        inputi = np.array(self.depths)
        imagesinput = inputi[:self.frames-1,::,::,::]
        #predictionimage = self.depthN.predict_image(imagesinput)
        # print(predictionimage.shape)
        # print('---------------')
        
        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot(121)
        # toplot = predictionimage[0,self.frames-2,::,::,0]
        # plt.imshow(toplot)
        # ground_truth = self.depths[self.frames-1][::, ::, 0]
        # plt.imshow(ground_truth)
        # ax = fig.add_subplot(122)
        # plt.savefig('depth%i.png',self.counter[1])


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def getimages(imagess):
    #topicname1 = '/turtlebot0/camera/image_raw'
    #rospy.Subscriber(topicname1, Image, callback_fun,callback_args=('rgb',imagess),queue_size=1)
    
    topicname2 = '/turtlebot0/camera/depth/image_raw'
    rospy.Subscriber(topicname2, Image, callback_fun,callback_args=('depth',imagess),queue_size=1)

    rospy.spin()


def callback_fun(data,args):
    imagetype = args[0]
    imagesclass = args[1]
    print(imagesclass.counter)
    print(imagetype)
    bridge = CvBridge()
    time.sleep(0.1)
    if imagetype=='rgb':
        try:
          image_now_cv = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        image_now = np.expand_dims(rgb2gray(np.array(image_now_cv)),axis=2)

        imagesclass.counter[0] += 1
        for i in range(imagesclass.frames):
            if i == (imagesclass.frames-1):
                imagesclass.rgbs[i] = image_now
            else:
                imagesclass.rgbs[i] = imagesclass.rgbs[i+1]
        
    else:
        try:
          image_now_cv = bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
          print(e)
        image_now = np.array(image_now_cv)
        r,c = np.shape(image_now)
        for k in range(r):
            for j in range(c):
                if math.isnan(image_now[k][j]):
                    image_now[k][j] = 5

        #image_now = np.expand_dims(image_now,axis=2)
        print(image_now.shape)

        saveiamge = image_now*(255/image_now.max())
        cv2.imwrite(os.path.join("./frames","{0:06d}.jpg".format(imagesclass.counter[1])), saveiamge)
        np.savetxt("./csvs/ground{0:06d}.csv".format(imagesclass.counter[1]), image_now, delimiter=",")


        imagesclass.counter[1] += 1
        for i in range(imagesclass.frames):
            if i == (imagesclass.frames-1):
                imagesclass.depths[i] = image_now
            else:
                imagesclass.depths[i] = imagesclass.depths[i+1]        
        #imagesclass.predictdepth()




if __name__ == "__main__":
    rospy.init_node('test', anonymous=True)
    imagess = Images()
    getimages(imagess)




    # image_out_NN = []
    # for i in range(len(track)):
    #     image_in_NN = track[i,::,::,::]
    #     print(image_in_NN.shape)

    #     output = imagenetwork.predict(image_in_NN)
    #     output = rgb2gray(output)
    #     image_out_NN.append(output)
        
    #     print(output.shape)
    #     print('--------------')
    # image_out_NN = np.array(image_out_NN)

    # predict_image = seq.predict(track)
    # ground_truth = noisy_movies[which][2, ::, ::, 0]












    