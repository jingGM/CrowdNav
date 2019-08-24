# Copyright 2017 The DRLCA Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import numpy as np
import scipy.signal
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
import math


def gauss_log_prob(mean, logstd, x):
    var = tf.exp(2 * logstd)
    lp = -tf.square(x - mean) / (
        2 * var) - .5 * tf.log(tf.constant(2 * np.pi)) - logstd
    return tf.reduce_sum(lp, 1)


def gauss_kl(mean1, logstd1, mean2, logstd2):
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)

    kl = tf.reduce_mean(logstd2 - logstd1 + (var1 + tf.square(mean1 - mean2)) /
                        (2 * var2) - 0.5)
    return kl


def gauss_entropy(mean, logstd):
    return tf.reduce_mean(
        logstd + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def line_search(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n, stepfrac) in enumerate(0.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0.:
            return True, xnew
    return False, x


def discount(rewards, gamma):
    assert rewards.ndim >= 1
    # rewards[::-1]: reverse rewards, from n to 0
    return scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]


class RunningStat(object):
    def __init__(self, shape):
        self._size = 0
        self._mean, self._std = np.zeros(shape), np.zeros(shape)

    def add(self, x):
        x = np.asarray(x)
        assert x.shape == self._mean.shape

        self._size += 1
        if self._size == 1:
            self._mean = x
        else:
            self._mean_old = self._mean.copy()
            self._mean = self._mean_old + (x - self._mean_old) / self._size
            self._std = self._std + (x - self._mean_old) * (x - self._mean)

    @property
    def size(self):
        return self._size

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._std / (self._size - 1) if self._size > 1 else self._std

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._mean.shape

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class RunningAverageFilter(object):
    """
    do a running average filter of the incoming observations and rewards
    """

    def __init__(self, shape, obstype=None,
                 demean=False, destd=False, update=False,
                 delta=False, clip=None):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.obstype = obstype
        self.update = update
        self.delta = delta

        self.rs = RunningStat(shape)


    def __call__(self, x):
        #CAMMAXDISTANCE = 5
        if len(x) > 1:
            filtered_x = []
            for i in range(len(x)):  # for each agent
                if self.obstype == "scan":
                    if self.delta:
                        data = np.stack(
                            (np.array(x[i].scan_now.ranges) - np.array(x[i].scan_pprev.ranges),
                             np.array(x[i].scan_now.ranges) - np.array(x[i].scan_prev.ranges),
                             np.array(x[i].scan_now.ranges)),
                            axis=1)
                    else:
                        data = np.stack(
                            (x[i].scan_pprev.ranges,
                             x[i].scan_prev.ranges,
                             x[i].scan_now.ranges),
                            axis=1)
                        #print(data.shape)
                        #print('==========scan shape======================')
                elif self.obstype == "image":
                    # print("in image: ")
                    bridge = CvBridge()
                    try:
                      image_now_cv = bridge.imgmsg_to_cv2(x[i].image_now.data, "bgr8")
                      # print("0")
                    except CvBridgeError as e:
                      print(e)
                    image_now = rgb2gray(np.array(image_now_cv))
                    # print(image_now.shape)

                    try:
                      image_p1rev_cv = bridge.imgmsg_to_cv2(x[i].image_p1rev.data, "bgr8")
                      # print("1")
                    except CvBridgeError as e:
                      print(e)
                    image_p1rev = rgb2gray(np.array(image_p1rev_cv))
                    # print(image_p1rev.shape)

                    try:
                      image_p2rev_cv = bridge.imgmsg_to_cv2(x[i].image_p2rev.data, "bgr8")
                      # print("2")
                    except CvBridgeError as e:
                      print(e)
                    image_p2rev = rgb2gray(np.array(image_p2rev_cv))
                    # print(image_p2rev.shape)


                    if self.delta:
                        data = np.stack((image_now - image_p2rev,image_now - image_p1rev,image_now),axis=2)
                    else:
                        data = np.stack((image_p2rev,image_p1rev,image_now),axis=2)
                        # print(data.shape)
                        # print('==========image shape======================')
                        
                elif self.obstype == "goal":
                    data = [x[i].goal_now.goal_dist, x[i].goal_now.goal_theta]
                elif self.obstype == "action":
                    data = [
                        [x[i].ac_pprev.vx, x[i].ac_pprev.vz],
                        [x[i].ac_prev.vx, x[i].ac_prev.vz]]
                elif self.obstype == "vel":
                    data = [x[i].vel_now.vx, x[i].vel_now.vz]
                else:
                    data = x
                data = np.array(data)

                # print("data shape before filter: {}".format(data.shape))
                if self.update:
                    self.rs.add(data[-1])
                if self.demean:
                    data -= self.rs.mean
                if self.destd:
                    data /= (self.rs.std + 1e-8)
                if self.clip is not None:
                    data = np.clip(data, -self.clip, self.clip)

                filtered_x.append(data)
            return np.array(filtered_x)
        else:
            if self.obstype == "scan":
                    if self.delta:
                        data = np.stack(
                            (np.array(x[0].scan_now.ranges) - np.array(x[0].scan_pprev.ranges),
                             np.array(x[0].scan_now.ranges) - np.array(x[0].scan_prev.ranges),
                             np.array(x[0].scan_now.ranges)),
                            axis=1)
                    else:
                        data = np.stack(
                            (x[0].scan_pprev.ranges,
                             x[0].scan_prev.ranges,
                             x[0].scan_now.ranges),
                            axis=1)
                    
            elif self.obstype == "image":
                #tempimage = np.array(x[0].image_now.data)
                #print(tempimage.shape)
                bridge = CvBridge()
                try:
                  image_now_cv = bridge.imgmsg_to_cv2(x[i].image_now.data, "bgr8")
                  # print("0")
                except CvBridgeError as e:
                  print(e)
                image_now = rgb2gray(np.array(image_now_cv))

                try:
                  image_p1rev_cv = bridge.imgmsg_to_cv2(x[i].image_p1rev.data, "bgr8")
                  # print("1")
                except CvBridgeError as e:
                  print(e)
                image_p1rev = rgb2gray(np.array(image_p1rev_cv))

                try:
                  image_p2rev_cv = bridge.imgmsg_to_cv2(x[i].image_p2rev.data, "bgr8")
                  # print("2")
                except CvBridgeError as e:
                  print(e)
                image_p2rev = rgb2gray(np.array(image_p2rev_cv))

                data = np.stack((image_p2rev,image_p1rev,image_now),axis=2)
                

            elif self.obstype == "goal":
                    data = np.array([x[0].goal_now.goal_dist, x[0].goal_now.goal_theta])
            elif self.obstype == "action":
                data = np.array([[x[0].ac_pprev.vx, x[0].ac_pprev.vz],[x[0].ac_prev.vx, x[0].ac_prev.vz]])
                
            elif self.obstype == "vel":
                data = np.array([x[0].vel_now.vx, x[0].vel_now.vz])
                
            else: 
                data = np.array(x)
                

            if self.update:
                self.rs.add(data)
            if self.demean:
                data -= self.rs.mean
            if self.destd:
                data /= (self.rs.std + 1e-8)
            if self.clip is not None:
                data = np.clip(data, -self.clip, self.clip)
            data = np.expand_dims(data, axis=0)
            return data
