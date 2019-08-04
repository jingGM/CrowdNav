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

    def __call__(self, x,datastore=[]):
        # x is scanObsBatch, len(x) == num_agents
        if len(x) > 1:
            filtered_x = []
            if len(datastore):
                for i in range(len(x)):  # for each agent
                    if self.obstype == "scan":
                        
                        tempdata = datastore[i]
                        tempdata_changed = np.swapaxes(tempdata, 0, 1)
                        tempdata_changed[2] = tempdata_changed[1]
                        tempdata_changed[1] = tempdata_changed[0]
                        tempdata_changed[0] = np.array(x[i].scan_now.ranges)
                        tempdata_back = np.swapaxes(tempdata_changed, 0, 1)
                        datastore[i] = tempdata_back

                        if self.delta:
                            data = np.stack(
                                (tempdata_changed[0] - tempdata_changed[2],
                                 tempdata_changed[0] - tempdata_changed[1],
                                 tempdata_changed[0]), axis=1)
                        else:
                            data = datastore[i]
                        '''
                        print(data.shape)
                        print(data[0][0])
                        print(data[0][1])
                        print(data[0][2])
                        print('==========scan shape======================')
                        '''
                    elif self.obstype == "image":
                        bridge = CvBridge()
                        try:
                          image_now = bridge.imgmsg_to_cv2(x[i].image_now.data, "bgr8")
                        except CvBridgeError as e:
                          print(e)

                        datastore[i][4] = datastore[i][3]
                        datastore[i][3] = datastore[i][2]
                        datastore[i][2] = datastore[i][1]
                        datastore[i][1] = datastore[i][0]
                        datastore[i][0] = np.array(image_now)
                        
                        if self.delta:
                            data = np.stack(
                                (datastore[i][0] - datastore[i][4],
                                 datastore[i][0] - datastore[i][3],
                                 datastore[i][0] - datastore[i][2],
                                 datastore[i][0] - datastore[i][1],
                                 datastore[i][0]), axis=0)
                        else:
                            data = datastore[i]
                            '''
                            print(data.shape)
                            print(data[0][0])
                            print(data[1][0])
                            print(data[2][0])
                            print(data[3][0])
                            print(data[4][0])
                            print('==========image shape======================')
                            '''
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
                
            else:
                for i in range(len(x)):  # for each agent
                    if self.obstype == "scan":
                        scantemp = np.stack((np.array(x[i].scan_now.ranges),np.array(x[i].scan_now.ranges),np.array(x[i].scan_now.ranges)),axis=1)
                        datastore.append(scantemp)

                        data = datastore[i]
                        #print(data.shape)
                        #print('==========scan shape======================')

                    elif self.obstype == "image":
                        bridge = CvBridge()
                        try:
                          image_now = bridge.imgmsg_to_cv2(x[i].image_now.data, "bgr8")
                        except CvBridgeError as e:
                          print(e)

                        imagetemp = np.stack((np.array(image_now),np.array(image_now),np.array(image_now),np.array(image_now),np.array(image_now)),axis=0)
                        datastore.append(imagetemp)

                        data = datastore[i]
                        '''
                        print(data.shape)
                        print(data[0][0])
                        print(data[1][0])
                        print(data[2][0])
                        print(data[3][0])
                        print(data[4][0])
                        print('==========image shape======================')
                        '''    
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
                
            return np.array(filtered_x),datastore
        else:
            if len(datastore): 
                if self.obstype == "scan":
                    tempdata = datastore
                    tempdata_changed = np.swapaxes(tempdata, 0, 1)
                    tempdata_changed[2] = tempdata_changed[1]
                    tempdata_changed[1] = tempdata_changed[0]
                    tempdata_changed[0] = np.array(x.scan_now.ranges)
                    tempdata_back = np.swapaxes(tempdata_changed, 0, 1)
                    datastore = tempdata_back

                    if self.delta:
                        data = np.stack(
                            (tempdata_changed[0] - tempdata_changed[2],
                             tempdata_changed[0] - tempdata_changed[1],
                             tempdata_changed[0]), axis=1)
                    else:
                        data = datastore
                    print(data.shape)
                    print('==========scan shape======================')

                elif self.obstype == "image":
                    bridge = CvBridge()
                    try:
                      image_now = bridge.imgmsg_to_cv2(x.image_now.data, "bgr8")
                    except CvBridgeError as e:
                      print(e)

                    datastore[4] = datastore[3]
                    datastore[3] = datastore[2]
                    datastore[2] = datastore[1]
                    datastore[1] = datastore[0]
                    datastore[0] = np.array(image_now)
                    
                    if self.delta:
                        data = np.stack(
                            (datastore[0] - datastore[4],
                             datastore[0] - datastore[3],
                             datastore[0] - datastore[2],
                             datastore[0] - datastore[1],
                             datastore[0]), axis=0)
                    else:
                        data = datastore
                        print(data.shape)
                        print('==========image shape======================')
                else:
                    data = np.array(x)
            else:
                if self.obstype == "scan":
                    datastore = np.stack((np.array(x.scan_now.ranges),np.array(x.scan_now.ranges),np.array(x.scan_now.ranges)),axis=1)

                    data = datastore
                    print(data.shape)
                    print('==========scan shape======================')

                elif self.obstype == "image":
                    bridge = CvBridge()
                    try:
                      image_now = bridge.imgmsg_to_cv2(x.image_now.data, "bgr8")
                    except CvBridgeError as e:
                      print(e)

                    datastore = np.stack((np.array(image_now),np.array(image_now),np.array(image_now),np.array(image_now),np.array(image_now)),axis=0)

                    data = datastore
                    print(data.shape)
                    print('==========image shape======================')
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
            return data,datastore
