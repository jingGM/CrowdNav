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

import os

# import keras
import numpy as np
import tensorflow as tf

import tensorlayer as tl
from utils import RunningAverageFilter
from vel_smoother import VelocitySmoother


class Agent(object):
    def __init__(self, args, session, obs_shape, ac_shape):
        if not os.path.exists("./ppo/model"):
            os.makedirs("./ppo/model")

        self.policy = Policy(session, obs_shape, ac_shape, args)
        self.value = Value(session, obs_shape)
        self.delta = False

        self.scan_filter  = RunningAverageFilter(obs_shape[1], obstype="scan",  demean=False, destd=False, update=False, delta=self.delta)
        self.image_filter = RunningAverageFilter([obs_shape[3],obs_shape[4],obs_shape[5],obs_shape[6]], obstype="image", demean=False, destd=False, update=False, delta=self.delta)
        self.goal_filter  = RunningAverageFilter(obs_shape[2], obstype="goal",  demean=False, destd=False, update=False, delta=self.delta)
        self.vel_filter   = RunningAverageFilter(ac_shape,     obstype="vel",   demean=False, destd=False, update=False, delta=self.delta)
        self.reward_filter= RunningAverageFilter((), demean=False, clip=1)

        self.imagestore = []
        self.scanstore = []

    def obs_filter(self, obs):

        image_filtered,imagestore = self.image_filter(obs.ImageObsBatch,self.imagestore)
        self.imagestore = imagestore
        scan_filtered,scanstore   = self.scan_filter(obs.scanObsBatch,self.scanstore)
        self.scanstore = scanstore
        goal_filtered,_  = self.goal_filter(obs.goalObsBatch)
        vel_filtered,_   = self.vel_filter(obs.velObsBatch)

        if not self.delta:
            scan_filtered /= 4.0
        else:
            scan_filtered /= 4.0
        # scan_filtered = np.asarray(scan_filtered)
        # goal_filtered = np.asarray(goal_filtered)
        # print 'shape: ', np.shape(vel_filtered)
        # print 'vel: ', vel_filtered[0]
        # print 'after shape: ', np.shape(scan_filtered)
        return [ scan_filtered, goal_filtered, vel_filtered, image_filtered]


class Policy(object):
    def __init__(self, session, obs_shape, ac_shape, args):
        self.sess = session
        # K.set_session(self.sess)
        self.obs_shape = obs_shape
        self.ac_shape = ac_shape
        self.vel_smoother = VelocitySmoother()

        self.test_var = args.test_var
        self.max_vx = args.max_vx

        self.model, self.obs, self.means, self.log_vars,self.sampled_act, self.test_act = self._policy_net()

        self.sess.run(tf.global_variables_initializer())

    def _policy_net(self):
        scan = tf.placeholder(tf.float32, [None, self.obs_shape[1], self.obs_shape[0]],'scan_ph')
        goal = tf.placeholder(tf.float32, [None, 2], 'goal_ph')
        vel  = tf.placeholder(tf.float32, [None, 2], 'vel_ph')
        image= tf.placeholder(tf.float32, [None, self.obs_shape[3],self.obs_shape[4],self.obs_shape[5],self.obs_shape[6]], 'image_ph')

        net = tl.layers.InputLayer(scan, name='scan_input')
        net = tl.layers.MeanPool1d(net, filter_size=3, strides=2, name='min_pooling1')
        net = tl.layers.MeanPool1d(net, filter_size=3, strides=2, name='min_pooling2')
        net = tl.layers.MeanPool1d(net, filter_size=3, strides=2, name='min_pooling3')
        net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[5, self.obs_shape[0], 8], stride=2,name='cnn1')
        net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[3, 8, 16], stride=2,name='cnn2')
        net = tl.layers.FlattenLayer(net, name='fl')
        net = tl.layers.DenseLayer(net, n_units=128, act=tf.nn.relu, name='scan_output')
        scan_output = net.outputs

        '''
        def keras_block(imagenetx):
            imagenet = tf.keras.layers.ConvLSTM2D(filters=16,kernel_size=[7,7],strides=[2,2],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenetx)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.ConvLSTM2D(filters=32,kernel_size=[5,5],strides=[2,2],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenet)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.ConvLSTM2D(filters=64,kernel_size=[3,3],strides=[2,2],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenet)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.ConvLSTM2D(filters=64,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenet)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3),activation=tf.nn.relu,padding='same')(imagenet)
            return imagenet

        imagenet = tl.layers.InputLayer(image, name='image_input')
        imagenet = tl.layers.LambdaLayer(imagenet, fn=keras_block, name='kerasImage')
        imagenet = tl.layers.FlattenLayer(imagenet, name='imagefl')
        imagenet = tl.layers.DenseLayer(imagenet, n_units=960, act=tf.nn.relu, name='image_output')
        image_output = imagenet.outputs
        #print(image_output.shape)
        '''
        imagenet = tl.layers.InputLayer(image, name='image_input')


        act_net = tl.layers.InputLayer(tf.concat([goal, vel, scan_output], axis=1), name='goal_input')
        act_net = tl.layers.DenseLayer(act_net, n_units=64, act=tf.nn.tanh, name='act1')
        act_net = tl.layers.DenseLayer(act_net, n_units=64, act=tf.nn.tanh, name='act2')
        linear  = tl.layers.DenseLayer(act_net, n_units=1, act=tf.nn.sigmoid, name='linear')
        angular = tl.layers.DenseLayer(act_net, n_units=1, act=tf.nn.tanh, name='angular')
        with tf.variable_scope('means'):
            action_mean = tf.concat([linear.outputs, angular.outputs], axis=1)

        logvar_speed = (10 * 128) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, 2), tf.float32,tf.constant_initializer(0.0))
        log_vars = tf.reduce_sum(log_vars, axis=0)

        sampled_act = action_mean + tf.exp(log_vars / 2.0) * tf.random_normal(shape=(2,))

        test_act = action_mean + self.test_var * tf.random_normal(shape=(2,))

        return [net, act_net,imagenet, linear, angular], [scan, goal, vel, image],action_mean, log_vars, sampled_act, test_act

    def act(self, obs, terminated, batch=True):
        if not batch:
            obs = [np.expand_dims(x, 0) for x in obs]

        actions = self.sess.run(self.sampled_act, feed_dict={
                self.obs[0]: obs[0],
                self.obs[1]: obs[1],
                self.obs[2]: obs[2],
                self.obs[3]: obs[3]
            })

        for i, t in enumerate(terminated):
            if t:
                actions[i] = np.zeros(self.ac_shape)

        return actions

    def act_test(self, obs, terminated):
        actions = self.sess.run(self.test_act, feed_dict={
                self.obs[0]: obs[0],
                self.obs[1]: obs[1],
                self.obs[2]: obs[2],
                self.obs[3]: obs[3]
            })

        for i, t in enumerate(terminated):
            if t:
                actions[i] = np.zeros(self.ac_shape)
        return actions

    def save_network(self, model_name):
        for i in range(len(self.model)):
            tl.files.save_npz(
                self.model[i].all_params,
                name='./ppo/model/{}_act_{}.npz'.format(model_name, i),
                sess=self.sess)

    def load_network(self, model_name):
        for i in range(len(self.model)):
            params = tl.files.load_npz(
                name='./ppo/model/{}_act_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, self.model[i])


class Value(object):
    def __init__(self, session, obs_shape):
        self.sess = session
        self.obs_shape = obs_shape

        self.model, self.obs, self.value = self._value_net()
        self.ret_ph = tf.placeholder(tf.float32, shape=[None, ], name='return_ph')
        self.loss = tf.reduce_mean(tf.square(self.value - self.ret_ph))
        self.optimizer = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def _value_net(self):
        scan = tf.placeholder(tf.float32, [None, self.obs_shape[1], self.obs_shape[0]],'scan_value_ph')
        goal = tf.placeholder(tf.float32, [None, 2], 'goal_value_ph')
        vel  = tf.placeholder(tf.float32, [None, 2], 'vel_ph')
        image= tf.placeholder(tf.float32, [None, self.obs_shape[3],self.obs_shape[4],self.obs_shape[5],self.obs_shape[6]], 'image_ph')

        net = tl.layers.InputLayer(scan, name='scan_input_value')
        net = tl.layers.MeanPool1d(net, filter_size=3, strides=2, name='min_pooling1_value')
        net = tl.layers.MeanPool1d(net, filter_size=3, strides=2, name='min_pooling2_value')
        net = tl.layers.MeanPool1d(net, filter_size=3, strides=2, name='min_pooling3_value')

        net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[5, self.obs_shape[0], 8], stride=2,name='cnn1_value')
        net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[3, 8, 16], stride=2,name='cnn2_value')
        net = tl.layers.FlattenLayer(net, name='fl_value')
        net = tl.layers.DenseLayer(net, n_units=128, act=tf.nn.relu, name='cnn_output_value')
        cnn_output = net.outputs
        '''
        def keras_block(imagenetx):
            imagenet = tf.keras.layers.ConvLSTM2D(filters=16,kernel_size=[7,7],strides=[2,2],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenetx)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.ConvLSTM2D(filters=32,kernel_size=[5,5],strides=[2,2],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenet)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.ConvLSTM2D(filters=64,kernel_size=[3,3],strides=[2,2],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenet)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.ConvLSTM2D(filters=64,kernel_size=[3,3],strides=[1,1],padding='same',activation=tf.nn.relu,return_sequences=True)(imagenet)
            imagenet = tf.keras.layers.BatchNormalization()(imagenet)
            imagenet = tf.keras.layers.Conv3D(filters=1, kernel_size=(3, 3, 3),activation=tf.nn.relu,padding='same')(imagenet)
            return imagenet

        imagevnet = tl.layers.InputLayer(image, name='image_input_value')
        imagevnet = tl.layers.LambdaLayer(imagevnet, fn=keras_block, name='kerasImage_value')
        imagevnet = tl.layers.FlattenLayer(imagevnet, name='imagefl_value')
        imagevnet = tl.layers.DenseLayer(imagevnet, n_units=960, act=tf.nn.relu, name='image_output_value')
        image_voutput = imagevnet.outputs
        '''
        imagevnet = tl.layers.InputLayer(image, name='image_input_value')


        value_net = tl.layers.InputLayer(tf.concat([goal, vel, cnn_output], axis=1),name='goal_input_value')
        value_net = tl.layers.DenseLayer(value_net, n_units=64, act=tf.nn.tanh, name='value1')
        value_net = tl.layers.DenseLayer(value_net, n_units=64, act=tf.nn.tanh, name='value2')
        value_net = tl.layers.DenseLayer(value_net, n_units=1, name='value')
        value = value_net.outputs

        return [net, value_net,imagevnet], [scan, goal, vel, image], value

    def update(self, obs, returns):
        feed_dict = {
            self.obs[0]: obs[0],
            self.obs[1]: obs[1],
            self.obs[2]: obs[2],
            self.obs[3]: obs[3],
            self.ret_ph: returns}

        self.sess.run(self.optimizer, feed_dict)

    def predict(self, obs):
        value = self.sess.run(
            self.value,
            feed_dict={
                self.obs[0]: obs[0],
                self.obs[1]: obs[1],
                self.obs[2]: obs[2],
            	self.obs[3]: obs[3]})
        return value

    def save_network(self, model_name):
        for i in range(len(self.model)):
            tl.files.save_npz(
                self.model[i].all_params,
                name='./ppo/model/{}_val_{}.npz'.format(model_name, i),
                sess=self.sess)

    def load_network(self, model_name):
        for i in range(len(self.model)):
            params = tl.files.load_npz(
                name='./ppo/model/{}_val_{}.npz'.format(model_name, i))
            tl.files.assign_params(self.sess, params, self.model[i])