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
from depthNN import ImageNN


class Agent(object):
    def __init__(self, args, session, obs_shape, ac_shape):
        if not os.path.exists("./ppo/model"):
            os.makedirs("./ppo/model")

        self.policy = Policy(session, obs_shape, ac_shape, args)
        self.value = Value(session, obs_shape)
        self.delta = False

        self.imagenetwork = ImageNN()
        
        self.scan_filter  = RunningAverageFilter(obs_shape[1], obstype="scan",  demean=False, destd=False, update=False, delta=self.delta)
        self.image_filter = RunningAverageFilter([obs_shape[3],obs_shape[4],obs_shape[5]], obstype="image", demean=False, destd=False, update=False, delta=self.delta)
        self.goal_filter  = RunningAverageFilter(obs_shape[2], obstype="goal",  demean=False, destd=False, update=False, delta=self.delta)
        self.vel_filter   = RunningAverageFilter(ac_shape,     obstype="vel",   demean=False, destd=False, update=False, delta=self.delta)
        self.reward_filter= RunningAverageFilter((), demean=False, clip=1)

    def obs_filter(self, obs):
        image_filtered = self.image_filter(obs.ImageObsBatch)
        scan_filtered  = self.scan_filter(obs.scanObsBatch)
        goal_filtered  = self.goal_filter(obs.goalObsBatch)
        vel_filtered   = self.vel_filter(obs.velObsBatch)

        image_out_NN = []
        for i in len(image_filtered):
            image_in_NN = image_filtered[i]
            output = self.imagenetwork.predict_image(image_in_NN)
            image_out_NN.append(output)
        image_out_NN = np.array(image_out_NN)

        if not self.delta:
            scan_filtered /= 4.0
        else:
            scan_filtered /= 4.0
        # scan_filtered = np.asarray(scan_filtered)
        # goal_filtered = np.asarray(goal_filtered)
        # print 'shape: ', np.shape(vel_filtered)
        # print 'vel: ', vel_filtered[0]
        # print 'after shape: ', np.shape(scan_filtered)
        return [ scan_filtered, goal_filtered, vel_filtered, image_out_NN]


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
        #scan = tf.placeholder(tf.float32, [None, self.obs_shape[1], self.obs_shape[0]],'scan_ph')
        goal = tf.placeholder(tf.float32, [None, 2], 'goal_ph')
        vel  = tf.placeholder(tf.float32, [None, 2], 'vel_ph')
        image= tf.placeholder(tf.float32, [None, self.obs_shape[3],self.obs_shape[4],self.obs_shape[5]], 'image_ph')

        # net = tl.layers.InputLayer(scan, name='scan_input')
        # net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[5, self.obs_shape[0], 32], stride=2,name='cnn1')
        # net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[3, 32, 16], stride=2,name='cnn2')
        # net = tl.layers.FlattenLayer(net, name='fl')
        # net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu, name='scan_output')
        # scan_output = net.outputs

        imagenet = tl.layers.InputLayer(image, name='image_input')
        imagenet = tl.layers.Conv2dLayer(imagenet,act=tf.nn.relu,shape=(5, 5, 1, 32),strides=(1,2,2,1),use_cudnn_on_gpu=True,name='Icnn1')
        imagenet = tl.layers.Conv2dLayer(imagenet,act=tf.nn.relu,shape=(3, 3,32, 64),strides=(1,2,2,1),use_cudnn_on_gpu=True,name='Icnn2')
        imagenet = tl.layers.FlattenLayer(imagenet, name='imagefl')
        imagenet = tl.layers.DenseLayer(imagenet, n_units=960, act=tf.nn.relu, name='image_output')
        image_output = imagenet.outputs

        act_net = tl.layers.InputLayer(tf.concat([goal, vel, image_output], axis=1), name='goal_input')
        
        #act_net = tl.layers.DenseLayer(act_net, n_units=64, act=tf.nn.tanh, name='act1')
        act_net = tl.layers.DenseLayer(act_net, n_units=128, act=tf.nn.relu, name='act2')
        linear  = tl.layers.DenseLayer(act_net, n_units=1, act=tf.nn.sigmoid, name='linear')
        angular = tl.layers.DenseLayer(act_net, n_units=1, act=tf.nn.tanh, name='angular')
        with tf.variable_scope('means'):
            action_mean = tf.concat([linear.outputs, angular.outputs], axis=1)

        logvar_speed = (10 * 128) // 48
        log_vars = tf.get_variable('logvars', (logvar_speed, 2), tf.float32,tf.constant_initializer(0.0))
        log_vars = tf.reduce_sum(log_vars, axis=0)

        sampled_act = action_mean + tf.exp(log_vars / 2.0) * tf.random_normal(shape=(2,))

        test_act = action_mean + self.test_var * tf.random_normal(shape=(2,))

        return [act_net,imagenet, linear, angular], [ goal, vel, image],action_mean, log_vars, sampled_act, test_act

    def act(self, obs, terminated, batch=True):
        if not batch:
            obs = [np.expand_dims(x, 0) for x in obs]

        actions = self.sess.run(self.sampled_act, feed_dict={
                self.obs[0]: obs[0],
                self.obs[1]: obs[1],
                self.obs[2]: obs[2]  })

        for i, t in enumerate(terminated):
            if t:
                actions[i] = np.zeros(self.ac_shape)

        return actions

    def act_test(self, obs, terminated):
        actions = self.sess.run(self.test_act, feed_dict={
                self.obs[0]: obs[0],
                self.obs[1]: obs[1],
                self.obs[2]: obs[2]#,
                #self.obs[3]: obs[3]
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
        #scan = tf.placeholder(tf.float32, [None, self.obs_shape[1], self.obs_shape[0]],'scan_value_ph')
        goal = tf.placeholder(tf.float32, [None, 2], 'goal_value_ph')
        vel  = tf.placeholder(tf.float32, [None, 2], 'vel_ph')
        image= tf.placeholder(tf.float32, [None, self.obs_shape[3],self.obs_shape[4],self.obs_shape[5]], 'image_ph')

        # net = tl.layers.InputLayer(scan, name='scan_input_value')
        # net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[5, self.obs_shape[0], 32], stride=2,name='cnn1_value')
        # net = tl.layers.Conv1dLayer(net, act=tf.nn.relu, shape=[3, 32, 16], stride=2,name='cnn2_value')
        # net = tl.layers.FlattenLayer(net, name='fl_value')
        # net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu, name='cnn_output_value')
        # cnn_output = net.outputs

        imagevnet = tl.layers.InputLayer(image, name='image_input_value')
        imagevnet = tl.layers.Conv2dLayer(imagevnet,act=tf.nn.relu,shape=(5, 5, 1, 32),strides=(1,2,2,1),use_cudnn_on_gpu=True,name='Icnnv1')
        #print(imagenet.outputs.shape)
        imagevnet = tl.layers.Conv2dLayer(imagevnet,act=tf.nn.relu,shape=(3, 3,32, 64),strides=(1,2,2,1),use_cudnn_on_gpu=True,name='Icnnv2')
        imagevnet = tl.layers.FlattenLayer(imagevnet, name='imageflv')
        imagevnet = tl.layers.DenseLayer(imagevnet, n_units=960, act=tf.nn.relu, name='image_voutput')
        image_voutput = imagevnet.outputs

        value_net = tl.layers.InputLayer(tf.concat([goal, vel,image_voutput], axis=1),name='goal_input_value')
        
        #value_net = tl.layers.DenseLayer(value_net, n_units=64, act=tf.nn.tanh, name='value1')
        value_net = tl.layers.DenseLayer(value_net, n_units=128, act=tf.nn.relu, name='value2')
        value_net = tl.layers.DenseLayer(value_net, n_units=1, name='value')
        value = value_net.outputs

        return [value_net,imagevnet], [goal, vel, image], value

    def update(self, obs, returns):
        feed_dict = {
            self.obs[0]: obs[0],
            self.obs[1]: obs[1],
            self.obs[2]: obs[2],
            #self.obs[3]: obs[3],
            self.ret_ph: returns}

        self.sess.run(self.optimizer, feed_dict)

    def predict(self, obs):
        value = self.sess.run(
            self.value,
            feed_dict={
                self.obs[0]: obs[0],
                self.obs[1]: obs[1],
                self.obs[2]: obs[2]#,
            	#self.obs[3]: obs[3]
                })
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
