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
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from utils import discount


class PPO(object):
    def __init__(self, args, agent, session, obs_shape, ac_shape):
        self.session = session
        self.args = args

        self.act_dim = ac_shape
        self.time_step = 0
        self.best_score = 0

        # actor param
        self.beta = args.beta
        self.eta = args.eta
        self.kl_targ = args.kl_target
        self.actor_epochs = args.actor_epochs
        self.actor_lr = args.actor_lr
        self.lr_multiplier = args.lr_multiplier

        # critic param
        self.replay_buffer_obs_scan = None
        self.replay_buffer_obs_goal = None
        self.replay_buffer_obs_vel = None
        self.replay_buffer_ret = None
        self.critic_epochs = args.critic_epochs
        self.critic_lr = args.critic_lr

        self.obs_scan, self.obs_goal, self.obs_vel = agent.policy.obs
        self.actor = agent.policy
        self.means = agent.policy.means
        self.log_vars = agent.policy.log_vars

        self.obs_scan_value, self.obs_goal_value, self.obs_vel_value = \
            agent.value.obs
        self.baseline = agent.value
        self.value = agent.value.value

        # save network

        self._placeholder()
        self._build_actor_training_method()
        self._build_critic_training_method()
        self._build_tensorboard()

        self.merge_all = tf.summary.merge_all()

        timeString = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not os.path.exists("./ppo/summary"):
            os.makedirs("./ppo/summary")
        self.writer = tf.summary.FileWriter(
            './ppo/summary/{}'.format(timeString), self.session.graph)

        self.session.run(tf.global_variables_initializer())

    def _placeholder(self):
        self.act_ph = tf.placeholder(
            tf.float32, [None, self.act_dim], 'act_ph')
        self.advantages_ph = tf.placeholder(
            tf.float32, [None, ], 'advantages_ph')

        self.old_log_vars_ph = tf.placeholder(
            tf.float32, [self.act_dim, ], 'old_log_vars')
        self.old_means_ph = tf.placeholder(
            tf.float32, [None, self.act_dim], 'old_means')

        self.beta_ph = tf.placeholder(tf.float32, name='beta')
        self.eta_ph = tf.placeholder(tf.float32, name='eta')
        self.lr_ph = tf.placeholder(tf.float32, name='lr')

        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')

    def _build_actor_training_method(self):
        # compute logprob
        self.logp = -0.5 * tf.reduce_sum(
            self.log_vars) + -0.5 * tf.reduce_sum(
                tf.square(self.act_ph - self.means) / tf.exp(self.log_vars),
                axis=1)
        self.logp_old = -0.5 * tf.reduce_sum(
            self.old_log_vars_ph) + -0.5 * tf.reduce_sum(
                tf.square(self.act_ph - self.old_means_ph) / tf.exp(self.old_log_vars_ph),
                axis=1)
        # compute kl
        with tf.variable_scope('kl'):
            self.kl = 0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    tf.exp(self.old_log_vars_ph - self.log_vars)) +
                tf.reduce_sum(
                    tf.square(self.means - self.old_means_ph) / tf.exp(self.log_vars),
                    axis=1) - self.act_dim + tf.reduce_sum(self.log_vars) -
                tf.reduce_sum(self.old_log_vars_ph))
        # compute entropy
        with tf.variable_scope('entropy'):
            self.entropy = 0.5 * (
                self.act_dim *
                (np.log(2 * np.pi) + 1) + tf.reduce_sum(self.log_vars))
        # compute actor loss
        with tf.variable_scope('actor_loss'):
            loss1 = -tf.reduce_mean(
                self.advantages_ph * tf.exp(self.logp - self.logp_old))
            loss2 = tf.reduce_mean(self.beta_ph * self.kl)
            loss3 = self.eta_ph * tf.square(
                tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))
            self.actor_loss = loss1 + loss2 + loss3
        # opt actor loss
        self.actor_opt = tf.train.AdamOptimizer(self.lr_ph).minimize(
            self.actor_loss)

    def _build_critic_training_method(self):
        with tf.variable_scope('critic_loss'):
            self.critic_loss = tf.reduce_mean(
                tf.square(tf.squeeze(self.value) - self.ret_ph))

    def _build_tensorboard(self):
        self.visual_reward = tf.placeholder(
            tf.float32, name="visual_reward")
        self.visual_kl = tf.placeholder(tf.float32, name="visual_kl")

        with tf.name_scope('param'):
            tf.summary.scalar('reward', self.visual_reward)
            tf.summary.scalar('kl', self.visual_kl)
            tf.summary.scalar('entropy', self.entropy)
            tf.summary.scalar('beta', self.beta_ph)
            tf.summary.scalar('actor_lr', self.lr_ph)

        with tf.name_scope('loss'):
            tf.summary.scalar('critic_loss', self.critic_loss)

    def update(self, paths):
        self.time_step += 1

        acts = np.concatenate([path["action"] for path in paths])
        obs_scan = np.concatenate([path["obs_scan"] for path in paths])
        obs_goal = np.concatenate([path["obs_goal"] for path in paths])
        obs_vel = np.concatenate([path["obs_vel"] for path in paths])

        baseline_value = self.baseline.predict(
            [obs_scan, obs_goal, obs_vel])

        last_path_size = 0
        for _, path in enumerate(paths):
            np.array(path["reward"])
            path["return"] = discount(path["reward"], self.args.gamma)
            b = path["baseline"] = baseline_value[
                last_path_size:last_path_size + path["done_id"]]
            b1 = np.append(b, 0 if path["terminated"] else b[-1])
            deltas = path["reward"] + self.args.gamma * b1[1:] - b1[:-1]
            path["advantage"] = discount(
                deltas, self.args.gamma * self.args.lamda)
            last_path_size = path["done_id"]

        rets = np.concatenate([path["return"] for path in paths])
        advs = np.concatenate([path["advantage"] for path in paths])
        advs = (advs - advs.mean()) / (advs.std() + 1e-6)

        if self.time_step > 1: # train acotr after trained critic
            kl = self.actor_update(obs_scan, obs_goal, obs_vel, acts, advs)
        self.critic_update(obs_scan, obs_goal, obs_vel, rets)

        stats = OrderedDict()

        epRewards = np.array([path["reward"].sum() for path in paths])
        epPathLengths = np.array([len(path["reward"]) for path in paths])
        stats["EpRewardsMean"] = epRewards.mean()
        stats["EpRewardsMax"] = epRewards.max()
        stats["EpRewardsMin"] = epRewards.min()
        stats["EpPathLengthsMean"] = epPathLengths.mean()
        stats["EpPathLengthsMax"] = epPathLengths.max()
        stats["EpPathLengthsMin"] = epPathLengths.min()
        stats["RewardPerStep"] = epRewards.sum() / epPathLengths.sum()
        if self.time_step > 1:
            stats["Beta"] = self.beta
            stats["ActorLearningRate"] = self.actor_lr * self.lr_multiplier
            stats["KL-Divergence"] = kl

            feed_dict = {
                self.obs_scan: obs_scan,
                self.obs_goal: obs_goal,
                self.obs_vel: obs_vel,
                self.obs_scan_value: obs_scan,
                self.obs_goal_value: obs_goal,
                self.obs_vel_value: obs_vel,
                self.act_ph: acts,
                self.advantages_ph: advs,
                self.beta_ph: self.beta,
                self.eta_ph: self.eta,
                self.lr_ph: self.actor_lr * self.lr_multiplier,
                self.ret_ph: rets,
                self.visual_kl: kl,
                self.visual_reward: epRewards.mean()
            }

            summary = self.session.run(self.merge_all, feed_dict)
            self.writer.add_summary(summary, self.time_step)

        if epRewards.mean() > self.best_score:
            self.actor.save_network('best')
            self.baseline.save_network('best')
            self.best_score = epRewards.mean()

        self.actor.save_network('last')
        self.baseline.save_network('last')

        return stats

    def test(self, paths):
        stats = OrderedDict()

        epRewards = np.array([path["reward"].sum() for path in paths])
        epPathLengths = np.array([len(path["reward"]) for path in paths])
        succ_agent = np.zeros(len(epRewards))
        for i, ep in enumerate(epRewards):
            if ep > 20:
                succ_agent[i] = 1

        stats["SuccessNum"] = succ_agent.sum()
        stats["SuccessRate"] = succ_agent.sum() / succ_agent.shape[0]
        stats["EpRewardsMean"] = epRewards.mean()
        stats["EpRewardsMax"] = epRewards.max()
        stats["EpRewardsMin"] = epRewards.min()
        stats["EpPathLengthsMean"] = (epPathLengths * succ_agent).sum() / succ_agent.sum()
        stats["EpPathLengthsMax"] = (epPathLengths * succ_agent).max()
        stats["EpPathLengthsMin"] = (epPathLengths * succ_agent).min()

        return stats, succ_agent

    def actor_update(self, obs_scan, obs_goal, obs_vel, acts, advs):
        feed_dict = {
            self.obs_scan: obs_scan,
            self.obs_goal: obs_goal,
            self.obs_vel: obs_vel,
            self.act_ph: acts,
            self.advantages_ph: advs,
            self.beta_ph: self.beta,
            self.eta_ph: self.eta,
            self.lr_ph: self.actor_lr * self.lr_multiplier
        }

        old_means_np, old_log_vars_np = self.session.run(
            [self.means, self.log_vars], feed_dict)

        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np

        for e in range(self.actor_epochs):
            self.session.run(self.actor_opt, feed_dict)
            kl = self.session.run(self.kl, feed_dict)
            if kl > self.kl_targ * 4:  # early stopping
                break

        if kl > self.kl_targ * 2:
            self.beta = np.minimum(35, 1.5 * self.beta)
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2.0:
            self.beta = np.maximum(1.0 / 35.0, self.beta / 1.5)
            if self.beta < (1.0 / 30.0) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        return kl

    def critic_update(self, obs_scan, obs_goal, obs_vel, rets):
        num_batches = max(obs_scan.shape[0] // 256, 1)
        batch_size = obs_scan.shape[0] // num_batches
        if self.replay_buffer_obs_scan is None:
            obs_scan_train, obs_goal_train, obs_vel_train, ret_train = \
                obs_scan, obs_goal, obs_vel, rets
        else:
            obs_scan_train = np.concatenate(
                [obs_scan, self.replay_buffer_obs_scan])
            obs_goal_train = np.concatenate(
                [obs_goal, self.replay_buffer_obs_goal])
            obs_vel_train = np.concatenate(
                [obs_vel, self.replay_buffer_obs_vel])
            ret_train = np.concatenate([rets, self.replay_buffer_ret])

        self.replay_buffer_obs_scan = obs_scan
        self.replay_buffer_obs_goal = obs_goal
        self.replay_buffer_obs_vel = obs_vel
        self.replay_buffer_ret = rets

        for e in range(self.critic_epochs):
            obs_scan_train, obs_goal_train, obs_vel_train, ret_train = shuffle(
                obs_scan_train, obs_goal_train, obs_vel_train, ret_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                obs_scan_set = obs_scan_train[start:end, :, :]
                obs_goal_set = obs_goal_train[start:end, :]
                obs_vel_set = obs_vel_train[start:end, :]
                ret_set = ret_train[start:end]

                self.baseline.update(
                    [obs_scan_set, obs_goal_set, obs_vel_set], ret_set)
