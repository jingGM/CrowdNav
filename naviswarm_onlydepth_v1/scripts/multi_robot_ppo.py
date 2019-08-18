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

import argparse
import itertools
import time
import csv
from collections import defaultdict

import numpy as np
import rospy
import tensorflow as tf
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from tabulate import tabulate

from ppo.agent import Agent
from ppo.ppo import PPO
from stage_env import StageEnv

parser = argparse.ArgumentParser(
    description='Multi-Robot Collision Avoidance with Local Sensing via'
    'Deep Reinforcement Learning')

parser.add_argument(
    '--train', default=True, type=bool, help='train or test')
parser.add_argument(
    '--num_agents', default=4, type=int, help='number of robots')
parser.add_argument(
    '--num_obstacles', default=0, type=int, help='number of obstacles')
parser.add_argument(
    '--agent_radius', default=0.12, type=float, help='radius of the robot')
parser.add_argument(
    '--max_vx', default=0.5, type=float, help='max vx')
parser.add_argument(
    '--env_size', default=2, type=float, help='size of environment')


parser.add_argument(
    '--gamma', default=0.99, type=float, help='discount factor')
parser.add_argument(
    '--lamda', default=0.95, type=float, help='gae')

# ppo
parser.add_argument(
    '--kl_target', default=0.0015, type=float)
parser.add_argument(
    '--beta', default=1., type=float)
parser.add_argument(
    '--eta', default=50., type=float)
parser.add_argument(
    '--actor_lr', default=1e-3, type=float)
parser.add_argument(
    '--actor_epochs', default=20, type=int)
parser.add_argument(
    '--critic_lr', default=1e-3, type=float)
parser.add_argument(
    '--critic_epochs', default=10, type=int)
parser.add_argument(
    '--lr_multiplier', default=1., type=float)

parser.add_argument(
    '--seed', default=333, type=int, help='random seed')
parser.add_argument(
    '--test_var', default=0., type=float, help='variance for test')

parser.add_argument(
    '--train_max_steps',
    default=3000000,
    type=int,
    help='max timesteps of the whole training')
parser.add_argument(
    '--batch_max_steps',
    default=200, #8000,
    type=int,
    help='max timesteps of a batch for updating')
parser.add_argument(
    '--episode_max_steps',
    default=100, #400,
    type=int,
    help='max timesteps of an episode')
parser.add_argument(
    '--train_max_iters',
    default=500, #4000,
    type=int,
    help='maximum training iterations')
parser.add_argument(
    '--load_network',
    default=False,
    type=bool,
    help='whether to load pretrained networks')

args = parser.parse_args()


class MultiRobotDRL(object):
    def __init__(self, env, agent, alg):
        self.env = env
        self.agent = agent
        self.alg = alg

        if args.load_network:
            self.agent.value.load_network( 'last')
            self.agent.policy.load_network('last')

        self.reward_pub = rospy.Publisher('/drl/reward', numpy_msg(Floats), queue_size=1)

        self.num_agents = args.num_agents
        self.episodes_counter = 0

    def _rollout(self, env):
        # use multiple agents to collect paths in one episode
        # n: number of agents
        terminateds = [False for _ in range(self.num_agents)]
        terminate_idxs = [0 for _ in range(self.num_agents)]
        terminate_flag = [False for _ in range(self.num_agents)]
        paths = defaultdict(list)

        obs_agents = env.reset()

        for step in range(args.episode_max_steps):
            env.render()
            # receive the obs_of_agents sent by stage
            # check status of the agent: reached_goal, collision, over_run
            # if running (not done): act(obs)

            #print(obs_agents.scanObsBatch)
            #print("============scan length=============================")
            obs_agents = self.agent.obs_filter(obs_agents)

            scan_input = obs_agents[0]
            goal_input = obs_agents[1]
            vel_input  = obs_agents[2]
            image_input= obs_agents[3]

            paths["obs_scan"].append(scan_input)
            paths["obs_goal"].append(goal_input)
            paths["obs_vel" ].append(vel_input)
            paths["obs_image"].append(image_input)

            #print(scan_input)
            #print("============scan=============================")
            #print(goal_input)
            #print(goal_input.shape)
            #print("============goal=============================")
            #print(vel_input)
            #print("============velocity=========================")
            #print(terminateds)
            #print("============terminats========================")

            if args.train:
                action_agents = self.agent.policy.act([goal_input, vel_input, image_input], terminateds)
            else:
                action_agents = self.agent.policy.act_test([goal_input, vel_input,image_input], terminateds)
            paths["action"].append(action_agents)

            #print(action_agents)
            #print("===agent actions===")
            obs_agents, reward_agents, done_agents, _ = env.step(action_agents,step)
            #print(reward_agents)
            #print("===agents reward===")
            if step == 0:
                reward_agents = np.zeros(args.num_agents)
            paths["reward"].append(reward_agents)
            self.plot_reward(np.asarray(reward_agents))

            for i, d in enumerate(done_agents):
                if d :
                    terminateds[i] = True
                    if terminate_flag[i] is False:
                        terminate_idxs[i] += 1
                    terminate_flag[i] = True
                else:
                    terminate_idxs[i] += 1
            if all(terminateds):
                break

        path_agents = []
        for i in range(self.num_agents):
            path = defaultdict(list)
            for k, v in paths.items():
                v = np.asarray(v)
                # print 'k: ', k, '   v: ', v[:terminate_idxs[i], i]
                path[k] = np.array(v[:terminate_idxs[i], i])
                path["terminated"] = terminateds[i]
            path["done_id"] = terminate_idxs[i]
            path_agents.append(path)

            #print "path_agents:", len(path_agents), "/ ", len(path_agents[0]["reward"])

        return path_agents

    def plot_reward(self, rewards):
        rewards = np.array(rewards, dtype=np.float32)
        self.reward_pub.publish(rewards)

    def _get_paths(self, seed_iter):
        paths_batch = []
        timesteps_counter = 0
        while True:
            np.random.seed(seed_iter.next())
            self.episodes_counter += 1
            print("***** Episode {} *****".format(self.episodes_counter))
            path_agents = self._rollout(env)
            for path in path_agents:
                paths_batch.append(path)
                #print "paths_batch:", len(paths_batch), "/ ", len(paths_batch[0])
                timesteps_counter += len(path["reward"])

            if timesteps_counter > args.batch_max_steps or args.train is False:
                break

        return paths_batch

    def _print_statistics(self, stats):
        print("*********** Iteration {} ************".format(stats["Iteration"]))
        print(tabulate(
            filter(lambda (k, v): np.asarray(v).size == 1, stats.items()),
            tablefmt="grid"))

    def run(self):
        tstart = time.time()
        seed_iter = itertools.count()

        iterCounter = 0
        while iterCounter < args.train_max_iters and not rospy.is_shutdown():
            iterCounter += 1
            paths = self._get_paths(seed_iter)
            #print("--{}--".format(paths))
            #print("=== path ===")
            if args.train:
                stats = self.alg.update(paths)
                #print("in training===")
            else:
                stats, succ_agent = self.alg.test(paths)
                stats["MeanDistance"] = (self.env.perfect_distance * succ_agent).sum() / succ_agent.sum()
                stats["MeanTrajectory"] = (self.env.trajectory * succ_agent).sum() / succ_agent.sum()
                stats["MeanVelocity-D"] = stats["MeanDistance"] / (stats["EpPathLengthsMean"] / 10.)
                stats["MeanVelocity-T"] = stats["MeanTrajectory"] / (stats["EpPathLengthsMean"] / 10.)
                stats["MeanTime"] = (stats["EpPathLengthsMean"] / 10.)
                stats["ExtraTime"] = stats["MeanTime"] - stats["MeanDistance"] / args.max_vx

            stats["TimeElapsed"] = time.time() - tstart
            stats["Iteration"] = iterCounter

            self._print_statistics(stats)

if __name__ == "__main__":
    rospy.init_node("multi_robot_drl_stage")

    # if args.seed > 0:
    #    np.random.seed(args.seed)

    # set tf graph and session
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    session = tf.Session(graph=graph, config=config)

    # initialize env, agent and algorithm
    env = StageEnv(
        args.num_agents, args.num_obstacles,
        args.agent_radius, args.env_size, args.max_vx)

    #print(env.image_space.shape)
    #print("+++++++++++++++++++++++++++++++++++++")
    obs_shape = [3, env.scan_space.shape[0], env.goal_space.shape[0],3,env.image_space.shape[0],env.image_space.shape[1]]
    ac_shape = env.action_space.shape[0]

    agent = Agent(args, session, obs_shape, ac_shape)
    alg = PPO(args, agent, session, obs_shape, ac_shape)

    learner = MultiRobotDRL(env, agent, alg)
    learner.run()
