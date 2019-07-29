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

import math

import numpy as np

from gym.utils import seeding


class Scenarios(object):
    def __init__(self, num_agents, num_obstacles, agent_size, env_size):
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.agent_size = agent_size
        self.env_size = env_size

        self._seed()
        self.reset()

    def reset(self):
        self.starts = []
        self.goals = []

    def mit_crossing_scene(self, alpha):
        alpha = math.radians(alpha)
        self.starts = [[-4., 0., 0.],
                       [4.*math.cos(alpha),
                        -4.*math.sin(alpha),
                        math.radians(180)-alpha]]

        self.goals = [[4., 0., 0.],
                      [-4.*math.cos(alpha),
                       4.*math.sin(alpha), 0.]]

        return self.starts, self.goals

    def crossing_scene(self, distance):
        assert self.num_agents == 8

        self.starts = [[distance/2., 0.5, math.radians(180)],
                       [distance/2., -0.5, math.radians(180)],
                       [distance/2.+1., 0.5, math.radians(180)],
                       [distance/2.+1., -0.5, math.radians(180)],
                       [0.5, distance/2., math.radians(-90)],
                       [-0.5, distance/2., math.radians(-90)],
                       [0.5, distance/2.+1. , math.radians(-90)],
                       [-0.5, distance/2.+1., math.radians(-90)],
                      ]

        self.goals = [[-distance/2.-1., 0.5, math.radians(180)],
                       [-distance/2.-1., -0.5, math.radians(180)],
                       [-distance/2., 0.5, math.radians(180)],
                       [-distance/2., -0.5, math.radians(180)],
                       [0.5, -distance/2.-1., math.radians(-90)],
                       [-0.5, -distance/2.-1., math.radians(-90)],
                       [0.5, -distance/2., math.radians(-90)],
                       [-0.5, -distance/2., math.radians(-90)],
                      ]

        return self.starts, self.goals

    def crossing_with_obstacle_scene(self, distance):
        assert self.num_agents == 8
        assert self.num_obstacles == 1

        self.starts = [[distance/2., 0.5, math.radians(180)],
                       [distance/2., -0.5, math.radians(180)],
                       [distance/2.+1., 0.5, math.radians(180)],
                       [distance/2.+1., -0.5, math.radians(180)],
                       [0.5, distance/2., math.radians(-90)],
                       [-0.5, distance/2., math.radians(-90)],
                       [0.5, distance/2.+1. , math.radians(-90)],
                       [-0.5, distance/2.+1., math.radians(-90)],
                       [0., 0., 0.]
                      ]

        self.goals = [[-distance/2.-1., 0.5, math.radians(180)],
                       [-distance/2.-1., -0.5, math.radians(180)],
                       [-distance/2., 0.5, math.radians(180)],
                       [-distance/2., -0.5, math.radians(180)],
                       [0.5, -distance/2.-1., math.radians(-90)],
                       [-0.5, -distance/2.-1., math.radians(-90)],
                       [0.5, -distance/2., math.radians(-90)],
                       [-0.5, -distance/2., math.radians(-90)],
                       [0., 0., 0.]
                      ]

        return self.starts, self.goals


    def random_scene(self):
        for _ in range(self.num_agents):
            succ = False
            sx, sy, sa, gx, gy, ga = 0., 0., 0., 0., 0., 0.
            while not succ:
                sx, sy, gx, gy = self.np_random.uniform(-self.env_size,
                                                        self.env_size, 4)
                sa, ga = self.np_random.uniform(-np.pi, np.pi, 2)
                succ = True

                if np.hypot(gx - sx, gy - sy) < 5.0:
                    succ = False

                if self.starts:
                    for s in self.starts:
                        if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 10.:
                            succ = False

                if self.goals:
                    for g in self.goals:
                        if np.hypot(gx - g[0], gy - g[1]) < self.agent_size * 10.:
                            succ = False

            self.starts.append([sx, sy, sa])
            self.goals.append([gx, gy, ga])

        return self.starts, self.goals

    def random_obstacles_scene(self):
        iter_num = 0
        for _ in range(self.num_agents + self.num_obstacles):
            succ = False
            sx, sy, sa, gx, gy, ga = 0., 0., 0., 0., 0., 0.
            iter_num += 1
            while not succ:
                sx, sy, gx, gy = self.np_random.uniform(
                    -self.env_size, self.env_size, 4)
                sa, ga = self.np_random.uniform(-np.pi, np.pi, 2)
                succ = True

                if iter_num <= self.num_agents:
                    if np.hypot(gx - sx, gy - sy) < 5.0:
                        succ = False
                else:
                    if abs(sx) > 2. or abs(sy) > 2.:
                        succ = False

                if self.starts:
                    for i, s in enumerate(self.starts):
                        if i > self.num_agents - 1:
                            if np.hypot(sx-s[0], sy-s[1]) < self.agent_size*3.:
                                succ = False

                        else:
                            if np.hypot(sx-s[0], sy-s[1]) < self.agent_size*6.:
                                succ = False

                    if len(self.starts) > self.num_agents - 1:
                        for i, g in enumerate(self.goals):
                            if np.hypot(sx - g[0], sy - g[1]) < self.agent_size*3.:
                                succ = False

                if self.goals:
                    for g in self.goals:
                        if np.hypot(gx-g[0], gy-g[1]) < self.agent_size*6.:
                            succ = False
                    if len(self.starts) > self.num_agents:
                        for i in range(len(self.starts) - self.num_agents):
                            if np.hypot(gx - self.starts[self.num_agents + i - 1][0], gy - self.starts[self.num_agents + i - 1][1]) < self.agent_size*6.:
                                succ = False

            self.starts.append([sx, sy, sa])
            self.goals.append([gx, gy, ga])

        return self.starts, self.goals

    def circle_scene(self):
        for _ in range(self.num_agents):
            succ = False
            sx, sy, sa = 0., 0., 0.
            while not succ:
                angle = self.np_random.uniform(0., 2*np.pi)
                sx = self.env_size * np.cos(angle)
                sy = self.env_size * np.sin(angle)
                sa = np.arctan2(-sy, -sx)
                succ = True

                if self.starts:
                    for s in self.starts:
                        if np.hypot(sx - s[0], sy - s[1]) < self.agent_size*4+0.15:
                            succ = False

            self.starts.append([sx, sy, sa])
            self.goals.append([-sx, -sy, sa])

        return self.starts, self.goals

    def circle_scene_with_obstacles(self):
        for _ in range(self.num_agents):
            succ = False
            sx, sy, sa = 0., 0., 0.
            while not succ:
                angle = self.np_random.uniform(0., 2*np.pi)
                sx = self.env_size * np.cos(angle)
                sy = self.env_size * np.sin(angle)
                sa = np.arctan2(-sy, -sx)
                succ = True

                if self.starts:
                    for s in self.starts:
                        if np.hypot(sx - s[0], sy - s[1]) < self.agent_size*4+0.15:
                            succ = False

            self.starts.append([sx, sy, sa])
            self.goals.append([-sx, -sy, sa])


        for _ in range(self.num_obstacles):
            succ = False
            sx, sy, sa = 0., 0., 0.
            while not succ:
                sa = self.np_random.uniform(0., 2*np.pi)
                sx, sy = self.np_random.uniform(-1.5, 1.5, 2)
                succ = True

                for s in self.starts:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size*5.:
                        succ = False

            self.starts.append([sx, sy, sa])
            self.goals.append([sx, sy, sa])

        return self.starts, self.goals


    def circle_scene_uniform(self):
        for i in range(self.num_agents):
            succ = False
            sx, sy, sa = 0., 0., 0.
            while not succ:
                angle = i*2*np.pi / self.num_agents
                sx = self.env_size * np.cos(angle)
                sy = self.env_size * np.sin(angle)
                sa = np.arctan2(-sy, -sx)
                succ = True

                if self.starts:
                    for s in self.starts:
                        if np.hypot(sx - s[0], sy - s[1]) < self.agent_size*4+0.15:
                            succ = False

            self.starts.append([sx, sy, sa])
            self.goals.append([-sx, -sy, sa])

        return self.starts, self.goals

    def no_obstacle_scene(self):
        assert self.num_agents == 16
        assert self.num_obstacles == 0

        s_x = -18
        s_y = -18
        for i in range(4):
            for j in range(4):
                angle, theta = self.np_random.uniform(-np.pi, np.pi, 2)
                distance = self.np_random.uniform(0.3, 3, 1)[0]
                self.starts.append([s_x+i*12, s_y+j*12, angle])
                self.goals.append([s_x+i*12+distance*np.cos(theta), s_y+j*12+distance*np.sin(theta), 0.])

        return self.starts, self.goals

    def corridor_scene(self):
        assert self.num_agents == 12
        assert self.num_obstacles == 12

        self.starts = [[2.7, 0.4, math.radians(-180)],
                       [2.7, -0.4, math.radians(-180)],
                       [-2.7, 0.4, math.radians(0)],
                       [-2.7, -0.4, math.radians(0)],
                       [3.7, 0.4, math.radians(-180)],
                       [3.7, -0.4, math.radians(-180)],
                       [4.7, 0.4, math.radians(-180)],
                       [4.7, -0.4, math.radians(-180)],
                       [-3.7, 0.4, math.radians(0)],
                       [-3.7, -0.4, math.radians(0)],
                       [-4.7, 0.4, math.radians(0)],
                       [-4.7, -0.4, math.radians(0)],
                       [-2.5, 1.2, 0.0],
                       [-2.5, -1.2, 0.0],
                       [2.5, 1.2, 0.0],
                       [2.5, -1.2, 0.0],
                       [5., 0., math.radians(90)],
                       [-5., 0., math.radians(90)],
                       [-1.3, -.6, 0.],
                       [1.3, .6, 0.],
                       [-.8, -.85, math.radians(90)],
                       [-1.8, -.85, math.radians(90)],
                       [.8, .85, math.radians(90)],
                       [1.8, .85, math.radians(90)],
                      ]

        self.goals = [[-4.7, 0.4, math.radians(-180)],
                       [-4.7, -0.4, math.radians(-180)],
                       [4.7, 0.4, math.radians(0)],
                       [4.7, -0.4, math.radians(0)],
                       [-3.7, 0.4, math.radians(-180)],
                       [-3.7, -0.4, math.radians(-180)],
                       [-2.7, 0.4, math.radians(-180)],
                       [-2.7, -0.4, math.radians(-180)],
                       [3.7, 0.4, math.radians(0)],
                       [3.7, -0.4, math.radians(0)],
                       [2.7, 0.4, math.radians(0)],
                       [2.7, -0.4, math.radians(0)],
                       [-2.5, 0.95, 0.0],
                       [-2.5, -0.95, 0.0],
                       [2.5, 0.95, 0.0],
                       [2.5, -0.95, 0.0],
                       [5., 0., math.radians(90)],
                       [-5., 0., math.radians(90)],
                       [-1., -.7, 0.],
                       [1., .7, 0.],
                       [-.5, -.95, math.radians(90)],
                       [-1.5, -.95, math.radians(90)],
                       [.5, .95, math.radians(90)],
                       [1.5, .95, math.radians(90)],
                     ]

        return self.starts, self.goals

    def evacuation_scene(self):
        assert self.num_agents == 6
        assert self.num_obstacles == 9

        self.starts = [[0., -1., math.radians(90.)],
                       [.4, -1., math.radians(90.)],
                       [-.4, -1., math.radians(90.)],
                       [.0, -1.8, math.radians(90.)],
                       [-.4, -1.8, math.radians(90.)],
                       [.4, -1.8, math.radians(901.)],
                       [-1.5, 0., math.radians(0.)],
                       [1.5, 0., math.radians(0.)],
                       [-2.5, 0., math.radians(90.)],
                       [2.5, 0., math.radians(90.)],
                       [1.5, -2., math.radians(0.)],
                       [-1.5, -2., math.radians(0.)],
                       [.0, -2., math.radians(0.)],
                       [-2.5, 4., math.radians(90.)],
                       [2.5, 4., math.radians(90.)],
                    ]

        self.goals = [[0., 3.5, math.radians(90.)],
                       [-.8, 3.5, math.radians(105.)],
                       [.8, 3.5, math.radians(75.)],
                       [.0, 2.5, math.radians(90.)],
                       [.8, 2.5, math.radians(75.)],
                       [-.8, 2.5, math.radians(105.)],
                      [-3.5, 2., math.radians(0.)],
                       [2.5, 2., math.radians(0.)],
                       [-6., 3., math.radians(90.)],
                       [5., 3., math.radians(90.)],
                       [3.5, -2., math.radians(0.)],
                       [-.5, -2., math.radians(0.)],
                       [2.5, -2., math.radians(0.)],
                       [-6., 4., math.radians(90.)],
                       [5., 4., math.radians(90.)],
                     ]

        return self.starts, self.goals

    def moving_car_scene(self):
        assert self.num_agents == 12

        self.starts = [[-3., 3., math.radians(-45)],
                       [-3., -3., math.radians(45)],
                       [3., 3., math.radians(-135)],
                       [3., -3., math.radians(135)],
                       [-3., -1., math.radians(0)],
                       [-3., -2., math.radians(0)],
                       [3., 1., math.radians(180)],
                       [3., 2., math.radians(180)],
                       [-1., 3., math.radians(-90)],
                       [-2., 3., math.radians(-90)],
                       [1., -3., math.radians(90)],
                       [2., -3., math.radians(90)],
                        ]

        self.goals = [[3.5, -3.5, math.radians(-45)],
                       [3.5, 3.5, math.radians(45)],
                       [-3.5, -3.5, math.radians(-135)],
                       [-3.5, 3.5, math.radians(135)],
                       [3., -1., math.radians(0)],
                       [3., -2., math.radians(0)],
                       [-3., 1., math.radians(0)],
                       [-3., 2., math.radians(0)],
                       [-1., -3., math.radians(-90)],
                       [-2., -3., math.radians(-90)],
                       [1., 3., math.radians(90)],
                       [2., 3., math.radians(90)],
                        ]

        return self.starts, self.goals

    def face_car_scene(self):
        assert self.num_agents == 8

        self.starts = [[-.5, -2., math.radians(90)],
                       [-.5, -3., math.radians(90)],
                       [.5, -2., math.radians(90)],
                       [.5, -3., math.radians(90)],
                       [-3., .5, 0.],
                       [-3., -.5, 0.],
                       [0.7, 5., math.radians(-90)],
                       [-0.7, 5., math.radians(-90)]
                      ]

        self.goals = [[-.5, 2., math.radians(90)],
                       [-.5, 3., math.radians(90)],
                       [.5, 2., math.radians(90)],
                       [.5, 3., math.radians(90)],
                       [3., .5, 0.],
                       [3., -.5, 0.],
                       [0.7, -5., math.radians(-90)],
                       [-0.7, -5., math.radians(-90)]
                      ]

        return self.starts, self.goals

    def cross_road_scene(self):
        assert self.num_agents == 11

        self.starts = [[-1.5, 0., math.radians(90)],
                       [0., 0., math.radians(90)],
                       [1.5, 0., math.radians(90)],
                       [-4.0, 4., 0.0],
                       [-5.0, 5., 0.0],
                       [-6.0, 6., 0.0],
                       [-7.0, 7., 0.0],
                       [4.5, 4.5, math.radians(180)],
                       [5.5, 5.5, math.radians(180)],
                       [6.5, 6.5, math.radians(180)],
                       [7.5, 7.5, math.radians(180)]
                      ]

        self.goals = [[1.5, 8.5, math.radians(90)],
                       [0., 8.5, math.radians(90)],
                       [-1.5, 8.5, math.radians(90)],
                       [4.0, 4., 0.0],
                       [5.0, 5., 0.0],
                       [6.0, 6., 0.0],
                       [7.0, 7., 0.0],
                       [-4.5, 4.5, math.radians(180)],
                       [-5.5, 5.5, math.radians(180)],
                       [-6.5, 6.5, math.radians(180)],
                       [-7.5, 7.5, math.radians(180)]
                      ]

        return self.starts, self.goals

    def pentagon_scene(self, radius):
        assert self.num_agents == 5

        points = [[radius * math.sin(math.radians(36)), -radius * math.cos(math.radians(36)), math.radians(126)],
                  [radius * math.sin(math.radians(72)), radius * math.cos(math.radians(72)), math.radians(198)],
                  [0, radius, math.radians(270)],
                  [-radius * math.sin(math.radians(72)), radius * math.cos(math.radians(72)), math.radians(-18)],
                  [-radius * math.sin(math.radians(36)), -radius * math.cos(math.radians(36)), math.radians(54)]]

        for i in range(5):
            self.starts.append(points[i])
            self.goals.append(points[(i + 2) % 5])

        return self.starts, self.goals

    def four_cross_scene(self):
        assert self.num_agents == 4

        self.starts = [[3., 0., math.radians(180)],
                       [0., 3., math.radians(-90)],
                       [-3., 0., math.radians(0)],
                       [0., -3., math.radians(90)],
                       [-2.75, 0.75, 0],
                       [-2.75, -0.75, 0],
                       [2.75, 0.75, 0],
                       [2.75, -0.75, 0],
                       [-0.75, -2.75, math.radians(90)],
                       [0.75, -2.75, math.radians(90)],
                       [-0.75, 2.75, math.radians(90)],
                       [0.75, 2.75, math.radians(90)],
                       [0, 4.75, 0],
                       [0, -4.75, 0],
                       [-4.75, 0, math.radians(90)],
                       [4.75, 0, math.radians(90)]]

        self.goals = [[-3., 0., math.radians(180)],
                       [0., -3., math.radians(-90)],
                       [3., 0., math.radians(0)],
                       [0., 3., math.radians(90)],
                       [2.75, 0.75, 0],
                       [2.75, -0.75, 0],
                       [2.75, 0.75, 0],
                       [2.75, -0.75, 0],
                       [0.75, -2.75, math.radians(90)],
                       [0.75, 2.75, math.radians(90)],
                       [0.75, 2.75, math.radians(90)],
                       [-0.75, 2.75, math.radians(90)],
                       [0, 4.75, 0],
                       [0, -4.75, 0],
                       [-4.75, 0, math.radians(90)],
                       [4.75, 0, math.radians(90)]]

        return self.starts, self.goals

    def five_cross_scene(self, radius):
        assert self.num_agents == 5

        self.starts = [[radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), -math.radians(150)],
                       [0, radius, -math.radians(90)],
                       [-radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), -math.radians(30)],
                       [-radius *  math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(30)],
                       [radius * math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(150)]]

        self.goals = [[-radius * math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(150)],
                      [0, -radius, -math.radians(-90)],
                      [radius * math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(150)],
                      [radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), math.radians(30)],
                      [-radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), -math.radians(30)]]

        return self.starts, self.goals

    def six_cross_scene(self, radius):
        assert self.num_agents == 6

        self.starts = [[radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), -math.radians(150)],
                       [0, radius, -math.radians(90)],
                       [-radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), -math.radians(30)],
                       [-radius * math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(30)],
                       [0, -radius, math.radians(90)],
                       [radius * math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(150)]]

        self.goals = [[-radius * math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(150)],
                      [0, -radius, -math.radians(-90)],
                      [radius * math.sin(math.radians(60)), -radius * math.sin(math.radians(30)), math.radians(150)],
                      [radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), math.radians(30)],
                      [0, radius, math.radians(-90)],
                      [-radius * math.sin(math.radians(60)), radius * math.sin(math.radians(30)), -math.radians(30)]]

        return self.starts, self.goals

    def ten_cross_scene(self, gap_dis=0.5, total_dis=4):
        sx = -4.
        sy = -(total_dis) / 2.
        sa = .5 * np.pi

        for i in range(5):
            self.starts.append([sx, sy, sa])
            self.starts.append([sx, -sy, sa + np.pi])
            self.goals.append([sx, -sy, sa + np.pi])
            self.goals.append([sx, sy, sa])

            sx += gap_dis
            if i % 2 == 0:
                sy -= gap_dis
            else:
                sy += gap_dis

        return self.starts, self.goals

    def multi_scenes(self):
        assert self.num_agents == 58
        assert self.num_obstacles == 65

        self.random_starts = []
        self.random_goals = []
        self.random_obstacles = []
        self.random_starts_lines = []
        self.random_goals_lines = []
        self.random_starts_circle = []
        self.random_goals_circle = []
        self.random_starts_group_crossing = []
        self.random_goals_group_crossing = []
        self.random_evacuation = []
        self.random_t_crossing = []

        for _ in range(5):
            self._generate_random_obstacles()

        for _ in range(21):
            self._generate_random_points()

        for _ in range(4):
            self._generate_random_lines()

        for _ in range(11):
            self._generate_random_circle([-12, -5], 3.5)

        for _ in range(4):
            self._generate_random_group_crossing(0)

        for _ in range(4):
            self._generate_random_group_crossing(1)

        for _ in range(6):
            self._generate_random_evacuation()

        for _ in range(4):
            self._generate_random_t_crossing()

        self.starts = [[self.random_starts_group_crossing[0][0], self.random_starts_group_crossing[0][1], self.random_starts_group_crossing[0][2]],
                       [self.random_starts_group_crossing[1][0], self.random_starts_group_crossing[1][1], self.random_starts_group_crossing[1][2]],
                       [self.random_starts_group_crossing[2][0], self.random_starts_group_crossing[2][1], self.random_starts_group_crossing[2][2]],
                       [self.random_starts_group_crossing[3][0], self.random_starts_group_crossing[3][1], self.random_starts_group_crossing[3][2]],
                       [self.random_starts_group_crossing[4][0], self.random_starts_group_crossing[4][1], self.random_starts_group_crossing[4][2]],
                       [self.random_starts_group_crossing[5][0], self.random_starts_group_crossing[5][1], self.random_starts_group_crossing[5][2]],
                       [self.random_starts_group_crossing[6][0], self.random_starts_group_crossing[6][1], self.random_starts_group_crossing[6][2]],
                       [self.random_starts_group_crossing[7][0], self.random_starts_group_crossing[7][1], self.random_starts_group_crossing[7][2]],
                       # top left, 4 crossing
                       [-9.0  + 0.2 * np.random.normal(), 6.0 + 0.2 * np.random.normal(), math.radians(-180 + 10 * np.random.normal())],
                       [-15.0 + 0.2 * np.random.normal(), 6.0 + 0.2 * np.random.normal(), math.radians(0 + 10 * np.random.normal())],
                       [-12.0 + 0.2 * np.random.normal(), 9.0 + 0.2 * np.random.normal(), math.radians(-90 + 10 * np.random.normal())],
                       [-12.0 + 0.2 * np.random.normal(), 3.0 + 0.2 * np.random.normal(), math.radians(90 + 10 * np.random.normal())],
                       # top right, evacuation
                       [self.random_evacuation[0][0], self.random_evacuation[0][1], self.random_evacuation[0][2]],
                       [self.random_evacuation[1][0], self.random_evacuation[1][1], self.random_evacuation[1][2]],
                       [self.random_evacuation[2][0], self.random_evacuation[2][1], self.random_evacuation[2][2]],
                       [self.random_evacuation[3][0], self.random_evacuation[3][1], self.random_evacuation[3][2]],
                       [self.random_evacuation[4][0], self.random_evacuation[4][1], self.random_evacuation[4][2]],
                       [self.random_evacuation[5][0], self.random_evacuation[5][1], self.random_evacuation[5][2]],
                       # top left, crossing with obstacle
                       [self.random_starts_circle[0][0], self.random_starts_circle[0][1], self.random_starts_circle[0][2]],
                       [self.random_starts_circle[1][0], self.random_starts_circle[1][1], self.random_starts_circle[1][2]],
                       [self.random_starts_circle[2][0], self.random_starts_circle[2][1], self.random_starts_circle[2][2]],
                       [self.random_starts_circle[3][0], self.random_starts_circle[3][1], self.random_starts_circle[3][2]],
                       [self.random_starts_circle[4][0], self.random_starts_circle[4][1], self.random_starts_circle[4][2]],
                       [self.random_starts_circle[5][0], self.random_starts_circle[5][1], self.random_starts_circle[5][2]],
                       # middle down, crossing through many obstacles
                       [self.random_starts_lines[0][0], self.random_starts_lines[0][1], self.random_starts_lines[0][2]],
                       [self.random_starts_lines[1][0], self.random_starts_lines[1][1], self.random_starts_lines[1][2]],
                       [self.random_starts_lines[2][0], self.random_starts_lines[2][1], self.random_starts_lines[2][2]],
                       [self.random_starts_lines[3][0], self.random_starts_lines[3][1], self.random_starts_lines[3][2]],
                       # down right, random
                       [self.random_starts[0][0], self.random_starts[0][1], self.random_starts[0][2]],
                       [self.random_starts[1][0], self.random_starts[1][1], self.random_starts[1][2]],
                       [self.random_starts[2][0], self.random_starts[2][1], self.random_starts[2][2]],
                       [self.random_starts[3][0], self.random_starts[3][1], self.random_starts[3][2]],
                       [self.random_starts[4][0], self.random_starts[4][1], self.random_starts[4][2]],
                       [self.random_starts[5][0], self.random_starts[5][1], self.random_starts[5][2]],
                       [self.random_starts[6][0], self.random_starts[6][1], self.random_starts[6][2]],
                       [self.random_starts[7][0], self.random_starts[7][1], self.random_starts[7][2]],
                       [self.random_starts[8][0], self.random_starts[8][1], self.random_starts[8][2]],
                       # down left, demo
                       [self.random_starts_circle[6][0], self.random_starts_circle[6][1], self.random_starts_circle[6][2]],
                       [self.random_starts_circle[7][0], self.random_starts_circle[7][1], self.random_starts_circle[7][2]],
                       [self.random_starts_circle[8][0], self.random_starts_circle[8][1], self.random_starts_circle[8][2]],
                       [self.random_starts_circle[9][0], self.random_starts_circle[9][1], self.random_starts_circle[9][2]],
                       [self.random_starts_circle[10][0], self.random_starts_circle[10][1], self.random_starts_circle[10][2]],
                       # middle down, T crossing
                       [self.random_t_crossing[0][0], self.random_t_crossing[0][1], self.random_t_crossing[0][2]],
                       [self.random_t_crossing[1][0], self.random_t_crossing[1][1], self.random_t_crossing[1][2]],
                       [self.random_t_crossing[2][0], self.random_t_crossing[2][1], self.random_t_crossing[2][2]],
                       [self.random_t_crossing[3][0], self.random_t_crossing[3][1], self.random_t_crossing[3][2]],
                       # add random agents
                       [self.random_starts[9][0], self.random_starts[9][1], self.random_starts[9][2]],
                       [self.random_starts[10][0], self.random_starts[10][1], self.random_starts[10][2]],
                       [self.random_starts[11][0], self.random_starts[11][1], self.random_starts[11][2]],
                       [self.random_starts[12][0], self.random_starts[12][1], self.random_starts[12][2]],
                       [self.random_starts[13][0], self.random_starts[13][1], self.random_starts[13][2]],
                       [self.random_starts[14][0], self.random_starts[14][1], self.random_starts[14][2]],
                       [self.random_starts[15][0], self.random_starts[15][1], self.random_starts[15][2]],
                       [self.random_starts[16][0], self.random_starts[16][1], self.random_starts[16][2]],
                       [self.random_starts[17][0], self.random_starts[17][1], self.random_starts[17][2]],
                       [self.random_starts[18][0], self.random_starts[18][1], self.random_starts[18][2]],
                       [self.random_starts[19][0], self.random_starts[19][1], self.random_starts[19][2]],
                       [self.random_starts[20][0], self.random_starts[20][1], self.random_starts[20][2]],
                       # middle, group crossing, obstacle
                       [-2.5, 0.95, math.radians(0)],
                       [-2.5, -0.95, math.radians(0)],
                       [2.5, 0.95, math.radians(0)],
                       [2.5, -0.95, math.radians(0)],
                       [5.0, 0.0, math.radians(90)],
                       [-5.0, 0.0, math.radians(90)],
                       # top left, 4 crossing, obstacle
                       [-14.75, 6.75, 0.0],
                       [-14.75, 5.25, 0.0],
                       [-9.25, 6.75, 0.0],
                       [-9.25, 5.25, 0.0],
                       [-12.75, 3.25, math.radians(90)],
                       [-11.25, 3.25, math.radians(90)],
                       [-11.25, 8.75, math.radians(90)],
                       [-12.75, 8.75, math.radians(90)],
                       [-12., 1.25, 0.0],
                       [-12., 10.75, 0.0],
                       [-16.75, 6.0, math.radians(90)],
                       [-7.25, 6.0, math.radians(90)],
                       # top right, evacuation, obstacle
                       [9.5, 6.0, 0.0],
                       [15.5, 6.0, 0.0],
                       [7.0, 4.0, math.radians(90)],
                       [18.0, 4.0, math.radians(90)],
                       [9.5, 2.0, 0.0],
                       [12.5, 2.0, 0.0],
                       [15.5, 2.0, 0.0],
                       [7.0, 8.0, math.radians(90)],
                       [18.0, 8.0, math.radians(90)],
                       # top left, crossing with obstacle, obstacle
                       [-.25, 6.0, math.radians(90)],
                       [-1.25, 6.0, math.radians(90)],
                       [-.75, 6.5, 0.0],
                       [-.75, 5.5, 0.0],
                       # middle down, crossing through many obstacles, obstacle
                       [-5.0, -3.5, math.radians(90)],
                       [5.0, -3.5, math.radians(90)], 
                       [-5.0, -8.5, math.radians(90)],
                       [5.0, -8.5, math.radians(90)],
                       [-2.5, -11., 0.0],
                       [2.5, -11.0, 0.0],
                       [-3.5, -8.0, 0.0],
                       [-3.5, -6.0, 0.0],
                       [-1.5, -8.0, 0.0],
                       [0.5, -7.0, 0.0],
                       [3.5, -7.0, 0.0],
                       [1.5, -5.0, 0.0],
                       [-2.5, -5.0, 0.0],
                       [0.0, -4.0, 0.0],
                       [-4.5, -4.0, 0.0],
                       [-2.5, -3.0, 0.0],
                       [3, -3.0, 0.0],
                       # middle down, T crossing, obstacle
                       [0.0, 9.5, 0.0], 
                       [0.0, 2.5, 0.0],
                       [-1.75, 8, 0.0],
                       [1.75, 8, 0.0],
                       [-1.75, 4, 0.0],
                       [1.75, 4, 0.0],
                       [-0.75, 6, math.radians(90)],
                       [0.75, 6, math.radians(90)],
                       [-2.75, 8.75, math.radians(90)],
                       [2.75, 8.75, math.radians(90)],
                       [-2.75, 3.25, math.radians(90)],
                       [2.75, 3.25, math.radians(90)],
                       # dynamic obstacle
                       [self.random_obstacles[0][0], self.random_obstacles[0][1], self.random_obstacles[0][2]],
                       [self.random_obstacles[1][0], self.random_obstacles[1][1], self.random_obstacles[1][2]],
                       [self.random_obstacles[2][0], self.random_obstacles[2][1], self.random_obstacles[2][2]],
                       [self.random_obstacles[3][0], self.random_obstacles[3][1], self.random_obstacles[3][2]],
                       [self.random_obstacles[4][0], self.random_obstacles[4][1], self.random_obstacles[4][2]]
                       ]

        self.goals = [[self.random_goals_group_crossing[0][0], self.random_goals_group_crossing[0][1], self.random_goals_group_crossing[0][2]],
                       [self.random_goals_group_crossing[1][0], self.random_goals_group_crossing[1][1], self.random_goals_group_crossing[1][2]],
                       [self.random_goals_group_crossing[2][0], self.random_goals_group_crossing[2][1], self.random_goals_group_crossing[2][2]],
                       [self.random_goals_group_crossing[3][0], self.random_goals_group_crossing[3][1], self.random_goals_group_crossing[3][2]],
                       [self.random_goals_group_crossing[4][0], self.random_goals_group_crossing[4][1], self.random_goals_group_crossing[4][2]],
                       [self.random_goals_group_crossing[5][0], self.random_goals_group_crossing[5][1], self.random_goals_group_crossing[5][2]],
                       [self.random_goals_group_crossing[6][0], self.random_goals_group_crossing[6][1], self.random_goals_group_crossing[6][2]],
                       [self.random_goals_group_crossing[7][0], self.random_goals_group_crossing[7][1], self.random_goals_group_crossing[7][2]],
                       # top left, 4 crossing
                       [-15.0, 6.0, math.radians(-180)],
                       [-9.0, 6.0, math.radians(0)],
                       [-12.0, 3.0, math.radians(-90)],
                       [-12.0, 9.0, math.radians(90)],
                       # top right, evacuation
                       [12.0, 8.0, math.radians(90)],
                       [13.0, 9.0, math.radians(90)],
                       [12.0, 10.0, math.radians(90)],
                       [13.0, 10.0, math.radians(90)],
                       [12.0, 9.0, math.radians(90)],
                       [13.0, 8.0, math.radians(90)],
                       # top left, crossing with obstacle
                       [self.random_goals_circle[0][0], self.random_goals_circle[0][1], self.random_goals_circle[0][2]],
                       [self.random_goals_circle[1][0], self.random_goals_circle[1][1], self.random_goals_circle[1][2]],
                       [self.random_goals_circle[2][0], self.random_goals_circle[2][1], self.random_goals_circle[2][2]],
                       [self.random_goals_circle[3][0], self.random_goals_circle[3][1], self.random_goals_circle[3][2]],
                       [self.random_goals_circle[4][0], self.random_goals_circle[4][1], self.random_goals_circle[4][2]],
                       [self.random_goals_circle[5][0], self.random_goals_circle[5][1], self.random_goals_circle[5][2]],
                       # middle down, crossing through many obstacles
                       [self.random_goals_lines[0][0], self.random_goals_lines[0][1], self.random_goals_lines[0][2]],
                       [self.random_goals_lines[1][0], self.random_goals_lines[1][1], self.random_goals_lines[1][2]],
                       [self.random_goals_lines[2][0], self.random_goals_lines[2][1], self.random_goals_lines[2][2]],
                       [self.random_goals_lines[3][0], self.random_goals_lines[3][1], self.random_goals_lines[3][2]],
                       # down right, random
                       [self.random_goals[0][0], self.random_goals[0][1], self.random_goals[0][2]],
                       [self.random_goals[1][0], self.random_goals[1][1], self.random_goals[1][2]],
                       [self.random_goals[2][0], self.random_goals[2][1], self.random_goals[2][2]],
                       [self.random_goals[3][0], self.random_goals[3][1], self.random_goals[3][2]],
                       [self.random_goals[4][0], self.random_goals[4][1], self.random_goals[4][2]],
                       [self.random_goals[5][0], self.random_goals[5][1], self.random_goals[5][2]],
                       [self.random_goals[6][0], self.random_goals[6][1], self.random_goals[6][2]],
                       [self.random_goals[7][0], self.random_goals[7][1], self.random_goals[7][2]],
                       [self.random_goals[8][0], self.random_goals[8][1], self.random_goals[8][2]],
                       # down left, demo
                       [self.random_goals_circle[6][0], self.random_goals_circle[6][1], self.random_goals_circle[6][2]],
                       [self.random_goals_circle[7][0], self.random_goals_circle[7][1], self.random_goals_circle[7][2]],
                       [self.random_goals_circle[8][0], self.random_goals_circle[8][1], self.random_goals_circle[8][2]],
                       [self.random_goals_circle[9][0], self.random_goals_circle[9][1], self.random_goals_circle[9][2]],
                       [self.random_goals_circle[10][0], self.random_goals_circle[10][1], self.random_goals_circle[10][2]],
                       # middle down, T crossing
                       [-2.25, 3.25, math.radians(0)],
                       [-1.5, 3.25, math.radians(0)],
                       [2.25, 3.25, math.radians(180)],
                       [1.5, 3.25, math.radians(180)],
                       # add random agents goals
                       [self.random_goals[9][0], self.random_goals[9][1], self.random_goals[9][2]],
                       [self.random_goals[10][0], self.random_goals[10][1], self.random_goals[10][2]],
                       [self.random_goals[11][0], self.random_goals[11][1], self.random_goals[11][2]],
                       [self.random_goals[12][0], self.random_goals[12][1], self.random_goals[12][2]],
                       [self.random_goals[13][0], self.random_goals[13][1], self.random_goals[13][2]],
                       [self.random_goals[14][0], self.random_goals[14][1], self.random_goals[14][2]],
                       [self.random_goals[15][0], self.random_goals[15][1], self.random_goals[15][2]],
                       [self.random_goals[16][0], self.random_goals[16][1], self.random_goals[16][2]],
                       [self.random_goals[17][0], self.random_goals[17][1], self.random_goals[17][2]],
                       [self.random_goals[18][0], self.random_goals[18][1], self.random_goals[18][2]],
                       [self.random_goals[19][0], self.random_goals[19][1], self.random_goals[19][2]],
                       [self.random_goals[20][0], self.random_goals[20][1], self.random_goals[20][2]],
                       # middle, group crossing
                       [-2.5, 0.95, math.radians(0)],
                       [-2.5, -0.95, math.radians(0)],
                       [2.5, 0.95, math.radians(0)],
                       [2.5, -0.95, math.radians(0)],
                       [5.0, 0.0, math.radians(90)],
                       [-5.0, 0.0, math.radians(90)],
                       # top left, 4 crossing
                       [-20.75, 18.75, 0.0],
                       [-20.75, 17.25, 0.0],
                       [-15.25, 18.75, 0.0],
                       [-15.25, 17.25, 0.0],
                       [-18.75, 15.25, math.radians(90)],
                       [-17.25, 15.25, math.radians(90)],
                       [-17.25, 20.75, math.radians(90)],
                       [-18.25, 20.75, math.radians(90)],
                       [-18., 13.25, 0.0],
                       [-18., 22.75, 0.0],
                       [-22.75, 18.0, math.radians(90)],
                       [-13.25, 18.0, math.radians(90)],
                       # top right, evacuation
                       [9.5, 12.0, 0.0],
                       [15.5, 12.0, 0.0],
                       [7.0, 10.0, math.radians(90)],
                       [18.0, 10.0, math.radians(90)],
                       [9.5, 8.0, 0.0],
                       [12.5, 8.0, 0.0],
                       [15.5, 8.0, 0.0],
                       [7.0, 14.0, math.radians(90)],
                       [18.0, 14.0, math.radians(90)],
                       # top left, crossing with obstacle
                       [-11.5, 9.0, math.radians(90)],
                       [-10.5, 9.0, math.radians(90)],
                       [-11.0, 9.5, 0.0],
                       [-11.0, 8.5, 0.0],
                       # middle down, crossing through many obstacles
                       [-5.0, -3.5, math.radians(90)],
                       [5.0, -3.5, math.radians(90)], 
                       [-5.0, -8.5, math.radians(90)],
                       [5.0, -8.5, math.radians(90)],
                       [-2.5, -11., 0.0],
                       [2.5, -11.0, 0.0],
                       [-3.5, -8.0, 0.0],
                       [-3.5, -6.0, 0.0],
                       [-1.5, -8.0, 0.0],
                       [0.5, -7.0, 0.0],
                       [3.5, -7.0, 0.0],
                       [1.5, -5.0, 0.0],
                       [-2.5, -5.0, 0.0],
                       [0.0, -4.0, 0.0],
                       [-4.5, -4.0, 0.0],
                       [-2.5, -3.0, 0.0],
                       [3, -3.0, 0.0],
                       # middle down, T crossing
                       [0.0, -15.5 , 0.0], 
                       [0.0, -22.5 , 0.0],
                       [-1.75, -17.0, 0.0],
                       [1.75, -17.0, 0.0],
                       [-1.75, -21.0, 0.0],
                       [1.75, -21.0, 0.0],
                       [-0.75, -19.0, math.radians(90)],
                       [0.75, -19.0, math.radians(90)],
                       [-2.75, -16.25, math.radians(90)],
                       [2.75, -16.25, math.radians(90)],
                       [-2.75, -21.75, math.radians(90)],
                       [2.75, -21.75, math.radians(90)],
                       # dynamic obstacle
                       [self.random_obstacles[0][0], self.random_obstacles[0][1], self.random_obstacles[0][2]],
                       [self.random_obstacles[1][0], self.random_obstacles[1][1], self.random_obstacles[1][2]],
                       [self.random_obstacles[2][0], self.random_obstacles[2][1], self.random_obstacles[2][2]],
                       [self.random_obstacles[3][0], self.random_obstacles[3][1], self.random_obstacles[3][2]],
                       [self.random_obstacles[4][0], self.random_obstacles[4][1], self.random_obstacles[4][2]]
                       ]

        return self.starts, self.goals

    def _generate_random_points(self):
        succ = False
        sx, sy, sa, gx, gy, ga = 0., 0., 0., 0., 0., 0.
        while not succ:
            sx, gx = self.np_random.uniform(7.0, 17.0, 2)
            sy, gy = self.np_random.uniform(.0, -10.0, 2)
            sa, ga = self.np_random.uniform(-np.pi, np.pi, 2)
            succ = True

            if np.hypot(gx - sx, gy - sy) < 5.0:
                succ = False

            if self.random_starts:
                for s in self.random_starts:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 5.:
                        succ = False

                for s in self.random_obstacles:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 5.:
                        succ = False

            if self.random_goals:
                for g in self.random_goals:
                    if np.hypot(gx - g[0], gy - g[1]) < self.agent_size * 5.:
                        succ = False

                for g in self.random_obstacles:
                    if np.hypot(gx - g[0], gy - g[1]) < self.agent_size * 5.:
                        succ = False

        self.random_starts.append([sx, sy, sa])
        self.random_goals.append([gx, gy, ga])

    def _generate_random_obstacles(self):
        succ = False
        sx, sy, sa = 0., 0., 0.
        while not succ:
            sx = self.np_random.uniform(9., 15.)
            sy = self.np_random.uniform(-2., -8.)
            sa = self.np_random.uniform(-np.pi, np.pi)
            succ = True

            if self.random_obstacles:
                for s in self.random_obstacles:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 5.:
                        succ = False

        self.random_obstacles.append([sx, sy, sa])

    def _generate_random_lines(self):
        succ = False
        sx, sy, sa, gx, gy, ga = 0., 0., 0., 0., 0., 0.
        while not succ:
            sx, gx = self.np_random.uniform(-4., 4., 2)
            sy = self.np_random.uniform(-9., -10.)
            gy = self.np_random.uniform(-2.25, -1.75)
            sa, ga = self.np_random.uniform(0, np.pi, 2)
            succ = True

            if self.random_starts_lines:
                for s in self.random_starts_lines:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 5.:
                        succ = False

            if self.random_goals_lines:
                for g in self.random_goals_lines:
                    if np.hypot(gx - g[0], gy - g[1]) < self.agent_size * 5.:
                        succ = False

        self.random_starts_lines.append([sx, sy, sa])
        self.random_goals_lines.append([gx, gy, ga])

    def _generate_random_circle(self, center, radius):
        succ = False
        sx, sy, sa, gx, gy, ga = 0., 0., 0., 0., 0., 0.
        while not succ:
            angle = self.np_random.uniform(0., 2*np.pi)
            sx = radius * np.cos(angle) + center[0]
            sy = radius * np.sin(angle) + center[1]
            gx = radius * np.cos(angle + np.pi) + center[0]
            gy = radius * np.sin(angle + np.pi) + center[1]
            sa = np.arctan2(-radius * np.sin(angle), -radius * np.cos(angle))
            ga = 0.0
            succ = True

            if self.random_starts_circle:
                for s in self.random_starts_circle:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size*4+0.15:
                            succ = False

        self.random_starts_circle.append([sx, sy, sa])
        self.random_goals_circle.append([gx, gy, ga])

    def _generate_random_group_crossing(self, region):
        succ = False
        sx, sy = 0., 0.
        while not succ:
            sa = 0. if region == 0 else np.pi
            sx = self.np_random.uniform(-4, -2) if region == 0 else self.np_random.uniform(2, 4)
            sy = self.np_random.uniform(-0.75, 0.75)
            succ = True
            if self.random_starts_group_crossing:
                for s in self.random_starts_group_crossing:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 4.:
                        succ = False

        self.random_starts_group_crossing.append([sx, sy, sa])
        self.random_goals_group_crossing.append([-sx, sy, sa])

    def _generate_random_evacuation(self):
        succ = False
        sx, sy, sa = 0., 0., 0.
        while not succ:
            sx = self.np_random.uniform(9., 16.)
            sy = self.np_random.uniform(3., 5.)
            sa = self.np_random.uniform(0., np.pi)
            succ = True

            if self.random_evacuation:
                for s in self.random_evacuation:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 6.:
                        succ = False

        self.random_evacuation.append([sx, sy, sa])

    def _generate_random_t_crossing(self):
        succ = False
        sx, sy, sa = 0., 0., 0.
        while not succ:
            sx = self.np_random.uniform(-2., 2.)
            sy = self.np_random.uniform(8.5, 9.)
            sa = self.np_random.uniform(-np.pi, 0.)
            succ = True

            if self.random_t_crossing:
                for s in self.random_t_crossing:
                    if np.hypot(sx - s[0], sy - s[1]) < self.agent_size * 4.:
                        succ = False

        self.random_t_crossing.append([sx, sy, sa])


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
