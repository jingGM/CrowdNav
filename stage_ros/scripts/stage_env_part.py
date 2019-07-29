import numpy as np
import rospy

from sensor_msgs.msg import LaserScan, CameraInfo, Image, 
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Twist
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

from tf import transformations

from scenarios import Scenarios

from ppo.vel_smoother import VelocitySmoother






class StageEnv(object):
    """
    Wrapped Stage simulator using gym's API.
    """

    def __init__(self, num_agents=5, num_obstacles=0, agent_radius=0.12, env_size=4.0,max_vx=1.0):
        
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles
        self.agent_size = agent_radius
        self.env_size = env_size
        self.max_vx = max_vx
        self.vel_smoother = VelocitySmoother()
        self.scenarios = Scenarios(
            num_agents, num_obstacles, agent_radius, env_size)

        #####################################################################################################################################################################
        ## need to figure out how to use the markers
        ##############################################################################################################################################################
        self.path_markers = MarkerArray()
        self.path_markers_id = 0
        self.goal_markers = MarkerArray()
        self.goal_markers_id = 0
        self.agent_markers = MarkerArray()
        self.agent_markers_id = 0
        self.agent_markers_pub = rospy.Publisher(
            'agent_markers', MarkerArray, queue_size=10)
        self.path_markers_pub = rospy.Publisher(
            'path_markers', MarkerArray, queue_size=10)
        self.goal_markers_pub = rospy.Publisher(
            'goal_markers', MarkerArray, queue_size=10)
        # a list of agent's current poses

        
        # a list of agent's color
        self.agent_colors = self._select_colors()
        self.option_colors = self._select_option_colors(options)


        # a list of agent's current actions
        self.agent_actions = [[0., 0.] for _ in range(self.num_agents)]

        self.agent_poses = [[] for _ in range(self.num_agents)]
        self.agent_last_poses = [[] for _ in range(self.num_agents)]

        # clear path markers in the last episode
        self.trajectory = np.zeros(self.num_agents)

        self.envdata = EnvData(self.num_agents)

        self.start_time = 0.

    def reset(self):
        """
        Resets the state of the env, returning an initial observation.
        When end of an episode is reached, reset() should be called to reset
        the env's internal state.

        Outputs:
        -------
        obs: the initial obs of the space. (Initial reward is assumed to be 0.)
        """

        if len(self.path_markers.markers) != 0:
            for m in self.path_markers.markers:
                m.action = Marker.DELETE
        self.path_markers_pub.publish(self.path_markers)

        self.path_markers = MarkerArray()
        self.path_markers_id = 0
        self.goal_markers = MarkerArray()
        self.goal_markers_id = 0

        # generate new starts and goals, and send them to Stage
        succ = False
        while not succ:
            succ = self._update_stage()

        # _show_goal only need to be called at the beginning of the episode
        self._show_goal()
        # print("after show goal")
        self.start_time = rospy.get_time()

        # request initial observation
        # receive and parse data
        obs, _, _, _ = self.envdata._read_data()
        self.agent_last_poses = self.agent_poses = self.starts
        # print("after read data")

        # clear path markers in the last episode
        self.trajectory = np.zeros(self.num_agents)

        return obs

    def _update_stage(self):
        self.starts = []
        self.goals = []

        self.scenarios.reset()
        # self.starts, self.goals = self.scenarios.multi_scenes()
        # self.starts, self.goals = self.scenarios.no_obstacle_scene()
        # self.starts, self.goals = self.scenarios.mit_crossing_scene(120)
        # self.starts, self.goals = self.scenarios.moving_car_scene()
        # self.starts, self.goals = self.scenarios.face_car_scene()
        # self.starts, self.goals = self.scenarios.cross_road_scene()
        # self.starts, self.goals = self.scenarios.random_scene()
        # self.starts, self.goals = self.scenarios.random_obstacles_scene()
        # self.starts, self.goals = self.scenarios.circle_scene()
        # self.starts, self.goals = self.scenarios.circle_scene_with_obstacles()
        self.starts, self.goals = self.scenarios.circle_scene_uniform()
        # self.starts, self.goals = self.scenarios.crossing_scene(6)
        # self.starts, self.goals = self.scenarios.ten_cross_scene(0.8, 6)
        # self.starts, self.goals = self.scenarios.crossing_with_obstacle_scene(6)
        # self.starts, self.goals = self.scenarios.corridor_scene()
        # self.starts, self.goals = self.scenarios.evacuation_scene()

        self.perfect_distance = self.get_perfect_distance()

        resp = self.envdata.setrobotposition(self.starts, self.goals)

        return resp.success

    def get_perfect_distance(self):
        delta = np.array(self.starts) - np.array(self.goals)
        perfect_distance = np.hypot(
            delta[:self.num_agents, 0], delta[:self.num_agents, 1])
        return perfect_distance


    def _show_goal(self):
        for i in range(self.num_agents):
            self._add_marker(
                Marker.CYLINDER, [1., 1., 0., .8], [.3, .3, .01],
                self.goals[i],
                ns='goal')
            self._add_marker(
                Marker.TEXT_VIEW_FACING, [0., 0.5, 0.5, .7], [.0, .0, .20],
                self.goals[i],
                ns='goal',
                text=str(i))
        self.goal_markers_pub.publish(self.goal_markers)

    def _add_marker(self,markerType,color,scale,pose=None,ns=None,text=None,points=None):
        if pose is not None:
            pose = self._to_pose(pose)
        marker = Marker()
        marker.header.frame_id = 'ground_truth'
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.type = markerType
        marker.action = Marker.ADD
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        if pose:
            marker.pose = pose
        if text:
            marker.text = text
            marker.pose.position.z += 0.1
            if ns == "goal":
                marker.pose.position.y += 0.2
        if points:
            marker.points = []
            for p in points:
                pt = Point()
                pt.x = p[0]
                pt.y = p[1]
                pt.z = 0.
                marker.points.append(pt)
        if ns == 'agent':
            marker.id = self.agent_markers_id
            self.agent_markers.markers.append(marker)
            self.agent_markers_id += 1
        if ns == 'path':
            marker.id = self.path_markers_id
            self.path_markers.markers.append(marker)
            self.path_markers_id += 1
        if ns == 'goal':
            marker.id = self.goal_markers_id
            self.goal_markers.markers.append(marker)
            self.goal_markers_id += 1
        if ns == "usv":
            marker.type = marker.MESH_RESOURCE
            marker.mesh_resource = "file:///home/jing/Documents/catkin_workspace/catkin_CamNavi/src/stage_ros/rviz/usv.dae"
            marker.id = self.agent_markers_id
            self.agent_markers.markers.append(marker)
            self.agent_markers_id += 1
            
    def _to_pose(self, data):
        pose = Pose()

        # pose: data = [x, y, yaw]
        if len(data) == 3:
            pose.position.x = data[0]
            pose.position.y = data[1]
            pose.orientation.w = np.cos(data[2] / 2)
            pose.orientation.z = np.sin(data[2] / 2)
            return pose

        # data = [translation, rotation]
        elif len(data) == 2 and len(data[0]) == 3 and len(data[1]) == 4:
            pose.position.x = data[0][0]
            pose.position.y = data[0][1]
            pose.position.z = data[0][2]
            pose.orientation.x = data[1][0]
            pose.orientation.y = data[1][1]
            pose.orientation.z = data[1][2]
            pose.orientation.w = data[1][3]
            return pose
        else:
            rospy.logerr("Invalid pose data.")
            raise RuntimeError




    # call it every step
    def render(self, options=None):
        self._show_agent(options=options)
        if options is None:
            self._show_path(rospy.get_time() - self.start_time)
        else:
            self._show_options_path(options, rospy.get_time() - self.start_time)
        self.agent_last_poses = self.agent_poses


    def _show_agent(self, options=None, usv=True):
        self.agent_markers = MarkerArray()
        self.agent_markers_id = 0
        for i in range(self.num_agents):
            if options is not None:
                self._add_marker(
                    Marker.CYLINDER, [self.option_colors[options[i]][0], self.option_colors[options[i]][1], self.option_colors[options[i]][2], 1.],
                    [2 * self.agent_size, 2 * self.agent_size, 0.01],
                    pose=self.agent_poses[i],
                    ns='agent')
            elif usv:
                self._add_marker(
                    Marker.CYLINDER, [.0, 0.12, 1., 1.],
                    [2 * self.agent_size, 2 * self.agent_size, 0.01],
                    pose=self.agent_poses[i],
                    ns='usv')
            else:
                self._add_marker(
                    Marker.CYLINDER, [1., 0.12, 0., 1.],
                    [2 * self.agent_size, 2 * self.agent_size, 0.01],
                    pose=self.agent_poses[i],
                    ns="agent")
            self._add_marker(
                Marker.TEXT_VIEW_FACING, [1., 1., 1., 1.],
                [0., 0., 1.8 * self.agent_size],
                pose=self.agent_poses[i],
                ns='agent',
                text=str(i))
            self._add_marker(
                Marker.ARROW, [0.25, 0.74, 0.15, 0.7],
                [self.agent_actions[i][0], 0.02, 0.02],
                pose=self.agent_poses[i],
                ns='agent')
            print "poses:", self.agent_poses[i]
            vz_q = transformations.quaternion_from_euler(
                0., 0., self.agent_poses[i][2] +
                np.sign(self.agent_actions[i][1]) * np.pi * 0.5)
            self._add_marker(
                Marker.ARROW,
                [0.25, 0.74, 0.15, 0.7],
                [abs(self.agent_actions[i][1]), 0.02, 0.02],
                # pose = translation, quaternion
                pose=[[self.agent_poses[i][0], self.agent_poses[i][1], 0.0],
                      vz_q],
                ns='agent')
        self.agent_markers_pub.publish(self.agent_markers)

    def _show_path(self, t):
        if len(self.agent_last_poses) == 0:
            self.agent_last_poses = self.agent_poses
        for i in range(self.num_agents):
            self._add_marker(
                Marker.LINE_STRIP, [
                    self.agent_colors[i][0], self.agent_colors[i][1],
                    self.agent_colors[i][2], 0.1 + t / 15.
                ], [0.1, 0., 0.],
                points=[self.agent_last_poses[i][:2], self.agent_poses[i][:2]],
                ns='path')
        self.path_markers_pub.publish(self.path_markers)

    def _show_options_path(self, options, t):
        if len(self.agent_last_poses) == 0:
            self.agent_last_poses = self.agent_poses
        for i, option in enumerate(options):
            if i == 20:
                self._add_marker(
                Marker.LINE_STRIP, [
                self.option_colors[option][0], self.option_colors[option][1],
                self.option_colors[option][2], 0.1 + t / 15.
                ], [0.1, 0., 0.],
                points=[self.agent_last_poses[i][:2], self.agent_poses[i][:2]],
                ns='path')
        self.path_markers_pub.publish(self.path_markers)




    def step(self, actions):
        """
        Run one timestep of the env's dynamics.

        Input
        -----
        actions: actions provided by the environment

        Outputs
        -------
        obss, rewards, dones, info
        """
        # new_actions = Actions()
        # for v, t in zip(vels, dones):
        #    a = Action()
        #    if t:
        #        a.vx, a.vz = 0., 0.
        #    else:
        #        a.vx, a.vz = v[0], v[1]
        #    rospy.logwarn('action: {0}'.format(a))
        #    new_actions.data.append(a)
        actions = np.array(actions)
        for action in actions:
            action[0] = np.clip(action[0], 0.0, self.max_vx)
            action[1] = np.clip(action[1], -1., 1.)
            # action = self.vel_smoother.step(action[0], action[1], 0.1)

        self.agent_actions = actions
        assert actions.shape[0] == self.num_agents

        actions_probuf = Actions()
        for a in actions:
            apb = Action()
            apb.vx = a[0]
            apb.vz = a[1]
            actions_probuf.data.append(apb)
            # rospy.logwarn('action: {0}'.format(apb))

        self.envdata.setvelocities(actions_probuf)

        self.agent_last_poses = copy.deepcopy(self.agent_poses)
        # tell Stage we've calculated actions, please execute actions
        # sending a request (for new_obs, reward, done) to Stage
        # succ = srv.call()
        # if succ: _read_data()

        # wait until receiving new_obs, reward, done from Stage
        # if we get a response from Stage, then parse and return these info
        states, rewards, terminals, _ = self.envdata._read_data()
        self.get_trajectory()
        assert len(rewards) == self.num_agents

        return states, rewards, terminals, {}

    def get_trajectory(self):
        delta = np.array(self.agent_poses) - np.array(self.agent_last_poses)
        distance = np.hypot(
            delta[:self.num_agents, 0], delta[:self.num_agents, 1])

        self.trajectory += distance




    


if __name__ == "__main__":
    #stage = StageEnv()

    # obs = stage.reset()
    # actions = [[0.5, 0.] for _ in range(stage.num_agents)]
    # actions = np.array(actions)
    # while not rospy.is_shutdown():
    #    states, rewards, terminals, _ = stage.step(actions)
    #    print(rewards)
    #    print(terminals)

    #stage.clear_memory()
    data = EnvData()
    print(data.)

