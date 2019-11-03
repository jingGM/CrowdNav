---
## files descriptions
launch: scripts to launch gazebo and also execute ros nodes for robots
	empty: there are no robots in environment, always used for testing scenarios
	turtlebots: all launch files with turtlebots
	jackals: all launch files with jackals

worlds: all world files of gazebo

models: models used by gazebo
	hokuyo: the lidar package, which is used by jackals and also turtlebot packages
	jackals: configuration files of jackal
	turtlebot2: configuration files of turtlebot

msg/srv:
	message files of ros, used for trasmission of data between python and c++ files

src: c++ files which are used to build a simulator and collect data from simulator

scripts: python files used for reinforcement learning methods:
	multi_robot_ppo.py: main file, setting iterations and global parameters
	stage_env.py: communicating with c++ and pass data to reinforcement learning algorithms
	scenarios.py: setting robots start and goal positions
	ppo folder: contains all configuration of reinforcement learning methods
		agent.py: setting networkds
		ppo.py: setting configuration of PPO algorithm
		utils.py: processing input data of networkds
		vel_smoother.py: smoother of velocities.


