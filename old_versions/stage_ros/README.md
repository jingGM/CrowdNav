The stage_ros package is tested under Ubuntu 14.04 with ROS Indigo and Ubuntu 16.04 with ROS Kinetic.

### install requirements
* install Stage from source:
```
git clone https://github.com/rtv/Stage.git
// see INSTALL.txt in the package
cd Stage && mkdir build 
cmake ..
make 
sudo make install
```
Now, you will have libstage.so in /usr/local/lib64, that what we need to compile the important drl\_stageros.cpp.

* install tensorflow, see: https://www.tensorflow.org/install/install_linux#InstallingNativePip

* install some packages 
```
sudo pip install gym, scipy, numpy, scikit-learn, scikit-image, tabulate, sysv_ipc, rospkg
```

* install keras version 1.2.2, if you need (but i think you do not need) to run multi\_robot\_sl.py, this is for comparison the supervised trained policy with the policy trained using reinforcement learning.
```
sudo pip install keras==1.2.2
```

* install tensrolayer from source (please do not install it using pip)
```
https://github.com/zsdonghao/tensorlayer.git
cd tensorlayer 
sudo python setup.py
```

* install any other packages if need

### compile
* before compiling it using catkin\_make, we should make sure the catkin\_make system can find Stage's header files correctly:
  * if you are using Ubuntu 14.04 & ROS indigo, remove the Stage simulator installed from apt-get (sudo apt-get remove ros-indigo-stage*), and comment 56th line in CMakeLists.txt (uncomment 
the line 57)
  * if you are using Ubuntu 16.04 & ROS Kinetic, install the Stage simulator (again) from apt-get using: sudo apt-get install ros-kinetic-stage, and comment 57th line in CMakeLists.txt (uncomment the line 56)
 
* go to you ros workspace, and try to compile it using catkin\_make 

Hope everything is ok with your compilation.


### run
* roscore && rviz, load the configure file for rviz from 'stage_ros/rviz/drl_stage.rviz' 
* first, you should launch the Stage simulator by: 
```
rosrun stage_ros drl_stageros '/home/jing/Documents/catkin_workspace/catkin_swarm/src/swarmmove/stage_ros/world/env_5.world'
```
**/home/xx/catkin_ws/src/stage_ros/world/env_10.world** is the path of your world file, which describes the training environment consisting of a map, robots, obstacles and so on. If you want the robots move in a new environment, you should make a new world file to describe the environment. Note that the digital after **'env_'** (here is 10) is the number of robots (how many robots in the environment).

* second, make sure you set the correct **robot\_num** (shoulde be same with the number after 'env\_') and **obstacle\_num** in **drl_stageros.cpp** (line 97) and **multi\_robot\_ppo.py** (line 25 and line 27). If you do not have any obstacles in **xx.world**, please set **obstacle_num** to 0 in both files.

* please read the parser part in the **multi\_robot\_xxx.py** file before you run it.

* ok, now please try to 
```
python multi_robot_ppo.py
```

Congratulations!

### one more thing ...
if you kill drl\_stageros (rosrun stage\_ros drl\_stageros 'xxx.world'), you need to run **stage\_env.py** in stage\_ros/scripts/ by:
```
python stage_env.py
```
to clear the shared memory, before you re-run drl\_stageros.

### questions?
please feel free to contact Tingxiang Fan via tingxiangfan@gmail.com
 
