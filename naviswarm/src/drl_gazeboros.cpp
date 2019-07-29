/* Copyright 2018 Adarsh Jagan Sathyamoorthy.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/**
    In this node,
    robot make decisions with odoms and scans,
    collect information from Gazebo simulator, and
    send/receive state and action through shared memory.

    TODO: * Reward for each robot 'i' will be published in /tbi/reward. The subscriber of that reward should be changed accordingly
          * Add images as a new state.
          * (Later) Add positions and velocity of pedestrians to States after prediction is ready
 **/

 #include <math.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string>

 #include <sys/types.h>
 #include <sys/stat.h>
 #include <unistd.h>
 #include <signal.h>
 #include <iostream>
 #include <fstream>
 #include <sstream>

 #include <ros/ros.h>
 #include <gazebo_msgs/ModelStates.h>
 #include <boost/thread/mutex.hpp>
 #include <boost/thread/thread.hpp>
 #include <sensor_msgs/LaserScan.h>
 #include <sensor_msgs/Image.h>
 #include <sensor_msgs/image_encodings.h>
 #include <sensor_msgs/CameraInfo.h>
 #include <nav_msgs/Odometry.h>
 #include <geometry_msgs/Twist.h>
 #include <geometry_msgs/Pose.h>
 #include <rosgraph_msgs/Clock.h>
 #include <tf/transform_broadcaster.h>
 #include <tf/transform_datatypes.h>
 #include <boost/tokenizer.hpp>

 // For subscribing to images from Gazebo
 #include <cv_bridge/cv_bridge.h>
 #include <image_transport/image_transport.h>

 // For bumper subscriber
 #include <kobuki_msgs/BumperEvent.h>

 // OpenCV headers
 #include <opencv2/core/core.hpp>
 #include <opencv2/highgui/highgui.hpp>
 #include <opencv2/imgcodecs/imgcodecs.hpp>
 #include <opencv2/imgproc/imgproc.hpp>
 #include <opencv2/opencv.hpp>

 // Msg definition headers (see msg folder)
 #include <naviswarm/Velocity.h>
 #include <naviswarm/Velocities.h>
 #include <naviswarm/Action.h>
 #include <naviswarm/Actions.h>
 #include <naviswarm/Goal.h>
 #include <naviswarm/Scan.h>
 #include <naviswarm/State.h>
 #include <naviswarm/States.h>
 #include <naviswarm/Transition.h>
 #include <naviswarm/Transitions.h>
 #include <naviswarm/RobotStatus.h>
// #include <naviswarm/UpdateStage.h>
 #include <naviswarm/Reward.h>

 // Service header
 #include <naviswarm/UpdateModel.h>

 // for the IPC (Inter-Process Communication) part
 #include <sys/ipc.h>		/* for system's IPC_xxx definitions */
 #include <sys/shm.h>		/* for shmget, shmat, shmdt, shmctl */
 #include <sys/sem.h>		/* for semget, semctl, semop */
 #include <errno.h>
 #include <semaphore.h>
 #include <unistd.h>

 // Preprocessor directives. Equivalent to substituting the words on the right with words on left.
 // I may not have used any of these in this code.
 #define USAGE "stageros <worldfile>"
 #define IMAGE "image"
 #define DEPTH "depth"
 #define CAMERA_INFO "camera_info"
 #define ODOM "odom"
 #define BASE_SCAN "scan"
 #define BASE_POSE_GROUND_TRUTH "base_pose_ground_truth"
 #define CMD_VEL "cmd_vel"
 #define REWARD "reward"
 #define KEY 42
 #define SIZE 2048000000
 #define PERMISSION 0600
 #define TRANSITION_KEY 43

// Robot parameters
 #define ROBOT_RADIUS 0.12
 #define MIN_DIST_BETWEEN_AGENTS (ROBOT_RADIUS+0.05)*2 // Distance between the centers
 #define OBSTACLE_NUM 0
 //#define MIN_DIST_BETWEEN_AGENTS ROBOT_RADIUS*2

//Class definition
class GazeboTrain {
private:
ros::NodeHandle nh;
boost::mutex msg_lock;

// for shared memory (sem -> semaphore)
int share_id, sem_id;
uint8_t *share_addr;

int num_robots = 1; // The actual value is assigned in the Constructor. By default it is 1.
std::vector<bool> collision_status;
ros::Subscriber groundtruth_sub; // Subscriber for Groundtruth data from Gazebo

// This structure can be used for defining multiple subscribers for multiple robots
struct Robot{
  int robot_id; // Used to number the robots to identify them
  ros::Subscriber scan_sub;
  ros::Subscriber odom_sub;
  ros::Subscriber img_sub;
  ros::Subscriber bumper_sub;
  ros::Publisher reward_pub;
};

// Data members to store the sensor data
geometry_msgs::Pose gpose;
nav_msgs::Odometry odom_data;
cv::Mat img_data; // stores image frames published in /camera/rgb/image_raw/compressed converted to Mat format.
sensor_msgs::LaserScan scan_data;

// Defining vec3 = {x, y, theta}
typedef struct {
    double x, y, theta;
} vec3;

// Defining vec3 = {x, y}
typedef struct {
    double x, y;
} vec2;

// Function to return distance = sqrt(x^2 + y^2) in 2-D
double GetDistance(double x, double y) {
    return sqrt(x * x + y * y);
}

// Function returns a transformed point (pt2 = Trans_matrix * pt1)
vec2 getTransformedPoint(const vec2 &vec, const tf::Transform &gt) {
    tf::Vector3 tf_vec, new_tf_vec;
    // Storing contents of vec into a variable of type tf::Vector3 (converting (x,y) -> (x, y, 0))
    tf_vec.setX(vec.x);
    tf_vec.setY(vec.y);
    tf_vec.setZ(0.0);
    // std::cout << "x: " << vec.x << " ---  y: " << vec.y << std::endl;
    new_tf_vec = gt * tf_vec; // gt is some representation of a transformation.
    vec2 new_vec;
    // Converting (x, y, 0) -> (x, y)
    new_vec.x = new_tf_vec.getX();
    new_vec.y = new_tf_vec.getY();
    return new_vec;
}

// Returns a transformation matrix
tf::Transform pose2transform(const geometry_msgs::Pose& pose){
    tf::Quaternion q_pose(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
    tf::Matrix3x3 m(q_pose);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    q_pose.setRPY(0.0, 0.0, yaw); // Stores the yaw
    tf::Transform t(q_pose, tf::Point(pose.position.x, pose.position.y, 0.0)); // 1st arg is orientation, 2nd arg is position/translation
    return t;
}

// Used to remember initial poses for soft reset
// Soft reset is when robots are spawned again in the original initial poses.
std::vector<geometry_msgs::Pose> initial_poses;
ros::ServiceServer reset_srv_;

// Service to reset simulation with a set new initial poses and goals for each robot.
ros::ServiceServer update_srv;

// for semaphore operation
void acquire_semaphore();
void release_semaphore();

// Training parameters
int num_episode;
int current_robot;
std::vector<vec2> current_goal;
naviswarm::States last_states;
naviswarm::Actions executed_actions;
std::vector<double> path_length;
std::vector<double> time_elapsed;

// Other ROS tf broadcaster. (I dont think this is needed for this code)
tf::TransformBroadcaster tf;
ros::Time sim_time;

public:
// Function declarations
GazeboTrain(int num_robots);

// Scan, odom and ground truth Callback functions
void scan_Callback(const sensor_msgs::LaserScan::ConstPtr& scan, int i);
void odom_Callback(const nav_msgs::Odometry::ConstPtr& odom, int i);
void gt_Callback(const gazebo_msgs::ModelStates gt);
void image_Callback(const sensor_msgs::ImageConstPtr& img_msg, int i);
void bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i);
bool cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response);
int loop();
};

// Function definitions
// Constructor for the class
GazeboTrain::GazeboTrain(int n){
  update_srv = nh.advertiseService("update_goals", &GazeboTrain::cb_update_srv, this);
  collision_status.resize(n, false);
  num_robots = n;
  loop(); // infinite loop to continuously subscribe to scan, odom, and groundtruth.
}

void GazeboTrain::acquire_semaphore()
{
    struct sembuf op[1];
    op[0].sem_num = 0;
    op[0].sem_flg = 0;
    op[0].sem_op = -1; // p operation (wait operation/ decrementing the semaphore)

    if (-1 == semop(sem_id, op, 1))// block until the operation finished
    {
        ROS_WARN("Watch out! Acquire semaphore failed.");
    }
}

void GazeboTrain::release_semaphore()
{
    struct sembuf op[1];
    op[0].sem_num = 0;
    op[0].sem_flg = 0;
    op[0].sem_op = 1; // v operation (signal operation/ Incrementing the semaphore)

    if (-1 == semop(sem_id, op, 1)) //
    {
        ROS_WARN("Watch out! Release semaphore failed.");
    }
}

// Scan Callback function
void GazeboTrain::scan_Callback(const sensor_msgs::LaserScan::ConstPtr& scan, int i)
{   // ROS_INFO("Inside scan callback");

    // Store contents of scan in the datamember scan_data
    scan_data.ranges = scan->ranges;

    // Checking for collision
    float min_range = 0.5;
    collision_status[i] = false; // NOTE: collision status for robot 0 is stored in collision_status[0].
    for (int j = 0; j < scan->ranges.size(); j++) {
        if (scan->ranges[j] < min_range) {
            collision_status[i] = true;  // true indicates presence of obstacle
        }
    }
}

//Odom Callback function
void GazeboTrain::odom_Callback(const nav_msgs::Odometry::ConstPtr& odom, int i)
{
  // ROS_INFO("Inside odom CallBack");
  // ROS_INFO("Frame: [%s]", odom->header.frame_id.c_str());
  odom_data.twist.twist.linear.x = odom->twist.twist.linear.x;
  odom_data.twist.twist.angular.z = odom->twist.twist.angular.z;
}

// Ground Truth callback
void GazeboTrain::gt_Callback(const gazebo_msgs::ModelStates gt) {
  // ROS_INFO("Inside GT CallBack and current_robot is %d", current_robot);
  // std::cout<<gt.name[0] << " "<< gt.name[1]<< " "<< gt.name[2]<<std::endl;
  for (int i = 0; i < gt.name.size(); i++){
    if(gt.name[i].substr(0,2) == "tb" && gt.name[i].compare(2, 1, std::to_string(current_robot)) == 0) {
      // std::cout <<"Robot "<<gt.name[i][2]<< " found!"<< std::endl;
      gpose = gt.pose[i];
      // std::cout <<"Robot "<<gt.name[i][2]<< " pose is "<< gpose.position.x << gpose.position.y << std::endl;
    }
  }
}

// Image CallBack
void GazeboTrain::image_Callback(const sensor_msgs::ImageConstPtr& img_msg, int i) {
  cv_bridge::CvImagePtr cvPtr;
  try {
    cvPtr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    // ROS_INFO("Inside image callback");
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  img_data = cvPtr->image;
}

// Bumper CallBack
void GazeboTrain::bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i) {
  // ROS_INFO("bumper hit. value = [%d] for robot %d", bumper_msg->bumper, i);
  if (bumper_msg->bumper == 1)
    collision_status[i] = true;
}

// Update Goal service CallBack
bool
GazeboTrain::cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response)
{
    ROS_INFO("Updatting Gazebo!");

    std::cout << "Request goal size: " << request.points.size() << std::endl;
    // std::cout << "position models size: " << this->positionmodels.size() << std::endl;
    //
    if (request.points.size() != num_robots) {
        ROS_WARN("Robot Number and number of goals don't match!");
        response.success = false;
    }
    else { // no. of poses and no. of robots match (no problem)
        // Setting the new goals
        for (size_t r = 0; r < request.points.size(); r++) {
            vec2 goal;
            goal.x = request.points[r].x;
            goal.y = request.points[r].y;
            current_goal[r] = goal;
            ROS_INFO("Goal_%d: %.3f, %.3f", int(r), goal.x, goal.y);
        }

        // WARNING: IT MAY NOT FREE THE MEMORY SPACE
        // NOTE: last_states is a private datamember of the stage_node class.
        last_states.actionObsBatch.clear();
        last_states.goalObsBatch.clear();
        last_states.scanObsBatch.clear();
        last_states.velObsBatch.clear();
        executed_actions.data.clear();
        // WARNING: IT MAY NOT FREE THE MEMORY SPACE

        response.success = true;
    }

    //ROS_INFO("update stage response: %d", response.success);
    // Whatever be the response.success value, this function returns True.
    return true;
}

// Infinite loop function
int GazeboTrain::loop(){
  while(ros::ok()){
    // This loop is equivalent to the for loop inside
    //StageNode::WorldCallback in drl_stageros.cpp
    naviswarm::States current_states;
    naviswarm::Transitions current_transitions;

    for (int j = 0; j < num_robots; ++j)
    {
        vec2 temp_goal;
        temp_goal.x = 0.;
        temp_goal.y = 0.;
        current_goal.push_back(temp_goal);
    }
    path_length.resize(num_robots, 0.0); // Resize to size = no. of robots, init the new spaces to 0.0.
    time_elapsed.resize(num_robots, 0.0);

    share_id = shmget(KEY, SIZE, IPC_CREAT | IPC_EXCL | PERMISSION);
    ROS_INFO("share_id: %d", share_id);
    if (share_id == -1){
        share_id = 0;
        ROS_FATAL("Shared memory allocate failed, Please run: python stage_env.py");
        return 1;
    }

    share_addr = (uint8_t *)shmat(share_id, NULL, 0);
    if (share_addr == (void *)-1){
        share_addr = NULL;
        ROS_FATAL("Address allocate failed");
        return 1;
    }

    // create semaphore
    sem_id = semget(KEY, 1, IPC_CREAT | IPC_EXCL | PERMISSION);
    if (sem_id == -1){
        sem_id = 0;
        ROS_FATAL("Semaphore allocate failed");
        return 1;
    }

    release_semaphore();

    for(int i = 0; i < num_robots; i++){
      current_robot = i;
      Robot* new_robot = new Robot; // Can be put inside the for loop also
      new_robot->robot_id = i;
      // Add namespace in front of the topic name (ex: /tb1/scan)
      std::string name_space = "/tb" + std::to_string(i);
      // std::string scan_topic_name = "/" + name_space + "/scan";
      // std::string odom_topic_name = "/" + name_space + "/odom";
      // std::string img_topic_name = "/" + name_space + "/camera/rgb/image_raw";

      groundtruth_sub = nh.subscribe("/gazebo/model_states", 100, &GazeboTrain::gt_Callback, this);
      new_robot->scan_sub = nh.subscribe<sensor_msgs::LaserScan>(name_space + "/scan", 10, boost::bind(&GazeboTrain::scan_Callback, this, _1, i));
      new_robot->odom_sub = nh.subscribe<nav_msgs::Odometry>(name_space + "/odom", 10, boost::bind(&GazeboTrain::odom_Callback, this, _1, i));
      new_robot->img_sub = nh.subscribe<sensor_msgs::Image>(name_space + "/camera/rgb/image_raw", 1, boost::bind(&GazeboTrain::image_Callback, this, _1, i));
      new_robot->bumper_sub = nh.subscribe<kobuki_msgs::BumperEvent>(name_space + "/mobile_base/events/bumper", 10, boost::bind(&GazeboTrain::bumper_Callback, this, _1, i));
      new_robot->reward_pub = nh.advertise<naviswarm::Reward>(name_space + "/reward", 100);
      ros::Rate loop_rate(10);
      ros::spinOnce(); // Call the gt, scan and odom callback functions once

      // std::cout<<"Collision status of robot "<< i <<" is "<< collision_status[new_robot->robot_id]<< std::endl;

      // NOTE: Block to write the subscribed data into Shared memory
      naviswarm::Transition current_transition;
      // Initially the robot is at ground truth (x, y, theta)
      current_transition.pose.push_back(gpose.position.x);
      current_transition.pose.push_back(gpose.position.y);
      tf::Quaternion q(gpose.orientation.x, gpose.orientation.y, gpose.orientation.z, gpose.orientation.w);
      tf::Matrix3x3 m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      current_transition.pose.push_back(yaw);

      // state is a combination of 4 Obs.
      naviswarm::State state;
      tf::Transform gt = pose2transform(gpose);
      // std::cout<< gpose.position.x << " "<< gpose.position.y<< std::endl;
      vec2 local_goal = getTransformedPoint(current_goal[i], gt.inverse());
      state.goalObs.goal_now.goal_dist = GetDistance(local_goal.x, local_goal.y);
      state.goalObs.goal_now.goal_theta = atan2(local_goal.y, local_goal.x);
      // std::cout<< state.goalObs.goal_now.goal_theta << std::endl;

      // v is subscribed from odom in our case.
      state.velObs.vel_now.vx = odom_data.twist.twist.linear.x;
      state.velObs.vel_now.vz = odom_data.twist.twist.angular.z;

      // initially last_states is empty. (cleared in the ros services.)
      if (last_states.goalObsBatch.size() == 0) {
          state.goalObs.goal_pprev = state.goalObs.goal_now; //pprev is previous to previous goal. (We take 3 instances of observations.)
          state.goalObs.goal_prev = state.goalObs.goal_now;
      }
      else {
          state.goalObs.goal_pprev = last_states.goalObsBatch[i].goal_prev;
          state.goalObs.goal_prev = last_states.goalObsBatch[i].goal_now;
      }
      // TODO: Need to confirm if this is same as what was given in drl_stageros.
      state.scanObs.scan_now.ranges = scan_data.ranges;
      if (last_states.goalObsBatch.size() == 0) {
          //robotmodel->lasermodels[0]->GetSensors()[0].ranges;
          state.scanObs.scan_pprev = state.scanObs.scan_now;
          state.scanObs.scan_prev = state.scanObs.scan_now;
      }
      else {
          state.scanObs.scan_pprev = last_states.scanObsBatch[i].scan_prev;
          state.scanObs.scan_prev = last_states.scanObsBatch[i].scan_now;
      }

      //
      if (last_states.goalObsBatch.size() == 0) {
          state.actionObs.ac_pprev.vx = 0.0;
          state.actionObs.ac_pprev.vz = 0.;
          state.actionObs.ac_prev.vx = 0.;
          state.actionObs.ac_prev.vz = 0.;
      }
      else { // ??? Don't understand the else part ???
          // should set last_states.data[r].acobs.ac_prev = actions.data[i] (this is ac_now)
          state.actionObs.ac_pprev = last_states.actionObsBatch[i].ac_pprev;
          state.actionObs.ac_prev = last_states.actionObsBatch[i].ac_prev;
      }

      // initial obs/state/transition
      if (last_states.goalObsBatch.size() == 0) {
          current_transition.state = state;
          //current_transition.next_state = msg;
          //stage_ros::Action empty_action;
          //current_transition.action = empty_action;
          current_transition.reward = 0.;
          current_transition.terminal = false;
      }
      else
      {
          naviswarm::Reward reward;
          reward.sum = 0.;
          reward.collision = 0.;
          reward.reached_goal = 0.;
          reward.penalty_for_deviation = 0.;
          reward.reward_approaching_goal = 0.;

          current_transition.state = state;
          //current_transition.next_state = msg;
          //current_transition.action = executed_actions.data[r];

          if (state.goalObs.goal_now.goal_dist < 0.2) {  // arrived the goal
              current_transition.terminal = true;
              current_transition.reward = 20.0;
              reward.reached_goal = 20.0;
          }
          else // if goal has not been reached
          {

              // rs.stalled[r] = collision;
              if(collision_status[i] == true) { // stalled is obtained from an in-built function from stage. we must write a function to detect collisions
                  current_transition.terminal = true;
                  current_transition.reward = -20.0;
                  reward.collision = -20.0;
              }
              else { // Goal not reached and no collisions
                  current_transition.terminal = false;

                  double reward_approaching_goal = 2.5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);

                  double penalty_for_bigvz = 0.0;
                  if (std::abs(executed_actions.data[i].vz) > 0.7) // executed actions = actions (set later)
                  {
                      penalty_for_bigvz = -0.05*std::abs(executed_actions.data[i].vz);
                  }

                  double penalty_for_deviation = 0.0;
                  // if (state.goalObs.goal_now.goal_dist < 1.)
                  // {
                  //     penalty_for_deviation = -0.1 * std::abs(state.goalObs.goal_now.goal_theta);
                  //     penalty_for_deviation -= 0.1 * v.x;
                  // }
                  // else
                  // {
                  if (std::abs(state.goalObs.goal_now.goal_theta) > 0.785)
                  {
                      penalty_for_deviation = -0.1 * (std::abs(state.goalObs.goal_now.goal_theta) - 0.785);
                  }
                  // }

                  // current_transition.reward = 0;
                  current_transition.reward = reward_approaching_goal;
                  // current_transition.reward = reward_approaching_goal + penalty_for_deviation;
                  reward.reward_approaching_goal = reward_approaching_goal;
                  reward.penalty_for_deviation = penalty_for_deviation;
              }
          }

         reward.sum = reward.collision + reward.reached_goal + reward.reward_approaching_goal + reward.penalty_for_deviation;
         new_robot->reward_pub.publish(reward); // We need a publisher for reward
         loop_rate.sleep();
      }

      current_transitions.data.push_back(current_transition);
      current_states.scanObsBatch.push_back(state.scanObs);
      current_states.goalObsBatch.push_back(state.goalObs);
      current_states.actionObsBatch.push_back(state.actionObs);
      current_states.velObsBatch.push_back(state.velObs);
      std::cout<<"Im at end of for loop"<<std::endl;
      } // end of for loop

      last_states = current_states;
      // transition_collection.frame.push_back(current_transitions);
      // send the transition, copy the information into the shared memory
      acquire_semaphore();
      uint32_t length = ros::serialization::serializationLength(current_transitions);
      // ROS_INFO("current state length is %d", length);
      boost::shared_array<uint8_t> buffer(new uint8_t[length]);
      ros::serialization::OStream stream(buffer.get(), length);
      ros::serialization::serialize(stream, current_transitions);
      memcpy(share_addr, &length, 4);
      memcpy(share_addr+4, buffer.get(), length);
      release_semaphore();
      // ROS_INFO("lock released");

      // wait command processed
      bool succ = false;
      while (!succ && ros::ok())
      {
          // wait for client to modify this value
          ros::Duration(0.005);
          acquire_semaphore();
          int new_length = *(int *) share_addr;
          std::cout<< "Length ="<< length <<" New length = "<< new_length << std::endl;
          if (new_length != length) {
            succ = true;
            std::cout<<"Im here."<<std::endl;
          } // Write has succeeded

          if (succ)
          {
              naviswarm::Actions actions;
              ros::serialization::IStream stream((share_addr + 4), new_length);
              ros::serialization::Serializer<naviswarm::Actions>::read(stream, actions); // Reads actions from shared memory
              release_semaphore();

              if (actions.data.size() != num_robots){
                  ROS_INFO("actions_size != robots_size, actions_size is %d", static_cast<int>(actions.data.size()));
                  ROS_BREAK();
              }
              //for(size_t r = 0; r < this->robotmodels_.size(); ++r)
              for (int j = 0 ; j < actions.data.size(); ++j){
                  // this->positionmodels[j]->SetSpeed(actions.data[j].vx, 0., actions.data[j].vz);
                  // Add cmd_vel publisher
                  last_states.actionObsBatch[j].ac_pprev = last_states.actionObsBatch[j].ac_prev;
                  last_states.actionObsBatch[j].ac_prev = actions.data[j];
              }

              executed_actions = actions;
              break;
          }
          else
          {
              release_semaphore();
          }
      }


  } // End of while ros::ok()
} // End of function

int main(int argc, char **argv){
  ros::init(argc, argv, "drl_gazeboros");
  GazeboTrain train(2);
  return 0;
}
