 #include <math.h>
 #include <stdio.h>    /* printf */
#include <cmath>       /* isnan, sqrt */
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
 #include <std_msgs/Header.h>
 #include <std_msgs/Float32.h>
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
#include <naviswarm/ActionObs.h>
 #include <naviswarm/Goal.h>
 #include <naviswarm/Scan.h>
 #include <naviswarm/State.h>
 #include <naviswarm/States.h>
 #include <naviswarm/Transition.h>
 #include <naviswarm/Transitions.h>
 #include <naviswarm/RobotStatus.h>
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

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
using namespace message_filters;

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

 #define LidarMaxDistance 5

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
    ros::Subscriber bumper_sub;
    ros::Subscriber scan_sub;
    ros::Subscriber image_sub;
    ros::Subscriber velocity_sub;
    ros::Publisher reward_pub;

    geometry_msgs::Pose gpose;
    naviswarm::Velocity odom_data;

    naviswarm::Scan         scan_data;
    naviswarm::CameraImage  img_data;

    std_msgs::Header img_header;
    std_msgs::Header scan_header;
    std_msgs::Header odom_header;

    int substatus[4] = {0,0,0,0};//check if get messages

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
    ros::ServiceServer update_srv=nh.advertiseService("update_goals", &GazeboTrain::cb_update_srv, this);

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

    void setvelocities( int robotindex, naviswarm::Action velocity){
        std::string topicname = "/turtlebot"+std::to_string(robotindex)+"/cmd_vel_mux/input/navi";
        ros::Publisher pubrobotvelocity = nh.advertise<geometry_msgs::Twist>(topicname, 1);
        geometry_msgs::Twist action;
        action.linear.x = velocity.vx;
        action.angular.z = velocity.vz;
        ros::Duration(0.01);
        int counter = 0;
        while ( (counter == 0) && ros::ok()){
          if (pubrobotvelocity.getNumSubscribers()>0){
            pubrobotvelocity.publish(action);
            counter = 1;
          }
        }
      }

  public:
    // Function declarations
    GazeboTrain(int num_robots);

    void gt_Callback(const gazebo_msgs::ModelStates gt);
    void sync_Callback( const sensor_msgs::ImageConstPtr& image,
                        const sensor_msgs::LaserScanConstPtr& scan);
    void image_Callback(const sensor_msgs::ImageConstPtr& image);
    void scan_Callback(const sensor_msgs::LaserScanConstPtr& scan);
    void bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i);
    bool cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response);
    void velocity_Callback(const nav_msgs::OdometryConstPtr& odom);
    int create_sharedmemory();
   
    int train();
};

// Constructor for the class
GazeboTrain::GazeboTrain(int n){
  collision_status.resize(n, false);
  num_robots = n;
  /*
    ROS_INFO("++++++++++++++");*/
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

void GazeboTrain::sync_Callback(const sensor_msgs::ImageConstPtr& image,
                const sensor_msgs::LaserScanConstPtr& scan)
{
  int robotindex = current_robot;
  substatus[0] = 1;

  img_data.data    = *image;
  scan_data.ranges = scan->ranges;

  img_header   = image->header;
  scan_header  = scan->header;

  //std::cout<<"----"<<std::endl;
  std::cout<<scan_header.stamp<<std::endl;
  //std::cout<<img_header<<std::endl;
  //std::cout<<odom_header<<std::endl;


  float min_range = 0.5;
    collision_status[robotindex] = false; // NOTE: collision status for robot 0 is stored in collision_status[0].
    for (int j = 0; j < scan->ranges.size(); j++) {
        if (scan->ranges[j] < min_range) {
            collision_status[robotindex] = true;  // true indicates presence of obstacle
        }
    }
}

void GazeboTrain::image_Callback(const sensor_msgs::ImageConstPtr& image){
	//ROS_INFO("running image callback");
	
	img_data.data= *image;
	img_header   = image->header;

  if (img_data.data.data.size()>0){substatus[0] =1;}

}

void GazeboTrain::scan_Callback(const sensor_msgs::LaserScanConstPtr& scan){
  //ROS_INFO("running scan callback");
  scan_data.ranges = scan->ranges;
  scan_header  = scan->header;
  if (scan_data.ranges.size()>0){substatus[1] =1;}

  float min_range = 0.5;
  collision_status[current_robot] = false; // NOTE: collision status for robot 0 is stored in collision_status[0].
  for (int j = 0; j < scan->ranges.size(); j++) {
      if (scan->ranges[j] < min_range) {
        collision_status[current_robot] = true;  // true indicates presence of obstacle
      }
      //if (not(scan->ranges[j])){scan->ranges[j] = LidarMaxDistance;}
  }
}

void GazeboTrain::velocity_Callback(const nav_msgs::OdometryConstPtr& odom){
	//ROS_INFO("running velocity callback");
  substatus[2]=1;
	odom_data.vx = odom->twist.twist.linear.x;
	odom_data.vz = odom->twist.twist.angular.z;
	odom_header  = odom->header;
  
  if (std::isnan(odom->twist.twist.angular.z) && std::isnan(odom->twist.twist.linear.x)){
    odom_data.vx = 0;
    odom_data.vz = 0;
  }

}

// Bumper CallBack
void GazeboTrain::bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i) {
	//ROS_INFO("running bumper callback");
	// ROS_INFO("bumper hit. value = [%d] for robot %d", bumper_msg->bumper, i);
	if (bumper_msg->bumper == 1)
		collision_status[i] = true;
}


// Ground Truth callback
void GazeboTrain::gt_Callback(const gazebo_msgs::ModelStates gt){
	//ROS_INFO("running frame_trans callback");

  std::string current_robot_name ="turtlebot"+std::to_string(current_robot);
	for (int i = 0; i < gt.name.size(); i++){
		if(gt.name[i]== current_robot_name) {
		  gpose = gt.pose[i];
		}
	}
  substatus[3] =1;

}

// Update Goal service CallBack
bool GazeboTrain::cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response)
{
    ROS_INFO("Updatting Gazebo!");

    std::cout << "Request goal size: " << request.points.size() << std::endl;

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

    return true;
}

int GazeboTrain::create_sharedmemory(){

  for (int j = 0; j < num_robots; ++j)
    {
        vec2 temp_goal;
        temp_goal.x = 0.;
        temp_goal.y = 0.;
        current_goal.push_back(temp_goal);
    }
    path_length.resize(num_robots, 0.0); // Resize to size = no. of robots, init the new spaces to 0.0.
    time_elapsed.resize(num_robots, 0.0);

  reward_pub = nh.advertise<naviswarm::Reward>("/reward", 1000);

  for (int i=0;i<num_robots;i++){
    std::string name_space = "/turtlebot" + std::to_string(i);
    naviswarm::ActionObs actionobs_data;
    actionobs_data.ac_prev.vx = 0; 
    actionobs_data.ac_prev.vz = 0; 
    actionobs_data.ac_pprev.vx = 0; 
    actionobs_data.ac_pprev.vz = 0; 
    last_states.actionObsBatch.push_back(actionobs_data);
  }


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
    return 0;
}
// Infinite loop function
int GazeboTrain::train(){
    // This loop is equivalent to the for loop inside
    //StageNode::WorldCallback in drl_stageros.cpp
    naviswarm::States current_states;
    naviswarm::Transitions current_transitions;

    for(int i = 0; i < num_robots; i++){
      current_robot = i;

      std::string name_space = "/turtlebot" + std::to_string(i);

      /* synchronizer is slow
      message_filters::Subscriber<sensor_msgs::Image>   image_sub(nh, name_space + "/camera/image_raw", 1);
      //message_filters::Subscriber<sensor_msgs::Image>   image_sub(nh, name_space + "/camera/depth/image_raw", 1);
      message_filters::Subscriber<sensor_msgs::LaserScan> scan_sub(nh, name_space + "/scan", 1);

      //according to situation, choose synchronize odometry or not
      TimeSynchronizer<sensor_msgs::Image,sensor_msgs::LaserScan> sync(image_sub,scan_sub, 1);
      sync.registerCallback(boost::bind(& GazeboTrain::sync_Callback,this, _1, _2));
      */

      image_sub   = nh.subscribe<sensor_msgs::Image>(name_space + "/camera/image_raw", 1, &GazeboTrain::image_Callback, this); //"/camera/depth/image_raw"
      scan_sub    = nh.subscribe<sensor_msgs::LaserScan>(name_space + "/scan_filtered", 1, &GazeboTrain::scan_Callback, this);
      //velocity_sub  = nh.subscribe<nav_msgs::Odometry>(name_space + "/odom", 1, &GazeboTrain::velocity_Callback, this);
      groundtruth_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, &GazeboTrain::gt_Callback, this);
      bumper_sub    = nh.subscribe<kobuki_msgs::BumperEvent>(name_space + "/mobile_base/events/bumper", 1, boost::bind(&GazeboTrain::bumper_Callback, this, _1, i));

		  int checkstatus[4] = {1,1,0,1};
      bool condition=(substatus[0]!=checkstatus[0])||(substatus[1]!=checkstatus[1])||(substatus[3]!=checkstatus[3])||(substatus[2]!=checkstatus[2]);
		  while(condition){condition=(substatus[0]!=checkstatus[0])||(substatus[1]!=checkstatus[1])||(substatus[3]!=checkstatus[3])||(substatus[2]!=checkstatus[2]);}
		  for(int ind=0;ind<4;ind++){substatus[ind]=0;}

      // NOTE: Block to write the subscribed data into Shared memory
      naviswarm::Transition current_transition;
      // Initially the robot is at ground truth (x, y, theta)
      current_transition.pose.push_back(gpose.position.x);
      current_transition.pose.push_back(gpose.position.y);
      tf::Quaternion q(gpose.orientation.x, gpose.orientation.y, gpose.orientation.z, gpose.orientation.w);
      tf::Matrix3x3  m(q);
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
      //state.velObs.vel_now.vx = odom_data.vx;
      //state.velObs.vel_now.vz = odom_data.vz;
      //std::cout<<last_states.actionObsBatch[current_robot].ac_prev<<std::endl;
      state.velObs.vel_now.vx = last_states.actionObsBatch[current_robot].ac_prev.vx;
      state.velObs.vel_now.vz = last_states.actionObsBatch[current_robot].ac_prev.vz;
      
      //std::cout<<last_states.goalObsBatch.size()<<std::endl;
      //ROS_INFO("+++++++++++++++++++size+++++++++++++++++++");

      if (last_states.goalObsBatch.size() == 0) {
          state.goalObs.goal_pprev = state.goalObs.goal_now; //pprev is previous to previous goal. (We take 3 instances of observations.)
          state.goalObs.goal_prev = state.goalObs.goal_now;
      }
      else {
          state.goalObs.goal_pprev = last_states.goalObsBatch[current_robot].goal_prev;
          state.goalObs.goal_prev = last_states.goalObsBatch[current_robot].goal_now;
      }

      // TODO: Need to confirm if this is same as what was given in drl_stageros.
      state.scanObs.scan_now.ranges = scan_data.ranges;
      if (last_states.goalObsBatch.size() == 0) {
          //robotmodel->lasermodels[0]->GetSensors()[0].ranges;
          state.scanObs.scan_pprev = state.scanObs.scan_now;
          state.scanObs.scan_prev = state.scanObs.scan_now;
      }
      else {
          state.scanObs.scan_pprev = last_states.scanObsBatch[current_robot].scan_prev;
          state.scanObs.scan_prev = last_states.scanObsBatch[current_robot].scan_now;
      }

      state.ImageObs.image_now.data = img_data.data;
      if (last_states.goalObsBatch.size() == 0) {
          //robotmodel->lasermodels[0]->GetSensors()[0].ranges;
          state.ImageObs.image_p1rev.data = img_data.data;
          state.ImageObs.image_p2rev.data = img_data.data;
          state.ImageObs.image_p3rev.data = img_data.data;
          state.ImageObs.image_p4rev.data = img_data.data;
      }
      else {
          state.ImageObs.image_p4rev.data = last_states.ImageObsBatch[current_robot].image_p3rev.data;
          state.ImageObs.image_p3rev.data = last_states.ImageObsBatch[current_robot].image_p2rev.data;
          state.ImageObs.image_p2rev.data = last_states.ImageObsBatch[current_robot].image_p1rev.data;
          state.ImageObs.image_p1rev.data = last_states.ImageObsBatch[current_robot].image_now.data;
      }

      /*state.DepthObs.image_now.data = depth_data.data;
      if (last_states.goalObsBatch.size() == 0) {
          //robotmodel->lasermodels[0]->GetSensors()[0].ranges;
          state.DepthObs.image_p1rev.data = depth_data.data;
          state.DepthObs.image_p2rev.data = depth_data.data;
          state.DepthObs.image_p3rev.data = depth_data.data;
          state.DepthObs.image_p4rev.data = depth_data.data;
      }
      else {
          state.DepthObs.image_p4rev.data = state.DepthObs.image_p3rev.data;
          state.DepthObs.image_p3rev.data = state.DepthObs.image_p2rev.data;
          state.DepthObs.image_p2rev.data = state.DepthObs.image_p1rev.data;
          state.DepthObs.image_p1rev.data = state.DepthObs.image_now.data;
      }*/

      //
      if (last_states.goalObsBatch.size() == 0) {
          state.actionObs.ac_pprev.vx = 0.0;
          state.actionObs.ac_pprev.vz = 0.;
          state.actionObs.ac_prev.vx = 0.;
          state.actionObs.ac_prev.vz = 0.;
      }
      else {
          // should set last_states.data[r].acobs.ac_prev = actions.data[i] (this is ac_now)
          state.actionObs.ac_pprev = last_states.actionObsBatch[current_robot].ac_pprev;
          state.actionObs.ac_prev = last_states.actionObsBatch[current_robot].ac_prev;
      }

      // initial obs/state/transition
      if (last_states.goalObsBatch.size() == 0) {
          current_transition.state = state;
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
                      penalty_for_bigvz = -0.05*std::abs(executed_actions.data[current_robot].vz);
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
         reward_pub.publish(reward); // We need a publisher for reward
      }

      current_transitions.data.push_back(current_transition);
      current_states.scanObsBatch.push_back(state.scanObs);
      current_states.goalObsBatch.push_back(state.goalObs);
      current_states.actionObsBatch.push_back(state.actionObs);
      current_states.velObsBatch.push_back(state.velObs);
      current_states.ImageObsBatch.push_back(state.ImageObs);
      //current_states.DepthObsBatch.push_back(state.DepthObs);

    } // end of for loop

    last_states = current_states;
    /*
    ROS_INFO("++++++++state+++++++");
    std::cout<<current_transitions.data[0].state.velObs.vel_now.vx<<current_transitions.data[0].state.velObs.vel_now.vz<<current_transitions.data[1].state.velObs.vel_now.vx<<current_transitions.data[1].state.velObs.vel_now.vz<<std::endl;
    int infodatasize1 = current_transitions.data[0].state.scanObs.scan_now.ranges.size();
    int infodatasize2 = current_transitions.data[1].state.scanObs.scan_now.ranges.size();
    int infodatasize3 = current_transitions.data[0].state.scanObs.scan_pprev.ranges.size();
    int infodatasize4 = current_transitions.data[1].state.scanObs.scan_pprev.ranges.size();
    int infodatasize5 = current_transitions.data[0].state.scanObs.scan_prev.ranges.size();
    int infodatasize6 = current_transitions.data[1].state.scanObs.scan_prev.ranges.size();
    std::cout<<infodatasize1<<'|'<<infodatasize2<<'|'<<infodatasize3<<'|'<<infodatasize4<<'|'<<infodatasize5<<'|'<<infodatasize6<<std::endl;
    int infodatasiz1 = current_transitions.data[0].state.ImageObs.image_now.data.data.size();
    int infodatasiz2 = current_transitions.data[1].state.ImageObs.image_now.data.data.size();
    int infodatasiz3 = current_transitions.data[0].state.ImageObs.image_p1rev.data.data.size();
    int infodatasiz4 = current_transitions.data[1].state.ImageObs.image_p1rev.data.data.size();
    int infodatasiz5 = current_transitions.data[0].state.ImageObs.image_p2rev.data.data.size();
    int infodatasiz6 = current_transitions.data[1].state.ImageObs.image_p2rev.data.data.size();
    int infodatasiz7 = current_transitions.data[0].state.ImageObs.image_p3rev.data.data.size();
    int infodatasiz8 = current_transitions.data[1].state.ImageObs.image_p3rev.data.data.size();
    int infodatasiz9 = current_transitions.data[0].state.ImageObs.image_p4rev.data.data.size();
    int infodatasiz0 = current_transitions.data[1].state.ImageObs.image_p4rev.data.data.size();
    std::cout<<infodatasiz1<<'|'<<infodatasiz2<<'|'<<infodatasiz3<<'|'<<infodatasiz4<<'|'<<infodatasiz5<<'|'<<infodatasiz6<<'|'<<infodatasiz7<<'|'<<infodatasiz8<<'|'<<infodatasiz9<<'|'<<infodatasiz0<<std::endl;
    std::cout<<current_transitions.data[0].state.goalObs.goal_now.goal_dist<<'|'<<current_transitions.data[0].state.goalObs.goal_now.goal_theta<<'/'<<current_transitions.data[0].state.goalObs.goal_prev.goal_dist<<'|'<<current_transitions.data[0].state.goalObs.goal_prev.goal_theta<<'/'<<current_transitions.data[0].state.goalObs.goal_pprev.goal_dist<<'|'<<current_transitions.data[0].state.goalObs.goal_pprev.goal_theta<<std::endl;
    std::cout<<current_transitions.data[1].state.goalObs.goal_now.goal_dist<<'|'<<current_transitions.data[1].state.goalObs.goal_now.goal_theta<<'/'<<current_transitions.data[1].state.goalObs.goal_prev.goal_dist<<'|'<<current_transitions.data[1].state.goalObs.goal_prev.goal_theta<<'/'<<current_transitions.data[1].state.goalObs.goal_pprev.goal_dist<<'|'<<current_transitions.data[1].state.goalObs.goal_pprev.goal_theta<<std::endl;
    std::cout<<current_transitions.data[0].state.actionObs.ac_prev.vx<<'|'<<current_transitions.data[0].state.actionObs.ac_prev.vz<<'|'<<current_transitions.data[0].state.actionObs.ac_pprev.vx<<'|'<<current_transitions.data[0].state.actionObs.ac_pprev.vz<<'/'<<current_transitions.data[1].state.actionObs.ac_prev.vx<<'|'<<current_transitions.data[1].state.actionObs.ac_prev.vz<<'|'<<current_transitions.data[1].state.actionObs.ac_pprev.vx<<'|'<<current_transitions.data[1].state.actionObs.ac_pprev.vz<<std::endl;
	*/
	
    //ROS_INFO("lock memory");
    acquire_semaphore();
    uint32_t length = ros::serialization::serializationLength(current_transitions);
    // ROS_INFO("current state length is %d", length);
    boost::shared_array<uint8_t> buffer(new uint8_t[length]);
    ros::serialization::OStream stream(buffer.get(), length);
    ros::serialization::serialize(stream, current_transitions);
    memcpy(share_addr, &length, 4);
    memcpy(share_addr+4, buffer.get(), length);
    release_semaphore();
    std::cout<<length<<std::endl;
    //ROS_INFO("lock released");

    // wait command processed
    bool succ = false;
    while (!succ && ros::ok()){
      // wait for client to modify this value
      //ROS_INFO("locked for actions");
      ros::Duration(0.005);
      acquire_semaphore();
      int new_length = *(int *) share_addr;
      //std::cout<< "Length ="<< length <<" New length = "<< new_length << std::endl;
      if (new_length != length) {
        succ = true;
        std::cout<<"data writen"<<std::endl;
      } // Write has succeeded

      if (succ)
      {
          naviswarm::Actions actions;
          ros::serialization::IStream stream((share_addr + 4), new_length);
          ros::serialization::Serializer<naviswarm::Actions>::read(stream, actions); // Reads actions from shared memory
          release_semaphore();
          ROS_INFO("got data and released memory");
          if (actions.data.size() != num_robots){
              ROS_INFO("actions_size != robots_size, actions_size is %d", static_cast<int>(actions.data.size()));
              ROS_BREAK();
          }
          
          std::cout<<actions.data[0]<<std::endl;
          std::cout<<actions.data[1]<<std::endl;
          std::cout<<actions.data[2]<<std::endl;
          std::cout<<actions.data[3]<<std::endl;
          ROS_INFO("=========");
          for (int j = 0 ; j < actions.data.size(); ++j){
              std::cout<<actions.data[j]<<std::endl;

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
}


int main(int argc, char **argv){
  ros::init(argc, argv, "drl_gazeboros");
  GazeboTrain gazeboc(4);

  if(gazeboc.create_sharedmemory() != 0)
        exit(-1);

  boost::thread t = boost::thread(boost::bind(&ros::spin));

  while(ros::ok() ){ //TODO: add method to check if gazebo is running
    gazeboc.train();
  }
  t.join();
  exit(0);
}
