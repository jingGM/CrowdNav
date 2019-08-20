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
 #include <naviswarm/ScanObs.h>
 #include <naviswarm/CameraImageObs.h>
 
 #include <naviswarm/Goal.h>
 #include <naviswarm/Scan.h>
 #include <naviswarm/State.h>
 #include <naviswarm/States.h>
 #include <naviswarm/Transition.h>
 #include <naviswarm/Transitions.h>
 #include <naviswarm/RobotStatus.h>
 #include <naviswarm/Reward.h>
 #include <naviswarm/SCtoCP.h>
 #include <naviswarm/waypoints.h>


 // Service header
 #include <naviswarm/UpdateModel.h>

 // for the IPC (Inter-Process Communication) part
 #include <sys/ipc.h>   /* for system's IPC_xxx definitions */
 #include <sys/shm.h>   /* for shmget, shmat, shmdt, shmctl */
 #include <sys/sem.h>   /* for semget, semctl, semop */
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

 #define LidarMinDistance 0.5
 #define LidarMaxDistance 5

//Class definition
class GazeboTrain {
  private:
    // Defining vec3 = {x, y, theta}
    typedef struct {
        double x, y, theta;
    } vec3;

    // Defining vec3 = {x, y}
    typedef struct {
        double x, y;
    } vec2;

    ros::Subscriber groundtruth_sub;
    struct Robotinfo
    {
        ros::Subscriber bumper_sub; //one odom
        ros::Subscriber scan_sub; //multiple lasers
        ros::Subscriber image_sub;
        ros::Subscriber velocity_sub;
    };
    std::vector<Robotinfo *> robots_info;


    bool resettingflag=false;
    ros::NodeHandle nh;

    boost::mutex msg_lock;
    // for shared memory (sem -> semaphore)
    int share_id, sem_id;
    uint8_t *share_addr;

    int num_robots = 1; // The actual value is assigned in the Constructor. By default it is 1.
    std::vector<bool> collision_status;
    std::vector<geometry_msgs::Pose> gpose;
    std::vector<naviswarm::Velocity> odom_data;
    std::vector<naviswarm::Scan> scan_data;
    std::vector<naviswarm::CameraImage> img_data;
    std::vector<naviswarm::waypoints> waypoint_data;

    std::vector<std_msgs::Header> img_header;
    std::vector<std_msgs::Header> scan_header;
    std::vector<std_msgs::Header> odom_header;

    ros::Publisher reward_pub;

  

    int current_steps = 0;
    
    int substatus[4] = {0,0,0,0};//check if get messages
    
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

    // Used to remember initial poses for soft reset
    // Soft reset is when robots are spawned again in the original initial poses.
    std::vector<geometry_msgs::Pose> initial_poses;
    ros::ServiceServer reset_srv_;

    // Service to reset simulation with a set new initial poses and goals for each robot.
    ros::ServiceServer update_srv=nh.advertiseService("update_goals", &GazeboTrain::cb_update_srv, this);

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

    // for semaphore operation
    void acquire_semaphore();
    void release_semaphore();

   
    

  public:
    // Function declarations
    GazeboTrain(int n);

    void gt_Callback(const gazebo_msgs::ModelStates gt);
    void image_Callback(const sensor_msgs::ImageConstPtr& image, int i);
    void scan_Callback(const sensor_msgs::LaserScanConstPtr& scan, int i);
    void bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i);
    //void velocity_Callback(const nav_msgs::OdometryConstPtr& odom, int i);

    bool cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response);
    int create_sharedmemory();
   
    int train();
};

// Constructor for the class
GazeboTrain::GazeboTrain(int n){
  
  geometry_msgs::Pose gpose_;
  naviswarm::Velocity odom_;
  naviswarm::Scan scan_;
  naviswarm::CameraImage img_;
  std_msgs::Header imgh_;
  std_msgs::Header scanh_;
  std_msgs::Header odomh_;
  gpose_.orientation.w=1;
  naviswarm::waypoints waypoint_;


  for(int i=0;i<n;i++){
    
    gpose.push_back(gpose_);
    odom_data.push_back(odom_);
    scan_data.push_back(scan_);
    img_data.push_back(img_);
    img_header.push_back(imgh_);
    odom_header.push_back(odomh_);
    scan_header.push_back(scanh_);
    waypoint_data.push_back(waypoint_);
  }
//std::cout<<"robots number: "<<collision_status.size()<<"/"<<gpose.size()<<std::endl;

  num_robots = n;
  
    //ROS_INFO("++++++++++++++");
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

void GazeboTrain::image_Callback(const sensor_msgs::ImageConstPtr& image, int i){
  //ROS_INFO("running image in");
  //std::cout<<"i= "<<i<<std::endl;
  if (resettingflag){}
  else{
    img_data[i].data= *image;
    img_header[i]   = image->header;

    
    if (last_states.ImageObsBatch.size() == 0) {
        last_states.ImageObsBatch[i].image_p1rev.data = img_data[i].data;
        last_states.ImageObsBatch[i].image_p2rev.data = img_data[i].data;
    }
    else {
        last_states.ImageObsBatch[i].image_p2rev.data = last_states.ImageObsBatch[i].image_p1rev.data;
        last_states.ImageObsBatch[i].image_p1rev.data = last_states.ImageObsBatch[i].image_now.data;
    }
    last_states.ImageObsBatch[i].image_now.data = img_data[i].data;

    //if (i==0){ROS_INFO(" ");}

    if (img_data[i].data.data.size()>0){substatus[0] =1;}
    //ROS_INFO("running image callback");
    usleep(100000);
  }
  
}

void GazeboTrain::scan_Callback(const sensor_msgs::LaserScanConstPtr& scan, int i){
  //ROS_INFO("running scan in");
  //std::cout<<"i= "<<i<<std::endl;
  if (resettingflag){
    //std::cout<<"in reseting: "<<i<<std::endl;
    //std::cout<<i<<"/"<<collision_status[0]<<"/"<<collision_status[1]<<"/"<<collision_status[2]<<"/"<<collision_status[3]<<std::endl;
    }
  else{
    scan_data[i].ranges = scan->ranges;
    scan_header[i]  = scan->header;

    
    if (last_states.scanObsBatch.size() == 0) {
        last_states.scanObsBatch[i].scan_pprev.ranges = scan_data[i].ranges;
        last_states.scanObsBatch[i].scan_prev.ranges = scan_data[i].ranges;
    }
    else {
        last_states.scanObsBatch[i].scan_pprev = last_states.scanObsBatch[i].scan_prev;
        last_states.scanObsBatch[i].scan_prev = last_states.scanObsBatch[i].scan_now;
    }
    last_states.scanObsBatch[i].scan_now.ranges = scan_data[i].ranges;


    if (scan_data[i].ranges.size()>0){substatus[1] =1;}

    float min_range = 5;
    //collision_status[current_robot] = false; // NOTE: collision status for robot 0 is stored in collision_status[0].
    for (int j = 0; j < scan->ranges.size(); j++) {
        if (scan->ranges[j] < min_range) {
          min_range = scan->ranges[j];
        }
    }

    //std::cout<<collision_status.size()<<std::endl;
    //std::cout<<collision_status[0]<<collision_status[1]<<collision_status[2]<<collision_status[3]<<std::endl;
    if (min_range<LidarMinDistance){
      collision_status[i] = true;
      //std::cout<<"set collision: "<<i<<std::endl;
    }
    //std::cout<<"min lidar: "<<min_range<<std::endl;
    //std::cout<<i<<"/"<<collision_status[0]<<"/"<<collision_status[1]<<"/"<<collision_status[2]<<"/"<<collision_status[3]<<std::endl;
    //ROS_INFO("running scan callback");
    usleep(100000);
  }

}

// Bumper CallBack
void GazeboTrain::bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i) {
  if (bumper_msg->bumper == 1)
    collision_status[i] = true;
  //ROS_INFO("running bumper callback");
}


// Ground Truth callback
void GazeboTrain::gt_Callback(const gazebo_msgs::ModelStates gt){
// ROS_INFO("running frame_trans in"); 
  if (resettingflag){}
  else{
    for (int j=0; j< num_robots;j++){
      for (int i = 0; i < gt.name.size(); i++){
        std::string current_robot_name ="turtlebot"+std::to_string(j);
        if(gt.name[i]== current_robot_name)
          gpose[j] = gt.pose[i];
      }
    }
    substatus[3] =1;
    // ROS_INFO("running frame_trans callback");
  }
}

// Update Goal service CallBack
bool GazeboTrain::cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response)
{

    ROS_INFO("Updatting Gazebo!");
    resettingflag = true;

    std::cout << "Request goal size: " << request.points.size() << std::endl;
    usleep(2000000);
    resettingflag = false;
    std::cout<<"resetted"<<std::endl;

    if (request.points.size() != num_robots) {
        ROS_WARN("Robot Number and number of goals don't match!");
        response.success = false;
    }
    else { // no. of poses and no. of robots match (no problem)

        // WARNING: IT MAY NOT FREE THE MEMORY SPACE
        // NOTE: last_states is a private datamember of the stage_node class.
        last_states.actionObsBatch.clear();
        last_states.goalObsBatch.clear();
        last_states.scanObsBatch.clear();
        last_states.velObsBatch.clear();
        last_states.ImageObsBatch.clear();

        executed_actions.data.clear();

        current_steps = 0;
        // for (int i=0;i<num_robots;i++){
        //   collision_status[i]=false;
        // }
        // WARNING: IT MAY NOT FREE THE MEMORY SPACE
        collision_status.clear();
        gpose.clear();
        odom_data.clear();
        scan_data.clear();
        img_data.clear();
        img_header.clear();
        scan_header.clear();
        odom_header.clear();
        waypoint_data.clear();


        bool collision_ = false;
        geometry_msgs::Pose gpose_;
        naviswarm::Velocity odom_;
        naviswarm::Scan scan_;
        naviswarm::CameraImage img_;
        std_msgs::Header imgh_;
        std_msgs::Header scanh_;
        std_msgs::Header odomh_;
        gpose_.orientation.w=1;
        naviswarm::waypoints waypoint_;
        
        for(int i=0;i<num_robots;i++){

          naviswarm::ScanObs scanobs_;
          naviswarm::CameraImageObs imageobs_;
          //naviswarm::GoalObs goalobs_;
          naviswarm::ActionObs actionobs_;
          naviswarm::VelocityObs velobs_;

          collision_status.push_back(collision_);
          gpose.push_back(gpose_);
          odom_data.push_back(odom_);
          scan_data.push_back(scan_);
          img_data.push_back(img_);
          img_header.push_back(imgh_);
          odom_header.push_back(odomh_);
          scan_header.push_back(scanh_);
          last_states.scanObsBatch.push_back(scanobs_);
          last_states.ImageObsBatch.push_back(imageobs_);
          //last_states.goalObsBatch.push_back(goalobs_);
          last_states.actionObsBatch.push_back(actionobs_);
          last_states.velObsBatch.push_back(velobs_);

          waypoint_data.push_back(waypoint_);
        }

        // Setting the new goals
        for (size_t r = 0; r < request.points.size(); r++) {
            vec2 goal;
            goal.x = request.points[r].x;
            goal.y = request.points[r].y;
            current_goal[r] = goal;
            ROS_INFO("Goal_%d: %.3f, %.3f", int(r), goal.x, goal.y);

            // naviswarm::waypoint waypoint_;
            // for(size_t NoWP=0; NoWP<request.waypoints[r].size(); NoWP++){
            //   geometry_msgs::Point position;
            //   position.x = request.waypoints[r].data[NoWP].x;
            //   position.y = request.waypoints[r].data[NoWP].y;
            //   waypoint_.push_back(position);
            // }
            waypoint_data[r].data=request.waypoints[r].data;
        }
        std::cout<<waypoint_data[0].data[0].x<<std::endl;
        std::cout<<waypoint_data[0].data[1].x<<std::endl;
        std::cout<<waypoint_data[1].data[0].x<<std::endl;
        std::cout<<waypoint_data[1].data[1].x<<std::endl;

        response.success = true;
    }

    //std::cout<<"collision status: "<<collision_status.size()<<"/"<<collision_status[0]<<"/"<<collision_status[1]<<"/"<<collision_status[2]<<"/"<<collision_status[3]<<"/"<<std::endl;
    //std::cout<<gpose.size()<<"/"<<scan_data.size()<<"/"<<img_data.size()<<std::endl;
    usleep(1000000);
    //std::cout<<"collision status: "<<collision_status.size()<<"/"<<collision_status[0]<<"/"<<collision_status[1]<<"/"<<collision_status[2]<<"/"<<collision_status[3]<<"/"<<std::endl;

    return true;
}

int GazeboTrain::create_sharedmemory(){



  for (int i = 0; i < num_robots; ++i)
    {
      std::string name_space = "/turtlebot" + std::to_string(i);

      naviswarm::ScanObs scanobs_;
      naviswarm::CameraImageObs imageobs_;
      //naviswarm::GoalObs goalobs_;
      naviswarm::VelocityObs velobs_;
      naviswarm::ActionObs actionobs_data;
      actionobs_data.ac_prev.vx = 0; 
      actionobs_data.ac_prev.vz = 0; 
      actionobs_data.ac_pprev.vx = 0; 
      actionobs_data.ac_pprev.vz = 0;
      bool collision_ = false;
      
      collision_status.push_back(collision_);
      last_states.actionObsBatch.push_back(actionobs_data);
      last_states.scanObsBatch.push_back(scanobs_);
      last_states.ImageObsBatch.push_back(imageobs_);
      //last_states.goalObsBatch.push_back(goalobs_);
      last_states.velObsBatch.push_back(velobs_);

      vec2 temp_goal;
      temp_goal.x = 0.;
      temp_goal.y = 0.;
      current_goal.push_back(temp_goal);

      Robotinfo* new_robot = new Robotinfo;

      new_robot->image_sub   = nh.subscribe<sensor_msgs::Image>(name_space + "/camera/depth/image_raw", 1, boost::bind(&GazeboTrain::image_Callback, this, _1, i)); //"/camera/depth/image_raw"
      new_robot->scan_sub    = nh.subscribe<sensor_msgs::LaserScan>(name_space + "/scan_filtered", 1, boost::bind(&GazeboTrain::scan_Callback, this, _1, i));
      new_robot->bumper_sub  = nh.subscribe<kobuki_msgs::BumperEvent>(name_space + "/mobile_base/events/bumper", 1, boost::bind(&GazeboTrain::bumper_Callback, this, _1, i));

      robots_info.push_back(new_robot);
    }
    groundtruth_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, &GazeboTrain::gt_Callback, this);
    path_length.resize(num_robots, 0.0); 
    time_elapsed.resize(num_robots, 0.0);

  reward_pub = nh.advertise<naviswarm::Reward>("/reward", 1000);

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
    // ROS_INFO("time");
    // This loop is equivalent to the for loop inside
    //StageNode::WorldCallback in drl_stageros.cpp
    naviswarm::States current_states;
    naviswarm::Transitions current_transitions;

    for(int i = 0; i < num_robots; i++){
      current_robot = i;

      //Robotinfo const * robotmodel = this->robots_info[i];


    //std::cout<<substatus[0]<<"/"<<substatus[1]<<"/"<<substatus[2]<<"/"<<substatus[3]<<std::endl;
     int checkstatus[4] = {1,1,0,1};
     bool condition=(substatus[0]!=checkstatus[0])||(substatus[1]!=checkstatus[1])||(substatus[3]!=checkstatus[3])||(substatus[2]!=checkstatus[2]);
     while(condition){
      condition=(substatus[0]!=checkstatus[0])||(substatus[1]!=checkstatus[1])||(substatus[3]!=checkstatus[3])||(substatus[2]!=checkstatus[2]);}
     for(int ind=0;ind<4;ind++){substatus[ind]=0;}


      naviswarm::Transition current_transition;

      current_transition.pose.push_back(gpose[i].position.x);
      current_transition.pose.push_back(gpose[i].position.y);
      tf::Quaternion q(gpose[i].orientation.x, gpose[i].orientation.y, gpose[i].orientation.z, gpose[i].orientation.w);
      tf::Matrix3x3  m(q);
      double roll, pitch, yaw;
      m.getRPY(roll, pitch, yaw);
      current_transition.pose.push_back(yaw);

      naviswarm::State state;
      tf::Transform gt = pose2transform(gpose[i]);
      

      vec2 local_goal = getTransformedPoint(current_goal[i], gt.inverse());
      state.goalObs.goal_now.goal_dist = GetDistance(local_goal.x, local_goal.y);
      state.goalObs.goal_now.goal_theta = atan2(local_goal.y, local_goal.x);


 //std::cout<<"goal"<<i<<": "<< state.goalObs.goal_now.goal_theta << std::endl;


      state.velObs.vel_now.vx = last_states.actionObsBatch[i].ac_prev.vx;
      state.velObs.vel_now.vz = last_states.actionObsBatch[i].ac_prev.vz;
      

      if (last_states.goalObsBatch.size() == 0) {
          state.goalObs.goal_pprev = state.goalObs.goal_now; 
          state.goalObs.goal_prev = state.goalObs.goal_now;
      }
      else {
          state.goalObs.goal_pprev = last_states.goalObsBatch[i].goal_prev;
          state.goalObs.goal_prev = last_states.goalObsBatch[i].goal_now;
      }


      state.scanObs.scan_now = last_states.scanObsBatch[i].scan_now;
      if (last_states.goalObsBatch.size() == 0) {
          state.scanObs.scan_pprev = state.scanObs.scan_now;
          state.scanObs.scan_prev = state.scanObs.scan_now;
      }
      else {
          state.scanObs.scan_pprev = last_states.scanObsBatch[i].scan_pprev;
          state.scanObs.scan_prev = last_states.scanObsBatch[i].scan_prev;
      }

//std::cout<<"scan"<<i<<":  "<< state.scanObs.scan_pprev.ranges.size()<<"/"<<state.scanObs.scan_prev.ranges.size()<<"/"<<state.scanObs.scan_now.ranges.size()<<"/"<<std::endl;

      state.ImageObs.image_now.data = last_states.ImageObsBatch[i].image_now.data;
      if (last_states.goalObsBatch.size() == 0) {
          state.ImageObs.image_p1rev.data = state.ImageObs.image_now.data;
          state.ImageObs.image_p2rev.data = state.ImageObs.image_now.data;
      }
      else {
          state.ImageObs.image_p2rev.data = last_states.ImageObsBatch[i].image_p2rev.data;
          state.ImageObs.image_p1rev.data = last_states.ImageObsBatch[i].image_p1rev.data;
//std::cout<<state.ImageObs.image_now.data.header.frame_id<<":  "<<state.ImageObs.image_p2rev.data.header.stamp<<"/"<<state.ImageObs.image_p1rev.data.header.stamp<<"/"<<state.ImageObs.image_now.data.header.stamp<<std::endl;
      }
      


      //
      if (last_states.goalObsBatch.size() == 0) {
          state.actionObs.ac_pprev.vx = 0.0;
          state.actionObs.ac_pprev.vz = 0.;
          state.actionObs.ac_prev.vx = 0.;
          state.actionObs.ac_prev.vz = 0.;
      }
      else {
          state.actionObs.ac_pprev = last_states.actionObsBatch[current_robot].ac_pprev;
          state.actionObs.ac_prev = last_states.actionObsBatch[current_robot].ac_prev;
      }
//ROS_INFO("------actions------");
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

          double reward_approaching_goal = 0;
          double penalty_for_bigvz = 0;
          double penalty_for_time = 0;
          double distance_to_obstacle = 0;
          double reached_way_point = 0;
          
          if (state.goalObs.goal_now.goal_dist < 0.5) {  // arrived the goal
              current_transition.terminal = true;
              double reched_goal_reward = 20;
              current_transition.reward = reched_goal_reward;
              reward.reached_goal = reched_goal_reward;
          }
          else // if goal has not been reached
          {
            //ROS_INFO("----reward----");
              // rs.stalled[r] = collision;
            std::cout<<"collision status"<<i<<":  "<<collision_status[i]<<std::endl;

              if(collision_status[i] == true) { // stalled is obtained from an in-built function from stage. we must write a function to detect collisions
                  current_transition.terminal = true;
                  double collision_penalty =-20;
                  current_transition.reward = collision_penalty;
                  reward.collision = collision_penalty;
              }
              else { // Goal not reached and no collisions
                  //ROS_INFO("in else");
                  double penalty_for_deviation = 0.0;
                  
                  if (std::abs(state.goalObs.goal_now.goal_theta) > 0.785)
                  {
                      penalty_for_deviation = -0.1 * (std::abs(state.goalObs.goal_now.goal_theta) - 0.785);
                  }

                  cv_bridge::CvImagePtr cvPtr;
                  try {
                    cvPtr = cv_bridge::toCvCopy(state.ImageObs.image_now.data, "32FC1");
                  } 
                  catch (cv_bridge::Exception& e) {
                    ROS_ERROR("cv_bridge exception: %s", e.what());
                  }
                  cv::Mat cam_data = cvPtr->image;
                  cam_data.setTo(5, cam_data!=cam_data );  
                  double depthmin, depthmax;
                  cv::minMaxLoc(cam_data, &depthmin, &depthmax);
                  std::cout<<"min:"<<depthmin<<",  max:"<<depthmax<<std::endl;
                  

                  for (int waypoint_index =0;waypoint_index<waypoint_data[i].data.size();waypoint_index++){
                    double distx = std::abs(gpose[i].position.x - waypoint_data[i].data[waypoint_index].x);
                    double disty = std::abs(gpose[i].position.y - waypoint_data[i].data[waypoint_index].y); 
                    double distance_to_waypoint = GetDistance(distx,disty);
                    if (distance_to_waypoint<0.4){reached_way_point +=10;}
                  }
                  
                  


                  current_transition.terminal = false;

                  reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
                  penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.1);
                  penalty_for_time = (current_steps+1) *(0);
                  distance_to_obstacle = -(LidarMaxDistance-depthmin)*0.1;
                  
                  current_transition.reward = reward_approaching_goal + penalty_for_bigvz + penalty_for_time+distance_to_obstacle+reached_way_point;
                  reward.reward_approaching_goal = reward_approaching_goal;
                  reward.penalty_for_deviation = penalty_for_deviation;
                  
              }
          }

        std::cout<<"Robot:"<<i<<" Rew: "<<current_transition.reward<<"  AG: "<<reward_approaching_goal<<" CurrSt: "<<penalty_for_time<<" Osc:"<<penalty_for_bigvz<<"  DtO: "<<distance_to_obstacle<<" RW:"<<reached_way_point<<"  GDist: "<<state.goalObs.goal_now.goal_dist<<" CollStat: "<<std::to_string(collision_status[i])<<std::endl;
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
    //std::cout<<length<<std::endl;
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
        //std::cout<<"data writen"<<std::endl;
      } // Write has succeeded

      if (succ)
      {
          
          naviswarm::SCtoCP read_data_as;
          ros::serialization::IStream stream((share_addr + 4), new_length);
          ros::serialization::Serializer<naviswarm::SCtoCP>::read(stream, read_data_as); // Reads actions from shared memory
          release_semaphore();

          naviswarm::Actions actions = read_data_as.actions;
          current_steps = read_data_as.step;

          
          //std::cout<<"===steps:  "<<current_steps<<"==="<<std::endl;
          //std::cout<<actions.data[0]<<std::endl;
          //std::cout<<actions.data[1]<<std::endl;
          //std::cout<<actions.data[2]<<std::endl;
          //std::cout<<actions.data[3]<<std::endl;
          
          //ROS_INFO("got data and released memory");
          if (actions.data.size() != num_robots){
              ROS_INFO("actions_size != robots_size, actions_size is %d", static_cast<int>(actions.data.size()));
              ROS_BREAK();
          }
          
          //usleep(300000);
          for (int j = 0 ; j < num_robots; ++j){
            std::string topicname = "/turtlebot"+std::to_string(j)+"/cmd_vel_mux/input/navi";
            ros::Publisher pubrobotvelocity = nh.advertise<geometry_msgs::Twist>(topicname, 1);
            geometry_msgs::Twist action;
            action.linear.x = actions.data[j].vx;
            action.angular.z = actions.data[j].vz;
            int velocitycounter = 0;
            while ( (velocitycounter == 0) && ros::ok()){
              if (pubrobotvelocity.getNumSubscribers()>0){
                pubrobotvelocity.publish(action);
                velocitycounter = 1;
              }
            }
            std::cout<<"robot"<<j<<":   "<<action.linear.x<<"|"<<action.angular.z<<std::endl;
            usleep(100000);
            //std::cout<<"-"; 
            last_states.actionObsBatch[j].ac_pprev = last_states.actionObsBatch[j].ac_prev;
            last_states.actionObsBatch[j].ac_prev = actions.data[j];
            //ros::Duration(1);
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
  //ros::Rate r(10);

  GazeboTrain gazeboc(8);

  if(gazeboc.create_sharedmemory() != 0)
        exit(-1);

  boost::thread t = boost::thread(boost::bind(&ros::spin));
  usleep(1000000);

  while(ros::ok() ){ //TODO: add method to check if gazebo is running
    gazeboc.train();
  }
  t.join();
  exit(0);
}
