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
 #include <naviswarm/Reward.h>
 #include <naviswarm/CameraImage.h>

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


class GazeboTrain {
	private:
		ros::NodeHandle nh;


		void setvelocities( int robotindex, naviswarm::Action velocity){
	      std::string topicname = "/turtlebot"+std::to_string(1)+"/cmd_vel_mux/input/navi";
	      ros::Publisher pubrobotvelocity = nh.advertise<geometry_msgs::Twist>(topicname, 1000);
	      geometry_msgs::Twist action;
	      action.linear.x = velocity.vx;
	      action.angular.z = velocity.vz;
	      int counter = 0;
	      while ( (counter == 0) && ros::ok()){
	        if (pubrobotvelocity.getNumSubscribers()>0){
	          pubrobotvelocity.publish(action);
	          counter = 1;
	        }
	      }
	    }

		int num_robots = 2; // The actual value is assigned in the Constructor. By default it is 1.
	    std::vector<bool> collision_status;

	    int num_episode;
	    int current_robot;

////////////////////////////////////////////////////////----------------------------------------------------------------------------
	    ros::Subscriber groundtruth_sub; // Subscriber for Groundtruth data from Gazebo
		ros::Subscriber bumper_sub;
		ros::Publisher reward_pub;

		geometry_msgs::Pose gpose;
	    //cv::Mat img_data; // stores image frames published in /camera/rgb/image_raw/compressed converted to Mat format.
	    //cv::Mat depth_data;
	    naviswarm::Velocity 	odom_data;
	    naviswarm::Scan 		scan_data;
	    naviswarm::CameraImage 	depth_data;
	    naviswarm::CameraImage 	img_data;

	    std_msgs::Header depth_header;
		std_msgs::Header img_header;
		std_msgs::Header scan_header;
		std_msgs::Header odom_header;

	public:
	    // Function declarations
	    GazeboTrain(int n){
	    }
	    
	    void gt_Callback(const gazebo_msgs::ModelStates gt);
	    void sync_Callback( const sensor_msgs::ImageConstPtr& image,
							const sensor_msgs::ImageConstPtr& depth_image,
							const sensor_msgs::LaserScanConstPtr& scan,
							const nav_msgs::OdometryConstPtr& odom);
	    void bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i);
	    bool cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response);

	    void runvelocity(){
	    	naviswarm::Action action;
	    	action.vx = 1;
	    	setvelocities( 0, action);
	    }


////////////////////////////////////////////////////////////------------------------------------------------------------------------------
	    void subscribedata(){

			for(int i = 0; i < num_robots; i++){
				current_robot = i;

				std::string name_space = "/turtlebot" + std::to_string(i);

				message_filters::Subscriber<sensor_msgs::Image> 	image_sub(nh, name_space + "/camera/image_raw", 1);
				message_filters::Subscriber<sensor_msgs::Image> 	depth_sub(nh, name_space + "/camera/depth/image_raw", 1);
				message_filters::Subscriber<sensor_msgs::LaserScan> scan_sub(nh, name_space + "/scan", 1);
				message_filters::Subscriber<nav_msgs::Odometry> 	velocity_sub(nh, name_space + "/odom", 1);

				TimeSynchronizer<sensor_msgs::Image,sensor_msgs::Image,sensor_msgs::LaserScan,nav_msgs::Odometry> sync(image_sub, depth_sub,scan_sub,velocity_sub, 4);
				sync.registerCallback(boost::bind(& GazeboTrain::sync_Callback,this, _1, _2, _3, _4));

				groundtruth_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 100, &GazeboTrain::gt_Callback, this);
				bumper_sub 		= nh.subscribe<kobuki_msgs::BumperEvent>(name_space + "/mobile_base/events/bumper", 50, boost::bind(&GazeboTrain::bumper_Callback, this, _1, i));
				//new_robot->reward_pub = nh.advertise<naviswarm::Reward>(name_space + "/reward", 100);

				ros::Rate loop_rate(50);
				ros::spin(); // Call the gt, scan and odom callback functions once
			}
	    }

};


void GazeboTrain::sync_Callback(const sensor_msgs::ImageConstPtr& image,
								const sensor_msgs::ImageConstPtr& depth_image,
								const sensor_msgs::LaserScanConstPtr& scan,
								const nav_msgs::OdometryConstPtr& odom){
	int robotindex = current_robot;

	depth_data.data  = depth_image->data;
	img_data.data 	 = image->data;
	scan_data.ranges = scan->ranges;
	odom_data.vx 	 = odom->twist.twist.linear.x;
  	odom_data.vz 	 = odom->twist.twist.angular.z;

  	depth_header = depth_image->header;
	img_header   = image->header;
	scan_header  = scan->header;
	odom_header  = odom->header;

	float min_range = 0.5;
    collision_status[robotindex] = false; // NOTE: collision status for robot 0 is stored in collision_status[0].
    for (int j = 0; j < scan->ranges.size(); j++) {
        if (scan->ranges[j] < min_range) {
            collision_status[robotindex] = true;  // true indicates presence of obstacle
        }
    }

    /*cv_bridge::CvImagePtr cvPtr;
	try {
	cvPtr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
	// ROS_INFO("Inside image callback");
	} catch (cv_bridge::Exception& e) {
	ROS_ERROR("cv_bridge exception: %s", e.what());
	return;
	}
	img_data = cvPtr->image;
	*/
}

// Bumper CallBack
void GazeboTrain::bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i) {
  // ROS_INFO("bumper hit. value = [%d] for robot %d", bumper_msg->bumper, i);
  if (bumper_msg->bumper == 1)
    collision_status[i] = true;
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

int main(int argc, char **argv){
  ros::init(argc, argv, "testcpp");
  GazeboTrain train(2);

  //train.runvelocity();
  train.subscribedata();
  return 0;
}