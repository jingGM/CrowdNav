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
#include <cmath>       /* isnan, sqrt */
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
		ros::Subscriber scan_sub;
		ros::Subscriber image_sub;
		ros::Subscriber velocity_sub;
		ros::Publisher reward_pub;

		geometry_msgs::Pose gpose;
		naviswarm::CameraImage 	img_data;
	    //cv::Mat img_data; // stores image frames published in /camera/rgb/image_raw/compressed converted to Mat format.
	    //cv::Mat depth_data;
	    naviswarm::Velocity 	odom_data;
	    naviswarm::Scan 		scan_data;

	    cv::Mat cam_data;

		std_msgs::Header img_header;
		std_msgs::Header scan_header;
		std_msgs::Header odom_header;

		int substatus[5] = {0,0,0,0,0};//check if get messages

	public:
	    // Function declarations
	    GazeboTrain(int n){
	    }
	    
	    void gt_Callback(const gazebo_msgs::ModelStates gt);
	    void sync_Callback( const sensor_msgs::ImageConstPtr& image,
							const sensor_msgs::LaserScanConstPtr& scan);
	    void image_Callback(const sensor_msgs::ImageConstPtr& image);
	    void scan_Callback(const sensor_msgs::LaserScanConstPtr& scan);
	    void bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i);
	    bool cb_update_srv(naviswarm::UpdateModelRequest& request, naviswarm::UpdateModelResponse& response);
	    void velocity_Callback(const nav_msgs::OdometryConstPtr& odom);

	    void camera_Callback(const sensor_msgs::ImageConstPtr& img_msg) {
		  cv_bridge::CvImagePtr cvPtr;
		  try {
		    cvPtr = cv_bridge::toCvCopy(img_msg, "32FC1");
		  } catch (cv_bridge::Exception& e) {
		    ROS_ERROR("cv_bridge exception: %s", e.what());
		    return;
		  }
		  cam_data = cvPtr->image;
		  cam_data.setTo(5, cam_data!=cam_data );  
		  double min, max;
		  cv::minMaxLoc(cam_data, &min, &max);
		  

		  std::cout<<"min:"<<min<<",  max:"<<max<<std::endl;
		  //std::cout<<cam_data<<std::endl;
		  //img_data=cam_data
		  //std::cout<<image_data<<std::endl;
		  //ROS_INFO("=================================================");
		}
	    
	    void subscribecamera(){
	    	image_sub		= nh.subscribe<sensor_msgs::Image>("turtlebot0/camera/depth/image_raw", 1, &GazeboTrain::camera_Callback, this);
	    	ros::spin();
	    }

	    void runvelocity(){
	    	naviswarm::Actions velocity;
		  	naviswarm::Action v1;
		  	v1.vx = 5;
		  	velocity.data.push_back(v1);
		  	setvelocities(0,velocity.data[0]);
	    }


////////////////////////////////////////////////////////////------------------------------------------------------------------------------
	    void subscribedata(){

			for(int i = 0; i < num_robots; i++){
				current_robot = i;

				std::string name_space = "/turtlebot" + std::to_string(i);

				/* synchronizer is slow
				message_filters::Subscriber<sensor_msgs::Image> 	image_sub(nh, name_space + "/camera/image_raw", 1);
				//message_filters::Subscriber<sensor_msgs::Image> 	image_sub(nh, name_space + "/camera/depth/image_raw", 1);
				message_filters::Subscriber<sensor_msgs::LaserScan> scan_sub(nh, name_space + "/scan", 1);

				//according to situation, choose synchronize odometry or not
				TimeSynchronizer<sensor_msgs::Image,sensor_msgs::LaserScan> sync(image_sub,scan_sub, 1);
				sync.registerCallback(boost::bind(& GazeboTrain::sync_Callback,this, _1, _2));
				*/

				image_sub		= nh.subscribe<sensor_msgs::Image>(name_space + "/camera/image_raw", 1, &GazeboTrain::image_Callback, this); //"/camera/depth/image_raw"
				// scan_sub		= nh.subscribe<sensor_msgs::LaserScan>(name_space + "/scan", 1, &GazeboTrain::scan_Callback, this);
				// velocity_sub	= nh.subscribe<nav_msgs::Odometry>(name_space + "/odom", 1, &GazeboTrain::velocity_Callback, this);
				// groundtruth_sub = nh.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1, &GazeboTrain::gt_Callback, this);
				// bumper_sub 		= nh.subscribe<kobuki_msgs::BumperEvent>(name_space + "/mobile_base/events/bumper", 1, boost::bind(&GazeboTrain::bumper_Callback, this, _1, i));

				// int checkstatus[5] = {0,0,0,0,0};
				// for(int i=0;i<5;i++){checkstatus[i]=1;}
				// checkstatus[3] = 0;
				// while(substatus!=checkstatus){
				// 	ros::spinOnce();
				// }
				// for(int i=0;i<5;i++){substatus[i]=0;}
				// ROS_INFO("out of while");
				ros::spin();
			}
	    }

};

void GazeboTrain::sync_Callback(const sensor_msgs::ImageConstPtr& image,
								const sensor_msgs::LaserScanConstPtr& scan)
{
	int robotindex = current_robot;
	substatus[0] = 1;

	img_data.data 	 = *image;
	scan_data.ranges = scan->ranges;

	img_header   = image->header;
	scan_header	 = scan->header;

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

void GazeboTrain::image_Callback(const sensor_msgs::ImageConstPtr& image){
	substatus[0] =1;
	img_data.data 	 = *image;
	img_header   = image->header;
	std::cout<<img_header.stamp<<std::endl;


}

void GazeboTrain::scan_Callback(const sensor_msgs::LaserScanConstPtr& scan){
	substatus[1] =1;
	scan_data.ranges = scan->ranges;
	scan_header	 = scan->header;

	float min_range = 0.5;
    collision_status[current_robot] = false; // NOTE: collision status for robot 0 is stored in collision_status[0].
    for (int j = 0; j < scan->ranges.size(); j++) {
        if (scan->ranges[j] < min_range) {
            collision_status[current_robot] = true;  // true indicates presence of obstacle
        }
    }
}

void GazeboTrain::velocity_Callback(const nav_msgs::OdometryConstPtr& odom){
	substatus[2] =1;
	odom_data.vx = odom->twist.twist.linear.x;
  	odom_data.vz = odom->twist.twist.angular.z;
  	odom_header  = odom->header;
}

// Bumper CallBack
void GazeboTrain::bumper_Callback(const kobuki_msgs::BumperEventConstPtr& bumper_msg, int i) {
	substatus[3] =0;
  // ROS_INFO("bumper hit. value = [%d] for robot %d", bumper_msg->bumper, i);
  if (bumper_msg->bumper == 1)
    collision_status[i] = true;
}


// Ground Truth callback
void GazeboTrain::gt_Callback(const gazebo_msgs::ModelStates gt) {
		substatus[4] =1;
  // ROS_INFO("Inside GT CallBack and current_robot is %d", current_robot);
  for (int i = 0; i < gt.name.size(); i++){
    if(gt.name[i].substr(0,2) == "tb" && gt.name[i].compare(2, 1, std::to_string(current_robot)) == 0) {
      gpose = gt.pose[i];
    }
  }
}

int main(int argc, char **argv){
  ros::init(argc, argv, "testcpp");
  GazeboTrain train(2);

  //train.runvelocity();
  
  train.runvelocity();
  train.subscribecamera();
  //train.subscribedata();
  return 0;
}