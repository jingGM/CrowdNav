/* Copyright 2018 The DRLCA Authors.  All rights reserved.

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
    In drl stage mode,
    robot make decisions with odoms and scans,
    collect information from stage simulator, and
    send/receive state and action through shared memory.
    
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <sstream>
// #include <random>

// libstage
#include <stage.hh>

// roscpp
#include <ros/ros.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <rosgraph_msgs/Clock.h>
#include <std_srvs/Empty.h>
#include <tf/transform_broadcaster.h>
#include <boost/tokenizer.hpp>

#include <stage_ros/Velocity.h>
#include <stage_ros/Velocities.h>
#include <stage_ros/Action.h>
#include <stage_ros/Actions.h>
#include <stage_ros/Goal.h>
#include <stage_ros/Scan.h>
#include <stage_ros/State.h>
#include <stage_ros/States.h>
#include <stage_ros/Transition.h>
#include <stage_ros/Transitions.h>
#include <stage_ros/RobotStatus.h>
#include <stage_ros/UpdateStage.h>
#include <stage_ros/Reward.h>

// for the ipc part
#include <sys/ipc.h>		/* for system's IPC_xxx definitions */
#include <sys/shm.h>		/* for shmget, shmat, shmdt, shmctl */
#include <sys/sem.h>		/* for semget, semctl, semop */
#include <errno.h> 
#include <semaphore.h>

#define USAGE "stageros <worldfile>"
#define IMAGE "image"
#define DEPTH "depth"
#define CAMERA_INFO "camera_info"
#define ODOM "odom"
#define BASE_SCAN "base_scan"
#define BASE_POSE_GROUND_TRUTH "base_pose_ground_truth"
#define CMD_VEL "cmd_vel"
#define REWARD "reward"
#define KEY 42
#define SIZE 2048000000
#define PERMISSION 0600
#define TRANSITION_KEY 43

#define ROBOT_RADIUS 0.12
#define MIN_DIST_BETWEEN_AGENTS (ROBOT_RADIUS+0.05)*2
#define OBSTACLE_NUM 0
//#define MIN_DIST_BETWEEN_AGENTS ROBOT_RADIUS*2

#include <unistd.h>



class StageNode
{
private:
    // roscpp-related bookkeeping
    ros::NodeHandle n_;
    
    // A mutex to lock access to fields that are used in message callbacks
    boost::mutex msg_lock;

    // for sharing memory
    int share_id, sem_id;
    uint8_t *share_addr;

    // The models that we're interested in
    int robot_num;
    std::vector<Stg::ModelRanger *> lasermodels;
    std::vector<Stg::ModelPosition *> positionmodels;

    //a structure representing a robot in the simulator
    struct StageRobot
    {
        //stage related models
        Stg::ModelPosition* positionmodel; //one position
        std::vector<Stg::ModelRanger *> lasermodels; //multiple rangers per position

        //ros publishers
        ros::Publisher odom_pub; //one odom
        ros::Publisher ground_truth_pub; //one ground truth
        std::vector<ros::Publisher> laser_pubs; //multiple lasers
        ros::Subscriber cmdvel_sub; //one cmd_vel subscriber
        ros::Publisher reward_pub;
    };
    

    typedef struct {
        double x, y, theta;
    } vec3;

    typedef struct {
        double x, y;
    } vec2;

    double GetDistance(double x, double y) {
        return sqrt(x * x + y * y);
    }

    vec2 getTransformedPoint(const vec2 &vec, const tf::Transform &gt) {
        tf::Vector3 tf_vec, new_tf_vec;
        tf_vec.setX(vec.x);
        tf_vec.setY(vec.y);
        tf_vec.setZ(0.0);

        // std::cout << "x: " << vec.x << " ---  y: " << vec.y << std::endl; 

        new_tf_vec = gt * tf_vec;
        vec2 new_vec;
        new_vec.x = new_tf_vec.getX();
        new_vec.y = new_tf_vec.getY();
        return new_vec;
    }

    tf::Transform pose2transform(const Stg::Pose& pose){
        tf::Quaternion q_pose;
        q_pose.setRPY(0.0, 0.0, pose.a);
        tf::Transform t(q_pose, tf::Point(pose.x, pose.y, 0.0));

        return t;
    }

    double normal_pdf(double x, double mean, double std){
        static const double inv_sqrt_2pi = 0.3989422804014327;
        double a = (x - mean) / std;

        return inv_sqrt_2pi / std * std::exp(-0.5*a*a);
    }
    
    std::vector<StageRobot const *> robotmodels_;
    
    // Used to remember initial poses for soft reset
    // (i.e. set robots to the same initial poses)
    std::vector<Stg::Pose> initial_poses;
    ros::ServiceServer reset_srv_;

    // update robots in Stage to new initial poses and
    // set new goals for each robot
    ros::ServiceServer update_srv_;
    
    ros::Publisher clock_pub_;
    ros::Publisher status_pub_;

    bool use_model_names;
    
    // A helper function that is executed for each stage model.  We use it
    // to search for models of interest.
    static void ghfunc(Stg::Model* mod, StageNode* node);
    
    static bool s_update(Stg::World* world, StageNode* node)
    {
        node->WorldCallback();
        // We return false to indicate that we want to be called again (an
        // odd convention, but that's the way that Stage works).
        return false;
    }
    
    // Appends the given robot ID to the given message name.  If omitRobotID
    // is true, an unaltered copy of the name is returned.
    const char *mapName(const char *name, size_t robotID, Stg::Model* mod) const;
    const char *mapName(const char *name, size_t robotID, size_t deviceID, Stg::Model* mod) const;
    
    // for semaphore operation
    void acquire_semaphore();
    void release_semaphore();


    int num_episode;
    std::vector<vec2> current_goal;
    stage_ros::States last_states;
    stage_ros::Actions executed_actions;
    std::vector<double> path_length;
    std::vector<double> time_elapsed;
    
    tf::TransformBroadcaster tf;
    
    // Last time that we received a velocity command
    ros::Time base_last_cmd;
    ros::Duration base_watchdog_timeout;
    
    // Current simulation time
    ros::Time sim_time;
    
    // Last time we saved global position (for velocity calculation).
    ros::Time base_last_globalpos_time;
    // Last published global pose of each robot
    std::vector<Stg::Pose> base_last_globalpos;
    
public:
    // Constructor; stage itself needs argc/argv.
    // fname is the .world file that stage should load.
    StageNode(int argc, char** argv, bool gui, const char* fname, bool use_model_names);
    ~StageNode();

    void FreeRobots();
    
    void ReleaseSharedMemory();
    
    // Subscribe to models of interest.  Currently, we find and subscribe
    // to the first 'laser' model and the first 'position' model.  Returns
    // 0 on success (both models subscribed), -1 otherwise.
    int SubscribeModels();
    
    // Our callback
    void WorldCallback();
    
    // Do one update of the world.  May pause if the next update time
    // has not yet arrived.
    bool UpdateWorld();
    
    // Message callback for a MsgBaseVel message, which set velocities.
    void cmdvelReceived(int idx, const boost::shared_ptr<geometry_msgs::Twist const>& msg);
    
    // Service callback for soft reset
    bool cb_reset_srv(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response);

    // Service callback for update to new start/goal
    bool cb_update_srv(stage_ros::UpdateStageRequest& request, stage_ros::UpdateStageResponse& response);
    //bool cb_start_srv(stage_ros::StartNewRound::Request& request, stage_ros::StartNewRound::Response& response);
    
    // The main simulator object
    Stg::World* world;
};

void StageNode::acquire_semaphore()
{
    struct sembuf op[1];
    op[0].sem_num = 0;
    op[0].sem_flg = 0;
    op[0].sem_op = -1; // p operation

    if (-1 == semop(sem_id, op, 1))// block until the operation finished
    {
        ROS_WARN("Watch out! Acquire semaphore failed.");
    }
}

void StageNode::release_semaphore()
{
    struct sembuf op[1];
    op[0].sem_num = 0;
    op[0].sem_flg = 0;
    op[0].sem_op = 1; // v operation

    if (-1 == semop(sem_id, op, 1))
    {
        ROS_WARN("Watch out! Release semaphore failed.");
    }
}

// since stageros is single-threaded, this is OK. revisit if that changes!
const char *
StageNode::mapName(const char *name, size_t robotID, Stg::Model* mod) const
{
    //ROS_INFO("Robot %lu: Device %s", robotID, name);
    bool umn = this->use_model_names;
    
    if ((positionmodels.size() > 1) || umn)
    {
        static char buf[100];
        std::size_t found = std::string(((Stg::Ancestor *) mod)->Token()).find(":");
	
        if ((found==std::string::npos) && umn){
            snprintf(buf, sizeof(buf), "/%s/%s", ((Stg::Ancestor *) mod)->Token(), name);
        }
        else{
            snprintf(buf, sizeof(buf), "/robot_%u/%s", (unsigned int)robotID, name);
        }
        return buf;
    }
    else
        return name;
}

const char *
StageNode::mapName(const char *name, size_t robotID, size_t deviceID, Stg::Model* mod) const
{
    //ROS_INFO("Robot %lu: Device %s:%lu", robotID, name, deviceID);
    bool umn = this->use_model_names;
    
    if ((positionmodels.size() > 1 ) || umn)
    {
        static char buf[100];
        std::size_t found = std::string(((Stg::Ancestor *) mod)->Token()).find(":");

        if ((found==std::string::npos) && umn){
            snprintf(buf, sizeof(buf), "/%s/%s_%u", ((Stg::Ancestor *) mod)->Token(), name, (unsigned int)deviceID);
        }
        else{
            snprintf(buf, sizeof(buf), "/robot_%u/%s_%u", (unsigned int)robotID, name, (unsigned int)deviceID);
        }
	
        return buf;
    }
    else
    {
        static char buf[100];
        snprintf(buf, sizeof(buf), "/%s_%u", name, (unsigned int)deviceID);
        return buf;
    }
}

void
StageNode::ghfunc(Stg::Model* mod, StageNode* node)
{
    if (dynamic_cast<Stg::ModelRanger *>(mod))
        node->lasermodels.push_back(dynamic_cast<Stg::ModelRanger *>(mod));

    if (dynamic_cast<Stg::ModelPosition *>(mod)) {
        Stg::ModelPosition * p = dynamic_cast<Stg::ModelPosition *>(mod);
        // remember initial poses
        node->positionmodels.push_back(p);
        node->initial_poses.push_back(p->GetGlobalPose());
    }
}

bool
StageNode::cb_reset_srv(std_srvs::Empty::Request& request, std_srvs::Empty::Response& response)
{
    ROS_INFO("Resetting stage!");
    for (size_t r = 0; r < this->positionmodels.size(); r++) {
        this->positionmodels[r]->SetPose(this->initial_poses[r]);
        this->positionmodels[r]->SetSpeed(0., 0., 0.);
        this->positionmodels[r]->SetStall(false);
    }
    return true;
}

bool 
StageNode::cb_update_srv(stage_ros::UpdateStageRequest& request, stage_ros::UpdateStageResponse& response)
{
    ROS_INFO("Updatting stage!");

    std::cout << "request poses size: " << request.poses.size() << std::endl;
    std::cout << "position models size: " << this->positionmodels.size() << std::endl;

    if (request.poses.size() != this->positionmodels.size()) {
        ROS_WARN("Robot Number Error");
        response.success = false;
    }
    else {
        for (size_t r = 0; r < this->positionmodels.size(); r++) {
            tf::Quaternion temp_quad;
            tf::quaternionMsgToTF(request.poses[r].orientation, temp_quad);
            Stg::Pose temp(request.poses[r].position.x, request.poses[r].position.y, request.poses[r].position.z, tf::getYaw(temp_quad));

            this->positionmodels[r]->SetPose(temp);
            this->positionmodels[r]->SetSpeed(0., 0., 0.);
            this->positionmodels[r]->SetStall(false);
        }

        for (size_t r = 0; r < this->positionmodels.size(); r++) {
            vec2 goal;
            goal.x = request.points[r].x;
            goal.y = request.points[r].y;
            current_goal[r] = goal;
            ROS_INFO("Goal_%d: %.3f, %.3f", int(r), goal.x, goal.y);
        }

        // WARMING: IT MAY NOT FREE THE MEMORY SPACE
        last_states.actionObsBatch.clear();
        last_states.goalObsBatch.clear();
        last_states.scanObsBatch.clear();
        last_states.velObsBatch.clear();
        executed_actions.data.clear();
        // WARMING: IT MAY NOT FREE THE MEMORY SPACE

        response.success = true;
    }

    //ROS_INFO("update stage response: %d", response.success);
    return true;
}

void
StageNode::cmdvelReceived(int idx, const boost::shared_ptr<geometry_msgs::Twist const>& msg)
{
    boost::mutex::scoped_lock lock(msg_lock);
    this->positionmodels[idx]->SetSpeed(msg->linear.x, msg->linear.y, msg->angular.z);
    this->base_last_cmd = this->sim_time;
}


StageNode::StageNode(int argc, char** argv, bool gui, const char* fname, bool use_model_names)
{
    this->use_model_names = use_model_names;
    this->sim_time.fromSec(0.0);
    this->base_last_cmd.fromSec(0.0);

    double t;
    ros::NodeHandle localn("~");
    if(!localn.getParam("base_watchdog_timeout", t))
        t = 0.2;
    this->base_watchdog_timeout.fromSec(t);
    
    // We'll check the existence of the world file, because libstage doesn't
    // expose its failure to open it.  Could go further with checks (e.g., is
    // it readable by this user).
    struct stat s;
    if(stat(fname, &s) != 0)
    {
        ROS_FATAL("The world file %s does not exist.", fname);
        ROS_BREAK();
    }
    
    // initialize libstage
    Stg::Init( &argc, &argv );
    
    if(gui)
        this->world = new Stg::WorldGui(600, 400, "Stage (ROS)");
    else
        this->world = new Stg::World();
    
    // Apparently an Update is needed before the Load to avoid crashes on
    // startup on some systems.
    // As of Stage 4.1.1, this update call causes a hang on start.
    //this->UpdateWorld();
    this->world->Load(fname);
    
    // We add our callback here, after the Update, so avoid our callback
    // being invoked before we're ready.
    this->world->AddUpdateCallback((Stg::world_callback_t)s_update, this);
    
    this->world->ForEachDescendant((Stg::model_callback_t)ghfunc, this);
}


// Subscribe to models of interest.  Currently, we find and subscribe
// to the first 'laser' model and the first 'position' model.  Returns
// 0 on success (both models subscribed), -1 otherwise.
//
// Eventually, we should provide a general way to map stage models onto ROS
// topics, similar to Player .cfg files.
int
StageNode::SubscribeModels()
{
    n_.setParam("/use_sim_time", true);
    
    for (size_t r = 0; r < this->positionmodels.size() - OBSTACLE_NUM; r++){
        StageRobot* new_robot = new StageRobot;
        new_robot->positionmodel = this->positionmodels[r];
        new_robot->positionmodel->Subscribe();
	
        for (size_t s = 0; s < this->lasermodels.size(); s++){
            if (this->lasermodels[s] and this->lasermodels[s]->Parent() == new_robot->positionmodel){
                new_robot->lasermodels.push_back(this->lasermodels[s]);
                this->lasermodels[s]->Subscribe();
            }
        }
	
        ROS_INFO("Found %lu laser devices and %lu ", new_robot->lasermodels.size(), r);
	
        new_robot->odom_pub = n_.advertise<nav_msgs::Odometry>(mapName(ODOM, r, static_cast<Stg::Model*>(new_robot->positionmodel)), 10);
        new_robot->ground_truth_pub = n_.advertise<nav_msgs::Odometry>(mapName(BASE_POSE_GROUND_TRUTH, r, static_cast<Stg::Model*>(new_robot->positionmodel)), 10);
        new_robot->cmdvel_sub = n_.subscribe<geometry_msgs::Twist>(mapName(CMD_VEL, r, static_cast<Stg::Model*>(new_robot->positionmodel)), 10, boost::bind(&StageNode::cmdvelReceived, this, r, _1));
        new_robot->reward_pub = n_.advertise<stage_ros::Reward>(mapName(REWARD, r, static_cast<Stg::Model*>(new_robot->positionmodel)), 10);
	
        for (size_t s = 0;  s < new_robot->lasermodels.size(); ++s){
            if (new_robot->lasermodels.size() == 1)
                new_robot->laser_pubs.push_back(n_.advertise<sensor_msgs::LaserScan>(mapName(BASE_SCAN, r, static_cast<Stg::Model*>(new_robot->positionmodel)), 10));
            else
                new_robot->laser_pubs.push_back(n_.advertise<sensor_msgs::LaserScan>(mapName(BASE_SCAN, r, s, static_cast<Stg::Model*>(new_robot->positionmodel)), 10));
        }


        this->robotmodels_.push_back(new_robot);
    }

    robot_num = robotmodels_.size();
    std::cout << "***** ROBOT_NUM: " << robot_num << " ******" << std::endl;
    for (int i = 0; i < robot_num; ++i)
    {
        vec2 temp_goal;
        temp_goal.x = 0.;
        temp_goal.y = 0.;
        current_goal.push_back(temp_goal);
    }

    path_length.resize(robot_num, 0.0);
    time_elapsed.resize(robot_num, 0.0);
    
    clock_pub_ = n_.advertise<rosgraph_msgs::Clock>("/clock", 10);
    status_pub_ = n_.advertise<stage_ros::RobotStatus>("/robot_status", 10);
    
    // advertising reset service
    reset_srv_ = n_.advertiseService("reset_positions", &StageNode::cb_reset_srv, this);
    update_srv_ = n_.advertiseService("update_positions", &StageNode::cb_update_srv, this);
    
    // create shared memory
    //this->ReleaseSharedMemory();
    std::cout << "i am here" << std::endl;
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


void StageNode::FreeRobots()
{
    for (std::vector<StageRobot const*>::iterator r = this->robotmodels_.begin(); r != this->robotmodels_.end(); ++r)
    {
        delete *r;
    }
}

StageNode::~StageNode()
{    
    this->FreeRobots();

    this->ReleaseSharedMemory();
}

bool
StageNode::UpdateWorld()
{
    return this->world->UpdateAll();
}

void StageNode::ReleaseSharedMemory()
{
    // this does NOT work, find a way to remove the shared memory
    struct shmid_ds shm_info;
    shmctl(share_id, IPC_RMID, NULL);
    semctl(sem_id, 0, IPC_RMID);
}

void
StageNode::WorldCallback()
{
    boost::mutex::scoped_lock lock(msg_lock);
    
    this->sim_time.fromSec(world->SimTimeNow() / 1e6);
    // We're not allowed to publish clock==0, because it used as a special
    // value in parts of ROS, #4027.
    if(this->sim_time.sec == 0 && this->sim_time.nsec == 0){
        ROS_DEBUG("Skipping initial simulation step, to avoid publishing clock==0");
        return;
    }
    
    // TODO make this only affect one robot if necessary
    if((this->base_watchdog_timeout.toSec() > 0.0) &&
        ((this->sim_time - this->base_last_cmd) >= this->base_watchdog_timeout)){
        for (size_t r = 0; r < this->positionmodels.size(); r++)
            this->positionmodels[r]->SetSpeed(0.0, 0.0, 0.0);
    }
    
    stage_ros::RobotStatus rs;
    rs.stalled.resize(this->robotmodels_.size());

    stage_ros::States current_states;
    stage_ros::Transitions current_transitions;

    // std::cout << "CallBack!" << std::endl;

    //loop on the robot models
    for (size_t r = 0; r < this->robotmodels_.size(); ++r)
    {
        StageRobot const * robotmodel = this->robotmodels_[r];
        
		rs.stalled[r] = robotmodel->positionmodel->Stalled();

        //loop on the laser devices for the current robot
        for (size_t s = 0; s < robotmodel->lasermodels.size(); ++s){
            Stg::ModelRanger const* lasermodel = robotmodel->lasermodels[s];
            const std::vector<Stg::ModelRanger::Sensor>& sensors = lasermodel->GetSensors();

            if(sensors.size() > 1)
                ROS_WARN( "ROS Stage currently supports rangers with 1 sensor only." );

            // for now we access only the zeroth sensor of the ranger - good
            // enough for most laser models that have a single beam origin
            const Stg::ModelRanger::Sensor& sensor = sensors[0];

            if(sensor.ranges.size())
            {
                // Translate into ROS message format and publish
                sensor_msgs::LaserScan msg;
                msg.angle_min = -sensor.fov/2.0;
                msg.angle_max = +sensor.fov/2.0;
                msg.angle_increment = sensor.fov/(double)(sensor.sample_count-1);
                msg.range_min = sensor.range.min;
                msg.range_max = sensor.range.max;
                msg.ranges.resize(sensor.ranges.size());
                msg.intensities.resize(sensor.intensities.size());

                for(unsigned int i = 0; i < sensor.ranges.size(); i++)
                {
                    msg.ranges[i] = sensor.ranges[i];
                    msg.intensities[i] = sensor.intensities[i];
                }

                if (robotmodel->lasermodels.size() > 1)
                    msg.header.frame_id = mapName("base_laser_link", r, s, static_cast<Stg::Model*>(robotmodel->positionmodel));
                else
                    msg.header.frame_id = mapName("base_laser_link", r, static_cast<Stg::Model*>(robotmodel->positionmodel));

                msg.header.stamp = sim_time;
                robotmodel->laser_pubs[s].publish(msg);
            }

            // Also publish the base->base_laser_link Tx.  This could eventually move
            // into being retrieved from the param server as a static Tx.
            Stg::Pose lp = lasermodel->GetPose();
            tf::Quaternion laserQ;
            laserQ.setRPY(0.0, 0.0, lp.a);
            tf::Transform txLaser =  tf::Transform(laserQ, tf::Point(lp.x, lp.y, robotmodel->positionmodel->GetGeom().size.z + lp.z));

            if (robotmodel->lasermodels.size() > 1)
                tf.sendTransform(tf::StampedTransform(txLaser, sim_time,
                                  mapName("base_link", r, static_cast<Stg::Model*>(robotmodel->positionmodel)),
                                  mapName("base_laser_link", r, s, static_cast<Stg::Model*>(robotmodel->positionmodel))));
            else
                tf.sendTransform(tf::StampedTransform(txLaser, sim_time,
                                  mapName("base_link", r, static_cast<Stg::Model*>(robotmodel->positionmodel)),
                                  mapName("base_laser_link", r, static_cast<Stg::Model*>(robotmodel->positionmodel))));
        }
	
        //the position of the robot
        tf.sendTransform(tf::StampedTransform(tf::Transform::getIdentity(),
                         sim_time,
                         mapName("base_footprint", r, static_cast<Stg::Model*>(robotmodel->positionmodel)),
                         mapName("base_link", r, static_cast<Stg::Model*>(robotmodel->positionmodel))));

        // Get latest odometry data
        // Translate into ROS message format and publish
        nav_msgs::Odometry odom_msg;
        odom_msg.pose.pose.position.x = robotmodel->positionmodel->est_pose.x;
        odom_msg.pose.pose.position.y = robotmodel->positionmodel->est_pose.y;
        odom_msg.pose.pose.orientation = tf::createQuaternionMsgFromYaw(robotmodel->positionmodel->est_pose.a);
        Stg::Velocity v = robotmodel->positionmodel->GetVelocity();
        odom_msg.twist.twist.linear.x = v.x;
        odom_msg.twist.twist.linear.y = v.y;
        odom_msg.twist.twist.angular.z = v.a;

        //@todo Publish stall on a separate topic when one becomes available
        //this->odomMsgs[r].stall = this->positionmodels[r]->Stall();
        odom_msg.header.frame_id = mapName("odom", r, static_cast<Stg::Model*>(robotmodel->positionmodel));
        odom_msg.header.stamp = sim_time;

        robotmodel->odom_pub.publish(odom_msg);

        // broadcast odometry transform
        tf::Quaternion odomQ;
        tf::quaternionMsgToTF(odom_msg.pose.pose.orientation, odomQ);
        tf::Transform txOdom(odomQ, tf::Point(odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, 0.0));

        // don't use the estimation data any more
        //tf.sendTransform(tf::StampedTransform(txOdom, sim_time,
        //                                    mapName("odom", r, static_cast<Stg::Model*>(robotmodel->positionmodel)),
        //                                     mapName("base_footprint", r, static_cast<Stg::Model*>(robotmodel->positionmodel))));

        // Also publish the ground truth pose and velocity
        Stg::Pose gpose = robotmodel->positionmodel->GetGlobalPose();
        tf::Transform gt = pose2transform(gpose);

        // Velocity is 0 by default and will be set only if there is previous pose and time delta > 0
        Stg::Velocity gvel(0., 0., 0., 0.);
        if (this->base_last_globalpos.size() > r){
            Stg::Pose prevpose = this->base_last_globalpos.at(r);
            double dT = (this->sim_time - this->base_last_globalpos_time).toSec();

            if (dT > 0){
                gvel = Stg::Velocity(
                    (gpose.x - prevpose.x)/dT,
                    (gpose.y - prevpose.y)/dT,
                    (gpose.z - prevpose.z)/dT,
                    Stg::normalize(gpose.a - prevpose.a)/dT);
            }

            // calculate the traveled path length, using the straight line model,
            // i.e. try the distance between two poses as a straight line
            path_length[r] += gpose.Distance2D(prevpose);

            this->base_last_globalpos.at(r) = gpose;
        }
        else //There are no previous readings, adding current pose...
            this->base_last_globalpos.push_back(gpose);

        nav_msgs::Odometry ground_truth_msg;
        ground_truth_msg.pose.pose.position.x = gt.getOrigin().x();
        ground_truth_msg.pose.pose.position.y = gt.getOrigin().y();
        ground_truth_msg.pose.pose.position.z = gt.getOrigin().z();
        ground_truth_msg.pose.pose.orientation.x = gt.getRotation().x();
        ground_truth_msg.pose.pose.orientation.y = gt.getRotation().y();
        ground_truth_msg.pose.pose.orientation.z = gt.getRotation().z();
        ground_truth_msg.pose.pose.orientation.w = gt.getRotation().w();
        ground_truth_msg.twist.twist.linear.x = gvel.x;
        ground_truth_msg.twist.twist.linear.y = gvel.y;
        ground_truth_msg.twist.twist.linear.z = gvel.z;
        ground_truth_msg.twist.twist.angular.z = gvel.a;

        // send transform to a ground truth frame
        tf.sendTransform(tf::StampedTransform(gt, sim_time, "ground_truth", mapName("base_footprint", r, static_cast<Stg::Model*>(robotmodel->positionmodel))));

        ground_truth_msg.header.frame_id = "ground_truth";
        //ground_truth_msg.header.frame_id = mapName("odom", r, static_cast<Stg::Model*>(robotmodel->positionmodel));
        ground_truth_msg.header.stamp = sim_time;

        robotmodel->ground_truth_pub.publish(ground_truth_msg);

        /******************************
         * 1. get observation
         * 2. execute action based on the observation
         * 3. get reward
         * 4. get new observation
         * 5. observation = new observation
         * 6. back to 2
         ******************************/

        // 1. get observation
        // get goal in the robot's local frame, local goal = [distance_to_goal, angular_between_goal]

        // contruct the transition
        stage_ros::Transition current_transition;

        current_transition.pose.push_back(gpose.x);
        current_transition.pose.push_back(gpose.y);
        current_transition.pose.push_back(gpose.a);

        stage_ros::State state;
        vec2 local_goal = getTransformedPoint(current_goal[r], gt.inverse());

        state.goalObs.goal_now.goal_dist = GetDistance(local_goal.x, local_goal.y);
        state.goalObs.goal_now.goal_theta = atan2(local_goal.y, local_goal.x);

        state.velObs.vel_now.vx = v.x;
        state.velObs.vel_now.vz = v.a;

        if (last_states.goalObsBatch.size() == 0) {
            state.goalObs.goal_pprev = state.goalObs.goal_now;
            state.goalObs.goal_prev = state.goalObs.goal_now;
        }
        else {
            state.goalObs.goal_pprev = last_states.goalObsBatch[r].goal_prev;
            state.goalObs.goal_prev = last_states.goalObsBatch[r].goal_now;
        }

        state.scanObs.scan_now.ranges = robotmodel->lasermodels[0]->GetSensors()[0].ranges;
        if (last_states.goalObsBatch.size() == 0) {
            //robotmodel->lasermodels[0]->GetSensors()[0].ranges;
            state.scanObs.scan_pprev = state.scanObs.scan_now;
            state.scanObs.scan_prev = state.scanObs.scan_now;
        }
        else {
            state.scanObs.scan_pprev = last_states.scanObsBatch[r].scan_prev;
            state.scanObs.scan_prev = last_states.scanObsBatch[r].scan_now;
        }

        if (last_states.goalObsBatch.size() == 0) {
            state.actionObs.ac_pprev.vx = 0.0;
            state.actionObs.ac_pprev.vz = 0.;
            state.actionObs.ac_prev.vx = 0.;
            state.actionObs.ac_prev.vz = 0.;
        }
        else {
            // should set last_states.data[r].acobs.ac_prev = actions.data[i] (this is ac_now)
            state.actionObs.ac_pprev = last_states.actionObsBatch[r].ac_pprev;
            state.actionObs.ac_prev = last_states.actionObsBatch[r].ac_prev;
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
            stage_ros::Reward reward;
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
            else
            {

                // rs.stalled[r] = collision;
                if(rs.stalled[r]) {
                    current_transition.terminal = true;
                    current_transition.reward = -20.0;
                    reward.collision = -20.0;
                }
                else {
                    current_transition.terminal = false;

                    double reward_approaching_goal = 2.5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);

                    double penalty_for_bigvz = 0.0;
                    if (std::abs(executed_actions.data[r].vz) > 0.7)
                    {
                        penalty_for_bigvz = -0.05*std::abs(executed_actions.data[r].vz);
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
           robotmodel->reward_pub.publish(reward);

        }

        current_transitions.data.push_back(current_transition);
        current_states.scanObsBatch.push_back(state.scanObs);
        current_states.goalObsBatch.push_back(state.goalObs);
        current_states.actionObsBatch.push_back(state.actionObs);
        current_states.velObsBatch.push_back(state.velObs);
    }

    last_states = current_states;
    //transition_collection.frame.push_back(current_transitions);
    
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
        if (new_length != length)
            succ = true;

        if (succ)
        {
            stage_ros::Actions actions;
            ros::serialization::IStream stream((share_addr + 4), new_length);
            ros::serialization::Serializer<stage_ros::Actions>::read(stream, actions);
            release_semaphore();

            if (actions.data.size() != this->robotmodels_.size()){
                ROS_INFO("actions_size != robots_size, actions_size is %d", static_cast<int>(actions.data.size()));
                ROS_BREAK();
            }
            //for(size_t r = 0; r < this->robotmodels_.size(); ++r)
            for (int i = 0 ; i < actions.data.size(); ++i){
                this->positionmodels[i]->SetSpeed(actions.data[i].vx, 0., actions.data[i].vz);
                last_states.actionObsBatch[i].ac_pprev = last_states.actionObsBatch[i].ac_prev;
                last_states.actionObsBatch[i].ac_prev = actions.data[i];
            }

            executed_actions = actions;
            break;
        }
        else
        {
            release_semaphore();
        }
    }

    
    this->status_pub_.publish(rs);
    this->base_last_globalpos_time = this->sim_time;
    rosgraph_msgs::Clock clock_msg;
    clock_msg.clock = sim_time;
    this->clock_pub_.publish(clock_msg);
}

int main(int argc, char** argv)
{ 
    if( argc < 2 )
    {
        puts(USAGE);
        exit(-1);
    }
    
    ros::init(argc, argv, "drl_stageros");
    
    bool gui = true;
    bool use_model_names = false;

    for(int i=0;i<(argc-1);i++)
    {
        if(!strcmp(argv[i], "-g"))
            gui = false;
        if(!strcmp(argv[i], "-u"))
            use_model_names = true;
    }
    
    StageNode sn(argc-1,argv,gui,argv[argc-1], use_model_names);
    
    if(sn.SubscribeModels() != 0)
        exit(-1);
    
    boost::thread t = boost::thread(boost::bind(&ros::spin));
    
    // New in Stage 4.1.1: must Start() the world.
    sn.world->Start();

    // PauseUntilNextUpdate() functionality.
    ros::WallRate r(10.0);
    while(ros::ok() && !sn.world->TestQuit())
    {
        if(gui)
            Fl::wait(r.expectedCycleTime().toSec());
        else
        {
            sn.UpdateWorld();
            r.sleep();
        }
    }
    t.join();

    exit(0);
}

