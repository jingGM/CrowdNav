environment: naviswarm_depth_v2
	4 robots
	empty environment
	depth camera

status:
 depth only training for pedestrians is not enough, may need to train for a longer time
 rgb is in training for static
 depth+lidar is in training for pedestrians, also in static environment.





double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.1);
penalty_for_time = (current_steps+1) *(-0.1);


environment: naviswarm_depth_v4
	NN with fixed parameter
	4 robots
	empty environment
	depth camera

double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.01);
penalty_for_time = (current_steps+1) *(0);



environment naviswarm_rdb_v2
	NN with fixed parameters
	4 robots
	empty environment
	rgb image

double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.01);
penalty_for_time = (current_steps+1) *(0);



environment naviswarm_onlydepth_v1
	NN with simple CNN
	4 robots
	empty environment
	rgb image

double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.01);
penalty_for_time = (current_steps+1) *(0);



environment naviswarm_onlydepth_v2
	NN with fixed parameters
	4 robots
	empty environment
	rgb image

double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.01);
penalty_for_time = (current_steps+1) *(0);

