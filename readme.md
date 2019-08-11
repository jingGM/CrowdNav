

venvironment: naviswarm_depth_v1
	4 robots
	empty environment
	depth camera

double reched_goal_reward = 20;
double collision_penalty =-20;
reward_approaching_goal = 2*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.01);
penalty_for_time = (current_steps+1) *(-0.002);




environment: naviswarm_depth_v2
	4 robots
	empty environment
	depth camera

double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.1);
penalty_for_time = (current_steps+1) *(-0.1);



environment naviswarm_rdb_v1
	4 robots
	empty environment
	rgb image

double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.01)+std::abs(state.velObs.vel_now.vx) * (0.5);
penalty_for_time = (current_steps+1) *(0);



environment naviswarm_rdb_v2
	4 robots
	empty environment
	rgb image

double reched_goal_reward = 40;
double collision_penalty =-40;
reward_approaching_goal = 5*(state.goalObs.goal_prev.goal_dist - state.goalObs.goal_now.goal_dist);
penalty_for_bigvz = std::abs(state.velObs.vel_now.vz) * (-0.01)+std::abs(state.velObs.vel_now.vx) * (0.1);
penalty_for_time = (current_steps+1) *(0);

