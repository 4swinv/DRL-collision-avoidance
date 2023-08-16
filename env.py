import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
import parameters as params

from model import Vehicle, Road, ODESolver
from apf import APF

DEG_TO_RAD = np.pi/180
RAD_TO_DEG = 180/np.pi
MPS_TO_KMH = 18/5
KMH_TO_MPS = 5/18
# Road params
ROAD_WIDTH = 3.7*2 # meters

class sim_environment(py_environment.PyEnvironment):

    def __init__(self, Is_test_flag = False,
                 test_x_waypoints = None,
                 test_y_waypoints = None,
                 nwp = None,
                 initial_obs_state = None):

        self._action_spec = array_spec.BoundedArraySpec( shape=(), dtype = np.int64, minimum = 0, maximum = 2, name = 'action' )
        self._observation_spec = array_spec.BoundedArraySpec(shape=(5,), dtype = np.float32, minimum = [- 3 * ROAD_WIDTH / 4, -np.pi, 0, 0, -1], 
                                                             maximum = [3 * ROAD_WIDTH / 4, np.pi, 200, 1, 1], name='observation')
        # [ side_track_error, course_angle_err, along_track_error, potential, r ]

        self.episode_ended = False
        self.counter = 0
        self.distanceToGoal = None

        # Define start and end points, set velocity
        self.x_init = 0
        self.y_init = 3 * ROAD_WIDTH/4
        self.x_goal = 200 # 200 m goal
        self.y_goal = 3 * ROAD_WIDTH/4
        self.setVelocity = 40 * KMH_TO_MPS

        self.Is_test_flag = Is_test_flag

        # Test waypoints
        self.test_x_waypoints = test_x_waypoints
        self.test_y_waypoints = test_y_waypoints
        self.nwp = nwp # number of waypoints
        self.wp_counter = 1 # waypoint counter
        self.initial_obs_state = initial_obs_state # initial state

        # Trajectories of state variables
        # Only stored on test
        self.x_traj = []
        self.y_traj = []
        self.psi_traj = []
        self.u_traj = []
        self.v_traj = []
        self.r_traj = []
        self.theta_traj = []
        self.force_traj = []

        # Trajectories of observation states
        # Only stored on test
        self.along_trk_err_traj = []
        self.course_ang_err_traj =[]
        self.distance_traj = []
        self.potential_traj = []

        # Trajectories of action
        # Only stored in testing
        self.action_traj = []

        # Reward in trajectory
        # Only stored in testing
        self.total_reward = []
        self.reward_01 = []    # Cross track error
        self.reward_02 = []    # Course angle error
        self.reward_03 = []    # Distance to goal
        
        # Make ego vehicle
        self.obs_state = np.zeros(8)
        self.obs_state[0] = 40 * KMH_TO_MPS # u = 40 km/hr
        self.obs_state[4] = 0 # x
        self.obs_state[5] = 3 * ROAD_WIDTH / 4 # y
        self.obs_state[3] = 0 # Heading
        self.resetState = self.obs_state.copy()

        self.tspan = (0, 0.1) # Time interval for 4th order integration. Evaluations at start and end.

        self.ego = Vehicle(x_init = self.obs_state)
        self.ego.name = "ego"
        self.ego.simtype = "control"

        # Make obstacle vehicle - stationary
        x_o1 = np.zeros(8)
        x_o1[0] = 0 * KMH_TO_MPS # u = 0 km/hr
        x_o1[4] = 100
        x_o1[5] = 3 * ROAD_WIDTH/4
        x_o1[3] = 0 # Heading
        self.obs1 = Vehicle(x_init = x_o1, solve = False)
        self.obs1.name = "Obs1"

        self.vehicleSet = [self.ego, self.obs1]
        self.obsSet = [self.obs1]

        # Make Road and Potential
        self.road = Road()
        self.apf = APF()
        self.ep = 0.2

        # Time Counter
        self.counter = 0

        # Action selection
        self.action_set_steering = [ - 15 * DEG_TO_RAD, 0, 15 * DEG_TO_RAD ]
        self.action_set_accel = [ -4 * self.ego.mass, 0, 4 * self.ego.mass]

        self.action_space = []
 
        for i in range(3):
            for j in range(3):
                self.action_space.append((self.action_set_steering[i], self.action_set_accel[j]))

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action_no):

        self.counter += 1

        # Action selection
        # print(action_no)
        theta_c = self.action_set_steering[action_no]
        accel_c = 0 #self.action_space[action_no][1]

        # Set control
        self.ego.uc[0] = theta_c
        self.ego.uc[1] = accel_c

        if self.Is_test_flag == 1:
            self.action_traj.append(np.array([theta_c,accel_c]))

        # Solve dynamics
        ODESolver( self.vehicleSet, t_span = self.tspan )

        # sol.y outputs 0)x vel 1) y vel 2) yaw vel 3) x_coord 4) y_coord 5) psi 6) delta 7) force
        # u,v,r - local coords, psi, x, y - global coords 

        u = self.ego.x_simulated[0][-1]
        v = self.ego.x_simulated[1][-1]
        r = self.ego.x_simulated[2][-1]
        psi_rad = self.ego.x_simulated[3][-1]  # psi
        x = self.ego.x_simulated[4][-1]
        y = self.ego.x_simulated[5][-1]
        psi = psi_rad #% (2 * np.pi)
        theta = self.ego.x_simulated[6][-1]
        force = self.ego.x_simulated[7][-1]

        # Populate obs_state vector

        self.obs_state[0] = u        # x velocity
        self.obs_state[1] = v        # y velocity
        self.obs_state[2] = r        # Yaw velocity
        self.obs_state[3] = psi_rad  # X cooridnate
        self.obs_state[4] = x        # Y coordinate
        self.obs_state[5] = y        # Heading angle
        self.obs_state[6] = theta    # Actual steering angle
        self.obs_state[7] = force    # Force

        #print("State", self.obs_state)
        # Update ego vehicle state
        self.ego.x = self.obs_state.copy()

        # DISTANCE TO GOAL
        self.distanceToGoal =  ((x - self.x_goal) ** 2 + (y - self.y_goal) ** 2) ** 0.5

        # SIDE TRACK & ALONG TRACK ERROR
        vec1 = np.array([self.x_goal - self.x_init, self.y_goal - self.y_init])
        vec2 = np.array([self.x_goal - x, self.y_goal - y])
        vec_a = np.array([x - self.x_init, y - self.y_init])
        vec1n = vec1 / np.linalg.norm(vec1)

        side_track_error = np.cross(vec2, vec1n)
        along_track_error = np.dot(vec_a, vec1n)
    
        # COURSE ANGLE ERROR
        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)
        vec3 = np.array([x_dot, y_dot])
        Unet = np.linalg.norm(vec3) # net velocity
        vec3 = vec3 / Unet
        vec2 = vec2 / np.linalg.norm(vec2)

        angle_btw23 = np.arccos(np.dot(vec2, vec3))
        angle_btw12 = np.arccos(np.dot(vec1n, vec2))
        #angle_btw13 = np.arccos(np.dot(vec1n, vec3))

        # course_angle = np.arctan2(vec3[1], vec3[0])
        # psi_vec2 = np.arctan2(vec2[1], vec2[0])
        # course_angle_err = course_angle - psi_vec2
        # course_angle_err = (course_angle_err + np.pi) % (2 * np.pi) - np.pi

        #course_angle_err = np.arcsin(np.cross(vec2, vec3))
        course_angle_err = -np.arccos(np.dot(vec1n, vec3))

        # COMPUTE POTENTIAL
        potential = self.apf.Obstacle_potential(self.ego, self.obsSet) + self.apf.Road_potential_2way_opp(y)

        # VELOCITY ERROR
        velocity_err = abs(Unet - self.setVelocity) / self.setVelocity

        # REWARD FUNCTIONS
        # R1 =   2 * np.exp(-0.1 * abs(side_track_error) ** 2) - 1 #2 * np.exp(-0.08 * cross_track_error ** 2) - 1 - potential ** (0.5)
        # R2 =   1.3 * np.exp(-10 * (abs(course_angle_err))) - 0.3
        # R3 =   - 2 * np.exp(abs(along_track_error) / self.x_goal) + 3
        # R4 = 0 #- velocity_err ** (0.5)

        # R1 = 1.5 * (- np.exp( abs(side_track_error) / ROAD_WIDTH ) + 1)
        # R2 = 1.5 * (- np.exp(2 * abs(course_angle_err) / np.pi) + 1)
        # al = abs(along_track_error) / self.x_goal # [1,0] [1,inf] 
        # R3 = 15 / np.exp( 4 * al ) 
        # R4 = - 1.5 * potential

        R1 = (- np.exp( side_track_error**2 / ROAD_WIDTH ) + 1)
        R2 = 2 * (- np.exp(2 * abs(course_angle_err) / np.pi) + 1)
        al = abs(along_track_error) / self.x_goal # [1,0] [1,inf] 
        R3 = 15 * al #20 / np.exp( 4 * al ) # 10
        R4 = - 2 * potential

        reward = (R1 + R2 + R3 + R4)

        if self.Is_test_flag:

            self.x_traj.append(x)
            self.y_traj.append(y)
            self.psi_traj.append(psi_rad)
            self.u_traj.append(u)
            self.v_traj.append(v)
            self.r_traj.append(r)
            self.theta_traj.append(theta)
            self.force_traj.append(force)

            # Check
            # x_init = self.test_x_waypoints[self.wp_counter - 1]
            # y_init = self.test_y_waypoints[self.wp_counter - 1]
            # x_goal = self.test_x_waypoints[self.wp_counter]
            # y_goal = self.test_y_waypoints[self.wp_counter]

            self.distance_traj.append(self.distanceToGoal)

            self.along_trk_err_traj.append(along_track_error)

            self.course_ang_err_traj.append(course_angle_err)

            self.potential_traj.append(potential)
            
            self.reward_01.append(R1)
            self.reward_02.append(R2)
            self.reward_03.append(R3)
            self.total_reward.append(reward)

        observation = [ side_track_error, course_angle_err, along_track_error, potential, r*RAD_TO_DEG ]

        #print("Observation", [ side_track_error, course_angle_err*RAD_TO_DEG])
        #print("Psi", psi*RAD_TO_DEG)
        #print("Reward",R1,R2,R3,R4)

        # COLLISION CHECK
        if self.ego.CheckCollision(self.obsSet):
            self.episode_ended = True
            reward = - 200
            print('Collision Detected at', x, y, psi*RAD_TO_DEG, course_angle_err*RAD_TO_DEG)
            #print('Velocity', u, v, "Psi", psi*RAD_TO_DEG)
            print("Reward", R1,R2,R3,R4)
            #print('theta', theta*RAD_TO_DEG)
            return ts.termination( np.array ( observation, dtype=np.float32 ) , reward )

        # DESTINATION CHECK
        if self.distanceToGoal <= 0.5 and abs(course_angle_err) < 5 * DEG_TO_RAD:
            reward = 1500
            self.episode_ended = True
            print('Destination reached', x, y, psi*RAD_TO_DEG, course_angle_err*RAD_TO_DEG)
            return ts.termination( np.array ( observation, dtype=np.float32 ) , reward )
        
        if x >= self.x_goal:
            reward = -200
            self.episode_ended = True
            print('X Goal Reached', x, y, psi*RAD_TO_DEG, course_angle_err*RAD_TO_DEG)
            return ts.termination( np.array ( observation, dtype=np.float32 ) , reward )

        # HEADING CHECK
        if angle_btw12 >= np.pi / 2 or angle_btw23 >= np.pi / 2:
            reward = -200
            self.episode_ended = True
            print('Angle out of bounds', angle_btw12*RAD_TO_DEG, angle_btw12*RAD_TO_DEG, vec3, "\n")
            return ts.termination(np.array(observation, dtype=np.float32), reward)
        
        # Reset ego state-space
        self.ego.x_simulated = None
        
        return ts.transition(np.array(observation, dtype=np.float32), reward)

    def _reset(self):

        # print("Next episode")

        self.counter = 0

        if self.Is_test_flag == False or self.Is_test_flag == True:

            self.obs_state = self.resetState.copy()
            self.ego.x_simulated = None
            self.ego.x = self.resetState.copy()

            u = self.obs_state[0]
            v = self.obs_state[1]
            r = self.obs_state[2]
            psi_rad = self.obs_state[3]  # psi
            x = self.obs_state[4]
            y = self.obs_state[5]
            psi = psi_rad % (2 * np.pi)


            # SIDE TRACK & ALONG TRACK ERROR
            vec1 = np.array([self.x_goal - self.x_init, self.y_goal - self.y_init])
            vec2 = np.array([self.x_goal - x, self.y_goal - y])
            vec_a = np.array([x - self.x_init, y - self.y_init])
            vec1n = vec1 / np.linalg.norm(vec1)

            side_track_error = np.cross(vec2, vec1n)
            along_track_error = np.dot(vec_a, vec1n)
        
            # COURSE ANGLE ERROR
            x_dot = u * np.cos(psi) - v * np.sin(psi)
            y_dot = u * np.sin(psi) + v * np.cos(psi)
            vec3 = np.array([x_dot, y_dot])
            Unet = np.linalg.norm(vec3) # net velocity
            vec3 = vec3 / Unet
            vec2 = vec2 / np.linalg.norm(vec2)

            course_angle = np.arctan2(vec3[1], vec3[0])
            psi_vec2 = np.arctan2(vec2[1], vec2[0])
            course_angle_err = course_angle - psi_vec2
            course_angle_err = (course_angle_err + np.pi) % (2 * np.pi) - np.pi
        
            potential = self.apf.Road_potential_2way_opp(y)
            velocity_err = abs(Unet - self.setVelocity) / self.setVelocity

            self.counter = self.counter + 1
            observation = np.array( [ side_track_error, course_angle_err, along_track_error, potential, r ], dtype=np.float32 )

        else:
            pass
            #observation = np.array([0, course_angle_err, dist_to_goal, 0], dtype=np.float32)

        return ts.restart(observation)

# ENDS HERE