import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import parameters as params
from tf_agents.environments import tf_py_environment
from tf_agents.environments import wrappers
import os
from env import sim_environment
import re

DEG_TO_RAD = np.pi/180
RAD_TO_DEG = 180/np.pi
MPS_TO_KMH = 18/5
KMH_TO_MPS = 5/18           
    
model_name = params.model_name
# pathname = 'saved_models/side/'
pathname = 'saved_models/pot/'


def testall():

    dir_list = os.listdir(pathname + model_name)
    regex = re.compile(model_name + '_ep_')
    new_dir_list = list(filter(regex.match, dir_list))

    for policy_name in new_dir_list:

        env = wrappers.TimeLimit(sim_environment(Is_test_flag = True), duration = 500)

        tf_env = tf_py_environment.TFPyEnvironment(env)

        train_step_counter = tf.Variable(0)

        # Reset the train step
        episodes = 1

        saved_policy = tf.compat.v2.saved_model.load(pathname+ model_name + '/' + policy_name)

        for _ in range(episodes):

            time_step = tf_env.reset()
            # time = 0

            # while not np.equal(time_step.step_type, 2):
            #     action_step = saved_policy.action(time_step)
            #     time_step = tf_env.step(action_step.action)

            while not np.equal(time_step.step_type, 2):

                epsilon = 0.0

                if np.random.random() < epsilon:
                    action_no1 = np.random.randint(0, 3)
                    action = tf.constant([action_no1],  shape=(1,), dtype = np.int64, name='action')
                    time_step = tf_env.step(action)

                else:
                    action_step = saved_policy.action(time_step)
                    time_step = tf_env.step(action_step.action)

        fig = plt.figure()
        X = env.x_traj
        Y = env.y_traj

        plt.plot(X,Y)

        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.axis('equal')
        plt.legend(["Path Followed","Start Waypoint","Predefined Path"],loc="center")
        plt_str = pathname + f'{model_name}/plots/{policy_name}.png'
        plt.savefig(plt_str)
        plt.close(fig)

        print("rmse:" , (np.sqrt(np.mean(np.square(env.potential_traj)))))


def testone():

    dir_list = os.listdir(pathname + model_name)
    regex = re.compile(model_name + '_ep_'+'3809')
    new_dir_list = list(filter(regex.match, dir_list))

    for policy_name in new_dir_list:

        env = wrappers.TimeLimit(sim_environment(Is_test_flag = True), duration = 500)

        tf_env = tf_py_environment.TFPyEnvironment(env)

        train_step_counter = tf.Variable(0)

        # Reset the train step
        episodes = 1

        saved_policy = tf.compat.v2.saved_model.load(pathname+ model_name + '/' + policy_name)

        for _ in range(episodes):

            time_step = tf_env.reset()
            # time = 0

            # while not np.equal(time_step.step_type, 2):
            #     action_step = saved_policy.action(time_step)
            #     time_step = tf_env.step(action_step.action)

            while not np.equal(time_step.step_type, 2):

                epsilon = 0.01

                if np.random.random() < epsilon:
                    action_no1 = np.random.randint(0, 3)
                    action = tf.constant([action_no1],  shape=(1,), dtype = np.int64, name='action')
                    time_step = tf_env.step(action)

                else:
                    action_step = saved_policy.action(time_step)
                    time_step = tf_env.step(action_step.action)


        plt.figure()
        X = env.x_traj
        Y = env.y_traj

        theta = np.array(env.theta_traj)

        plt.plot(X,Y)

        plt.xlabel("X")
        plt.ylabel("Y")
        #plt.axis('equal')
        plt.legend(["Path Followed","Start Waypoint","Predefined Path"],loc="center")
        ax = plt.gca()
        plt_str = pathname + f'{model_name}/plots/{policy_name}.png'
        #plt.savefig(plt_str)
        #plt.show()

        plt.figure()
        plt.plot(theta*RAD_TO_DEG)
        plt.xlabel("time")
        plt.ylabel("theta")
        # plt.axis('equal')
        plt_str = pathname + f'{model_name}/plots/{policy_name}-theta.png'
        plt.savefig(plt_str)
        print("rmse:" , (np.sqrt(np.mean(np.square(env.potential_traj)))))

def Plottest():

    dir_list = os.listdir(pathname + model_name)
    regex = re.compile(model_name + '_ep_'+'3809')
    new_dir_list = list(filter(regex.match, dir_list))

    for policy_name in new_dir_list:

        env = wrappers.TimeLimit(sim_environment(Is_test_flag = True), duration = 500)

        tf_env = tf_py_environment.TFPyEnvironment(env)

        train_step_counter = tf.Variable(0)

        # Reset the train step
        episodes = 1

        saved_policy = tf.compat.v2.saved_model.load(pathname+ model_name + '/' + policy_name)

        for _ in range(episodes):

            time_step = tf_env.reset()
            # time = 0

            while not np.equal(time_step.step_type, 2):

                epsilon = 0.00
                action_step = saved_policy.action(time_step)
                time_step = tf_env.step(action_step.action)

                if np.random.random() < epsilon:
                    action_no1 = np.random.randint(0, 3)
                    action = tf.constant([action_no1],  shape=(1,), dtype = np.int64, name='action')
                    time_step = tf_env.step(action)

        X = env.x_traj
        Y = env.y_traj
        theta = env.psi_traj

        fig, ax = plt.subplots()

        ax.plot(X,Y)
        ax.set_xlabel("x-axis [m]")
        ax.set_ylabel("y-axis [m]")

        # Plot car
        x_car = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0,  0])*4.5
        y_car = np.array([  -1,    1,    1,   0,   -1,   -1,    0,   0,    1, 1, -1])*1.8/2
        
        # Draw car body, first and last
        m_indx = [1,-1]
        for m_i in m_indx:
            x_new_car = X[m_i] + x_car * np.cos(theta[m_i]) - y_car * np.sin(theta[m_i])
            y_new_car = Y[m_i] + x_car * np.sin(theta[m_i]) + y_car * np.cos(theta[m_i])
            ax.plot(x_new_car, y_new_car)

        # Draw obs car body, first and last
        Xo = [env.obs1.x[4]]
        Yo = [env.obs1.x[5]]
        theta = [0]
        m_i = 0
        x_new_car = Xo[m_i] + x_car * np.cos(theta[m_i]) - y_car * np.sin(theta[m_i])
        y_new_car = Yo[m_i] + x_car * np.sin(theta[m_i]) + y_car * np.cos(theta[m_i])
        ax.plot(x_new_car, y_new_car)

        env.road.draw_road_section(ax)

# testall()
testone()
# Plottest()