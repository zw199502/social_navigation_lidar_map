import argparse
import numpy as np
import random
import os
import torch
import time
from collections import namedtuple

from algos.DRL_VO.drl_vo import DRLVoAgent
from algos.DRL_VO.utils import eval_policy_mode, set_seed_everywhere
from info import *
from crowd_sim_complex_constraint_drl_vo import CrowdSim
from torch.utils.tensorboard import SummaryWriter
from threading import Lock

# digit in mujoco
from digit_mujoco.cfg.digit_env_config import DigitEnvConfig
from digit_mujoco.envs.digit.digit_env_flat import DigitEnvFlat

# digit in arsim
from digit_arsim.digit_env_arsim import DigitEnvArSim

import time
import psutil

# # Get the current process
# p = psutil.Process(os.getpid())

# # Set CPU affinity to core 0 (you can change the core number as needed)
# p.cpu_affinity([0])

# please specify cpu cores, otherwise, segmentation fault may occur !!!!
# It is still unknown why this fault happens.
# If the fault shows up, please use the command sudo reboot to restart the PC
# otherwise, the fault may occur again.
# Get list of all CPU core IDs
all_cores = list(range(psutil.cpu_count()))

# Define which cores to use
cores_to_use = [0, 1]  # Example: use core 0 and core 1

# Validate cores to use
cores_to_use = [c for c in cores_to_use if c in all_cores]

# Set affinity
if cores_to_use:
    pid = os.getpid()
    os.sched_setaffinity(pid, cores_to_use)
    print(f"Process {pid} set affinity to cores: {cores_to_use}")
else:
    print("Invalid core selection")
    
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_or_test_env, current_steps, 
                eval_episodes=100, save_directory=None, 
                if_save_video=False, final_test=False):
    avg_reward = 0.
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    collision_cases = []
    timeout_cases = []
    if_save_data = False
    for i in range(eval_episodes):
        if_save_data = (i < 10 or final_test)
        obs, robot_goal_state  = eval_or_test_env.reset(eval=True, save_data=if_save_data)
        # eval_or_test_env.render()
        # time.sleep(0.2)
        done = False
        ep_step = 0
        
        while ep_step < eval_or_test_env.max_episode_step:
            # t1 = time.time()
            with eval_policy_mode(policy):
                action, _ = policy.select_action(obs, robot_goal_state, eval=True)
            # action = eval_or_test_env.dwa_compute_action()
            obs, robot_goal_state, reward, done, info = eval_or_test_env.step(action, eval=True, save_data=if_save_data)
            # print('time: ', time.time() - t1) # 7.5ms
            # eval_or_test_env.render()
            # time.sleep(0.2)
            avg_reward += reward
            ep_step = ep_step + 1
            if done or ep_step == eval_or_test_env.max_episode_step or isinstance(info, ReachGoal):
                if ep_step == eval_or_test_env.max_episode_step:
                    timeout += 1
                    timeout_cases.append(i)
                    timeout_times.append(eval_or_test_env.time_limit)
                    print('evaluation episode ' + str(i) + ', time out: ' + str(ep_step))
                else:
                    if isinstance(info, ReachGoal):
                        success += 1
                        success_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', goal reaching at evaluation step: ' + str(ep_step))
                    elif isinstance(info, Collision):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', collision occur at evaluation step: ' + str(ep_step))  
                    elif isinstance(info, DigitCrazy):
                        collision += 1
                        collision_cases.append(i)
                        collision_times.append(eval_or_test_env.global_time)
                        print('evaluation episode ' + str(i) + ', crazy digit at evaluation step: ' + str(ep_step))            
                    else:
                        raise ValueError('Invalid end signal from environment')
                break
        if save_directory is not None:
            if isinstance(info, ReachGoal):
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '.npz'
            else:
                file_name = save_directory + '/eval_' + str(current_steps) + '_' + str(i) + '_fail' + '.npz'
            if if_save_data:
                np.savez_compressed(file_name, **eval_or_test_env.log_env)
                if if_save_video:
                    eval_or_test_env.save_video(current_steps, i)

    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = sum(success_times) / len(success_times) if success_times else eval_or_test_env.time_limit

    
    return success_rate, collision_rate, avg_nav_time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy", default="drl_vo")
    # options, lip, digit_mujoco, now digit_arsim is not supported for training
    parser.add_argument("--robot_model", default="digit_mujoco")
    # options, lip, digit_mujoco, digit_arsim, and digit_hardware
    parser.add_argument("--robot_test_model", default="digit_mujoco")
    # device
    parser.add_argument("--device", type=str, default='cuda:0')
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=1, type=int)
     # How often (time steps) we evaluate the trained model
    parser.add_argument("--eval_freq", default=20000, type=int)
    # How often (time steps) we save the trained model
    parser.add_argument("--save_model_freq", default=200000, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=6e6, type=int)
    
    # training
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--buffer_capacity', default=1000, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    # ppo
    parser.add_argument('--discount', default=0.99, type=float)
    # clip parameter for PPO
    parser.add_argument('--eps_clip', default=0.2, type=float)
    # update policy for K epochs in one PPO update
    parser.add_argument('--k_epochs', default=20, type=int)

    # Model load file name, "" doesn't load, "default" uses file_name
    # args.load_model, format, step_NO_success_NO
    # parser.add_argument("--load_model", type=str, default="step_480000_success_88")
    parser.add_argument("--load_model", type=str, default="")

    # parser.add_argument("--load_test_model", type=str, default="step_1620000_success_90")
    parser.add_argument("--load_test_model", type=str, default="")
    # environment settings
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--lidar_dim", type=int, default=1800)
    parser.add_argument("--lidar_feature_dim", type=int, default=50)
    # lidar to image
    parser.add_argument('--image_size', default=100, type=int)
    # 2 robot speed, 2 local goal
    parser.add_argument("--robot_goal_state_dim", type=int, default=4)
    parser.add_argument("--laser_angle_resolute", type=float, default=0.003490659)
    parser.add_argument("--laser_min_range", type=float, default=0.27)
    parser.add_argument("--laser_max_range", type=float, default=6.0)
    parser.add_argument("--square_width", type=float, default=10.0)
    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, RobotModel: {args.robot_model}, Seed: {args.seed}")
    print("---------------------------------------")
    
    if args.device == 'cpu':
        file_prefix = './logs/' + args.policy + '_' + args.robot_model + '_cpu'
    else:
        file_prefix = './logs/' + args.policy + '_' + args.robot_model + '_gpu'
   
    file_prefix = file_prefix + '/seed_' + str(args.seed)  
    file_results = file_prefix + '/results'
    file_models = file_prefix + '/models'
    file_evaluation_episodes = file_prefix + '/evaluation_episodes'
    file_final_test_episodes = file_prefix + '/final_test_episodes'
    file_buffer = file_prefix + '/buffer'

    if not os.path.exists(file_results):
        os.makedirs(file_results)

    if not os.path.exists(file_models):
        os.makedirs(file_models)

    if not os.path.exists(file_evaluation_episodes):
        os.makedirs(file_evaluation_episodes)
        
    if not os.path.exists(file_final_test_episodes):
        os.makedirs(file_final_test_episodes)
        
    if not os.path.exists(file_buffer):
        os.makedirs(file_buffer)

    writer = SummaryWriter(log_dir=file_results)

    action_range = np.array([[0.0, -0.5],
                             [0.4,  0.5]])
    
    action_resolution = [0.05, 0.1]
    action_num = [int((action_range[1, 0] - action_range[0, 0]) / action_resolution[0]) + 1,
                  int((action_range[1, 1] - action_range[0, 1]) / action_resolution[1]) + 1]
    action_choice_1 = np.linspace(action_range[0, 0], action_range[1, 0], num=action_num[0])
    action_choice_2 = np.linspace(action_range[0, 1], action_range[1, 1], num=action_num[1])
    action_choice_dim = action_num[0] * action_num[1]
    action_choices = np.zeros((action_num[0], action_num[1], 2), dtype=np.float32)
    for i in range(action_num[0]):
        for j in range(action_num[1]):
            action_choices[i, j, :] = np.array([action_choice_1[i], action_choice_2[j]]) 
    action_choices = np.reshape(action_choices, (action_choice_dim, 2))
    
    # config
    cfg_digit_env_eval = DigitEnvConfig()
    cfg_digit_env_eval.vis_record.visualize = True
    cfg_digit_env_eval.vis_record.record = True
    digit_env_eval = DigitEnvFlat(cfg_digit_env_eval, file_evaluation_episodes)
    
    cfg_digit_env_test = DigitEnvConfig()
    cfg_digit_env_test.vis_record.visualize = True
    cfg_digit_env_test.vis_record.record = True
    digit_env_test = DigitEnvFlat(cfg_digit_env_test, file_final_test_episodes)
    
    cfg_digit_env_train = DigitEnvConfig()
    digit_env_train = DigitEnvFlat(cfg_digit_env_train, file_evaluation_episodes)

    if args.robot_model == 'digit_mujoco':
        digit_dim = digit_env_eval.observation_space.shape[0]
        env = CrowdSim(args, action_range, digit_dim, 
                       action_choices=action_choices, digit_env=digit_env_train)
        eval_env = CrowdSim(args, action_range, digit_dim, 
                            action_choices=action_choices, digit_env=digit_env_eval)
        if_save_video = True
    elif args.robot_model == 'lip':
        digit_dim = 0
        env = CrowdSim(args, action_range, digit_dim, action_choices=action_choices)
        eval_env = CrowdSim(args, action_range, digit_dim, action_choices=action_choices)
        if_save_video = False
    else:
        raise NotImplementedError(args.robot_model)

    if args.robot_test_model == 'digit_mujoco':
        if_save_video_test = True
        test_env = CrowdSim(args, action_range, digit_dim, 
                            action_choices=action_choices, digit_env=digit_env_test)
    elif args.robot_test_model == 'lip':
        if_save_video_test = False
        test_env = CrowdSim(args, action_range, digit_dim, action_choices=action_choices)
    elif args.robot_test_model == 'digit_arsim':
        if_save_video_test = False
        digit_env_test = DigitEnvArSim()
        test_env = CrowdSim(args, action_range, digit_dim, 
                            action_choices=action_choices, digit_env=digit_env_test)
    elif args.robot_test_model == 'digit_hardware':
        if_save_video_test = False
        from digit_arsim.digit_env_hardware import DigitEnvHardware
        import rospy
        digit_env_test = DigitEnvHardware()
        test_env = CrowdSim(args, action_range, digit_dim, 
                            action_choices=action_choices, digit_env=digit_env_test)
    else:
        raise NotImplementedError(args.robot_test_model)
    # please manually set seeds when test with digit_arsim
    # otherwise, set it as args.seed
    set_seed_everywhere(args.seed)

    device = torch.device(args.device)
    obs_shape = (3, args.image_size, args.image_size)
    robot_goal_state_dim = args.robot_goal_state_dim
    action_shape = (2,)
    agent = DRLVoAgent(
            obs_shape,
            robot_goal_state_dim,
            digit_dim,
            action_shape,
            action_range,
            device,
            batch_size=args.batch_size,
            buffer_capacity=args.buffer_capacity,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            k_epochs = args.k_epochs,
            eps_clip = args.eps_clip,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    
    checkpoint_steps = 0
    if args.load_model != "":
        # replay_buffer.load(file_buffer)
        agent.load(file_models + '/' + args.load_model)
        # args.load_model, format, step_NO_success_NO
        # extract the first number
        checkpoint_steps = int(args.load_model.split('_')[1])

    if args.load_test_model != "":
        print('start to test')
        agent.load(file_models + '/' + args.load_test_model)
        debug = False
        if debug:
            test_times = 1
            current_step = 9999
        else:
            test_times = 500
            current_step = 0
        if args.robot_test_model == 'digit_arsim':
            test_times = 1 # currently, arsim only supports one time test because resetting the environments with python has not been developed.
        success_rate, collision_rate, avg_nav_time = eval_policy(agent, test_env, current_step, 
                                                                 eval_episodes=test_times, save_directory=file_final_test_episodes, 
                                                                 if_save_video=if_save_video_test, final_test=True)
        if args.robot_test_model == 'digit_arsim':
            test_env.digit_env.join_threads() # after the episode, let the digit in arsim step in place
        print('success_rate, collision_rate, avg_nav_time')
        print(success_rate, collision_rate, avg_nav_time)
        return

    evaluations = []

    episode_num = 0
    t = checkpoint_steps + 1
    Transition = namedtuple('Transition', ['obs', 'robot_goal_state', 'a', 'a_log_p', 'r', 'not_done', 'obs_', 'robot_goal_state_'])
    while t < int(args.max_timesteps) + 1:
        obs, robot_goal_state = env.reset()
        done = False
        episode_timesteps = 0
        for tt in range(1, env.max_episode_step + 1):
            # select action with policy
            action, action_log_prob = agent.select_action(obs, robot_goal_state)
            obs_, robot_goal_state_, reward, done, info = env.step(action)

            start_update = agent.store(Transition(obs, robot_goal_state, action, action_log_prob, 
                                                  reward, float(1.0 - done), obs_, robot_goal_state_))
            obs = obs_
            robot_goal_state = robot_goal_state_

            episode_timesteps += 1

            t += 1

            # update PPO agent
            if start_update:
                agent.update(writer, t)

            # Evaluate episode
            if t % args.eval_freq == 0:
                success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, t, 
                                                                         save_directory=file_evaluation_episodes, 
                                                                         if_save_video=if_save_video)
                file_name = '/step_' + str(t) + '_success_' + str(int(success_rate * 100))
                print('success_rate, collision_rate, avg_nav_time at step ' + str(t))
                print(success_rate, collision_rate, avg_nav_time)
                writer.add_scalar('eval/success_rate', success_rate, t)
                writer.add_scalar('eval/collision_rate', collision_rate, t)
                evaluations.append(success_rate)
                np.savetxt(file_results + file_name + '.txt', evaluations)
                if success_rate > 0.85 or t % args.save_model_freq == 0:
                    agent.save(file_models + file_name)
                    # replay_buffer.save(file_buffer)
            if done:
                break
        episode_num += 1
        if done or episode_timesteps == env.max_episode_step or isinstance(info, ReachGoal):
            if episode_timesteps == env.max_episode_step:
                print('total step ' + str(t) + ', train episode ' + str(episode_num) + ', time out: ' + str(episode_timesteps))
            else:
                if isinstance(info, ReachGoal):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num) + 
                        ', goal reaching at train step: ' + str(episode_timesteps))
                elif isinstance(info, Collision):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num) + 
                        ', collision occur at train step: ' + str(episode_timesteps))     
                elif isinstance(info, DigitCrazy):
                    print('total step ' + str(t) + ', train episode ' + str(episode_num) + 
                        ', crazy digit at train step: ' + str(episode_timesteps))                
                else:
                    raise ValueError('Invalid end signal from environment')
        
         
    print('final test')
    success_rate, collision_rate, avg_nav_time = eval_policy(agent, eval_env, t, 
                                                             eval_episodes=500, save_directory=file_final_test_episodes,
                                                             if_save_video=if_save_video, final_test=True)
    print('success_rate, collision_rate, avg_nav_time')
    print(success_rate, collision_rate, avg_nav_time)


if __name__ == "__main__":
    main()
