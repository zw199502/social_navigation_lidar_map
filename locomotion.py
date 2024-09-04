from digit_mujoco.cfg.digit_env_config import DigitEnvConfig
from digit_mujoco.envs.digit.digit_env_flat import DigitEnvFlat
import numpy as np
import matplotlib.pylab as plt
from time import time, sleep

file_evaluation_episodes = 'abc'
cfg_digit_env_eval = DigitEnvConfig()
cfg_digit_env_eval.vis_record.visualize = True
digit_env_eval = DigitEnvFlat(cfg_digit_env_eval, log_dir=file_evaluation_episodes)
digit_env_eval.reset()

positions = []
vel_command_to_digit = {
    'x_vel': 0.0,
    'y_vel': 0.0,
    'yaw_vel': -0.2
}
digit_env_eval.set_vel_command(vel_command_to_digit)
time_step = 10.0
for i in range(int(time_step / digit_env_eval.cfg.control.control_dt)):
    st_time = time()
    digit_env_step = digit_env_eval.step(np.zeros(12))
    # print(i, digit_env_eval.domain_switch)
    end_time = time()
    if (end_time - st_time) < digit_env_eval.cfg.control.control_dt:
        sleep(digit_env_eval.cfg.control.control_dt - (end_time - st_time))
    positions.append(np.array([digit_env_eval.root_xy_pos[0], digit_env_eval.root_xy_pos[1]]))

vel_command_to_digit = {
    'x_vel': 0.3,
    'y_vel': 0.0,
    'yaw_vel': 0.0
}

digit_env_eval.set_vel_command(vel_command_to_digit)
time_step = 20.0
for i in range(int(time_step / digit_env_eval.cfg.control.control_dt)):
    st_time = time()
    digit_env_step = digit_env_eval.step(np.zeros(12))
    end_time = time()
    if (end_time - st_time) < digit_env_eval.cfg.control.control_dt:
        sleep(digit_env_eval.cfg.control.control_dt - (end_time - st_time))
    positions.append(np.array([digit_env_eval.root_xy_pos[0], digit_env_eval.root_xy_pos[1]]))
    # print(digit_env_eval.root_lin_vel[:2], digit_env_eval.qvel[:2])
    # print(digit_env_eval.root_ang_vel[2])
positions_numpy = np.array(positions)
plt.plot(positions_numpy[:, 0], positions_numpy[:, 1])
plt.show()
sleep(50)