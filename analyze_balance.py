import numpy as np
import glob
from math import fabs
import statistics

# file_names = sorted(glob.glob('./logs/td3_lstm_digit_mujoco_gpu/seed_1/final_test_episodes/*.npz'))
# file_names = sorted(glob.glob('./logs/sac_ae_digit_mujoco_gpu/seed_1/final_test_episodes_final_test_episodes_step_2020000_success_94_ypr/*.npz'))
file_names = sorted(glob.glob('./logs/sac_ae_digit_mujoco_gpu/seed_1/final_test_episodes/*.npz'))
# file_names = sorted(glob.glob('./logs/sac_ae_lip_gpu/seed_1/final_test_episodes/*.npz'))
# file_names = sorted(glob.glob('./logs/dwa_digit_mujoco/seed_1/final_test_episodes/*.npz'))
# file_names = sorted(glob.glob('./logs/gcn_digit_mujoco_gpu/seed_1/final_test_episodes/*.npz'))
# file_names = sorted(glob.glob('./logs/drl_vo_digit_mujoco_gpu/seed_1/final_test_episodes/*.npz'))
file_number = len(file_names)
# print(file_number)

yaw_c = []
yaw = []
pitch = []
roll = []
success_times =[]
for i in range(file_number):
    if 'fail' in file_names[i]:
        continue
    success_time = 0
    log_data = np.load(file_names[i])
    robot = log_data['robot']
    steps = robot.shape[0]
    ypr = log_data['ypr']
    for j in range(steps):
        if ypr[j][0] == -100 or ypr[j][0] == 100:
            break
        success_time += 0.4
        yaw_c.append(fabs(robot[j][3]))
        yaw.append(fabs(ypr[j][2]))
        pitch.append(fabs(ypr[j][1]))
        roll.append(fabs(ypr[j][0]))
        # yaw_c.append(robot[j][3])
        # yaw.append(ypr[j][2])
        # pitch.append(ypr[j][1])
        # roll.append(ypr[j][0])
    success_times.append(success_time)
print('---------')
print('average yaw command: ', statistics.mean(yaw_c))
print('stdev yaw command: ', statistics.stdev(yaw_c))
print('---------')
print('average yaw: ', statistics.mean(yaw))
print('stdev yaw: ', statistics.stdev(yaw))
print('---------')
print('average pitch: ', statistics.mean(pitch))
print('stdev pitch: ', statistics.stdev(pitch))
print('---------')
print('average roll: ', statistics.mean(roll))
print('stdev roll: ', statistics.stdev(roll))
print('---------')
print(len(success_times))
print('average navigation: ', statistics.mean(success_times))
print('stdev navigation: ', statistics.stdev(success_times))
