import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle
import glob
import time
from math import cos, sin, hypot, fabs

import psutil
import os

# Get list of all CPU core IDs
all_cores = list(range(psutil.cpu_count()))

# Define which cores to use
cores_to_use = [2, 3]  # Example: use core 0 and core 1

# Validate cores to use
cores_to_use = [c for c in cores_to_use if c in all_cores]

# Set affinity
if cores_to_use:
    pid = os.getpid()
    os.sched_setaffinity(pid, cores_to_use)
    print(f"Process {pid} set affinity to cores: {cores_to_use}")
else:
    print("Invalid core selection")

n_laser = 1800


complex_env = True
# file_name = sorted(glob.glob('./logs/PPO_RNN_discrete_multi_workers/crowd_sim/seed_0/eval_120000_8_fail/*.npz'))
# specific_file_name = './logs/gcn_digit_mujoco_gpu/seed_1/final_test_episodes/eval_9999_0_fail.npz'
# specific_file_name = './logs/sac_ae_digit_mujoco_gpu/seed_1/evaluation_episodes/eval_1280000_1.npz'
# specific_file_name = './logs/sac_ae_digit_mujoco_gpu/seed_1/final_test_episodes/eval_9999_0.npz'
specific_file_name = './logs/sac_ae_digit_mujoco_gpu_final/seed_1/final_test_episodes/eval_0_0_1725289931.1949942.npz'
# specific_file_name = './logs/dwa_digit_mujoco/seed_1/final_test_episodes/eval_9999_0.npz'
# specific_file_name = './logs/td3_lstm_digit_mujoco_gpu/seed_1/final_test_episodes/eval_9999_0_fail.npz'
# specific_file_name = './logs/drl_vo_digit_mujoco_gpu/seed_1/final_test_episodes/eval_9999_0_fail.npz'
# specific_file_name = './logs/sac_ae_lip_gpu/seed_1/final_test_episodes/eval_9999_0_fail.npz'
log_env = np.load(specific_file_name)

robot = log_env['robot']
steps = robot.shape[0]
humans = log_env['humans']
laser = log_env['laser']
laser_beam = laser.shape[1]
human_num = humans.shape[1]
goal = log_env['goal']
ypr = np.abs(log_env['ypr'])

static_obstacles = []



#####real time plot for simulation case########
# radius = 0.3
# plt.ion()
# plt.show()
# fig, ax = plt.subplots(figsize=(10, 10))
# for i in range(steps):
#     if fabs(robot[i][0]) > 50.0:
#         break
#     ax.set_xlim(-5.0, 5.0)
#     ax.set_ylim(-5.0, 5.0)
#     scan_intersection = []
#     for laser_i in range(laser_beam):
#         scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
#     for human_i in range(human_num):
#         human_circle = plt.Circle(humans[i][human_i], radius, fill=False, color='b')
#         ax.add_artist(human_circle)
#     ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius, fill=True, color='r'))
#     ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
#     plt.text(-4.5, -4.5, str(round(i * 0.4, 2)), fontsize=20)
#     print('action: ', robot[i][2], robot[i][3])
#     print('ypr: ', ypr[i])
#     if complex_env:
#         static_obstacles = log_env['static_obstacles']
#         static_obstacle_num = static_obstacles.shape[1]
#         for static_i in range(static_obstacle_num):
#             ax.add_patch(Rectangle([static_obstacles[i][static_i][0], static_obstacles[i][static_i][1]], 
#                                     static_obstacles[i][static_i][2], static_obstacles[i][static_i][3],
#                                     facecolor='c',
#                                     fill=True))

#     x, y, theta = robot[i][0], robot[i][1], robot[i][4]
#     print('robot position: ', x, y)
#     dx = cos(theta)
#     dy = sin(theta)
#     ax.arrow(x, y, dx, dy,
#         width=0.01,
#         length_includes_head=True, 
#         head_width=0.15,
#         head_length=1,
#         fc='r',
#         ec='r')
    
#     ii = 0
#     lines = []
#     while ii < n_laser:
#         lines.append(scan_intersection[ii])
#         ii = ii + 36
#     lc = mc.LineCollection(lines)
#     ax.add_collection(lc)
#     plt.draw()
#     plt.pause(0.001)
#     plt.cla()
#     time.sleep(0.2)
####real time plot for simulation case########

###### save pdf for simulation case #########
# radius = 0.3
# for i in range(steps):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.set_xlim(-5.0, 5.0)
#     ax.set_ylim(-5.0, 5.0)
#     scan_intersection = []
#     for laser_i in range(laser_beam):
#         scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
#     for human_i in range(human_num):
#         human_circle = plt.Circle(humans[i][human_i], radius, fill=False, color='b')
#         ax.add_artist(human_circle)

#     ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius, fill=True, color='r'))
#     ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))
#     # plt.text(-4.5, -4.5, str(round(i * 0.2, 2)), fontsize=20)

#     ii = 0
#     lines = []
#     while ii < n_laser:
#         lines.append(scan_intersection[ii])
#         ii = ii + 36
#     lc = mc.LineCollection(lines)
#     ax.add_collection(lc)
#     plt.savefig("SimImagePDF_" + str(i) + ".pdf", format="pdf", bbox_inches="tight")
#     plt.show()
# ####### save pdf for simulation case #########

# ##### video ########
radius = 0.3
metadata = dict(title='EGO', artist='Matplotlib',comment='EGO test')
writer = FFMpegWriter(fps=2.5, metadata=metadata)

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
plt.tick_params(labelsize=24)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Times New Roman") for label in labels]



ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
 
with writer.saving(fig, "digit_hardware_seed_5.mp4", 250):
    for i in range(steps):
        if fabs(robot[i][0]) > 50.0:
            break
        ax.clear()
        ax.set_xlabel('x/m', fontproperties = 'Times New Roman', fontsize=24)
        ax.set_ylabel('y/m', fontproperties = 'Times New Roman', fontsize=24) 
        plt.tick_params(labelsize=24)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname("Times New Roman") for label in labels]

        
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.0, 5.0)

        for human_i in range(human_num):
            human_circle = plt.Circle(humans[i][human_i], radius, fill=False, color='b')
            ax.add_artist(human_circle)
        ax.add_artist(plt.Circle((robot[i][0], robot[i][1]), radius, fill=True, color='r'))
        ax.add_artist(plt.Circle(goal[i], radius, fill=True, color='g'))

        x, y, theta = robot[i][0], robot[i][1], robot[i][4]
        dx = cos(theta)
        dy = sin(theta)
        ax.arrow(x, y, dx, dy,
            width=0.01,
            length_includes_head=True, 
            head_width=0.15,
            head_length=1,
            fc='r',
            ec='r')
    
        if complex_env:
            static_obstacles = log_env['static_obstacles']
            static_obstacle_num = static_obstacles.shape[1]
            for static_i in range(static_obstacle_num):
                ax.add_patch(Rectangle([static_obstacles[i][static_i][0], static_obstacles[i][static_i][1]], 
                                        static_obstacles[i][static_i][2], static_obstacles[i][static_i][3],
                                        facecolor='c',
                                        fill=True))
        scan_intersection = []
        for laser_i in range(laser_beam):
            scan_intersection.append([(laser[i][laser_i][0], laser[i][laser_i][1]), (laser[i][laser_i][2], laser[i][laser_i][3])])
        ii = 0
        lines = []
        while ii < n_laser:
            lines.append(scan_intersection[ii])
            ii = ii + 36
        lc = mc.LineCollection(lines)
        ax.add_collection(lc)

        ax.text(-4.5, -4.5, str(round(i * 0.4, 2)), fontsize=20)
        writer.grab_frame()

# if ffmeg error occurs
# do not use anaconda env