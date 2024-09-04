import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
from math import fabs, cos, sin

features = np.loadtxt('feature_data/features.txt')
log_env = np.load('feature_data/eval_9999_0.npz')

feature_dim = 50
feature_idx = 40
radius = 0.3
n_laser = 1800

steps = 0
for i in range(features.shape[0]):
    if fabs(features[i][0]) > 50:   
        break
    steps += 1

robot = log_env['robot']
humans = log_env['humans']
laser = log_env['laser']
laser_beam = laser.shape[1]
human_num = humans.shape[1]

for j in range(9):
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_xlim(-5.0, 5.0)
    ax1.set_ylim(-5.0, 5.0)
    plt.xlabel("x/m", fontsize=18)  # Set the x-axis label text size to X
    plt.ylabel("y/m", fontsize=18)  # Set the y-axis label text size to X
    plt.tick_params(axis='x', labelsize=18)  # Change x-axis tick label size to X
    plt.tick_params(axis='y', labelsize=18)  # Change y-axis tick label size to X
    scan_intersection = []
    for laser_i in range(laser_beam):
        scan_intersection.append([(laser[feature_idx - j][laser_i][0], laser[feature_idx - j][laser_i][1]), 
                                  (laser[feature_idx - j][laser_i][2], laser[feature_idx - j][laser_i][3])])
    for human_i in range(human_num):
        human_circle = plt.Circle(humans[feature_idx - j][human_i], radius, fill=False, color='b', alpha=1.0-j*0.1)
        ax1.add_artist(human_circle)
    ax1.add_artist(plt.Circle((robot[feature_idx - j][0], robot[feature_idx - j][1]), radius, fill=True, color='r', alpha=1.0-j*0.1))
    static_obstacles = log_env['static_obstacles']
    static_obstacle_num = static_obstacles.shape[1]
    for static_i in range(static_obstacle_num):
        ax1.add_patch(Rectangle([static_obstacles[feature_idx - j][static_i][0], static_obstacles[feature_idx - j][static_i][1]], 
                                static_obstacles[feature_idx - j][static_i][2], static_obstacles[feature_idx - j][static_i][3],
                                facecolor='c',
                                fill=True,
                                alpha=1.0-j*0.1))

    x, y, theta = robot[feature_idx - j][0], robot[feature_idx - j][1], robot[feature_idx - j][4]
    dx = cos(theta)
    dy = sin(theta)
    ax1.arrow(x, y, dx, dy,
        width=0.01,
        length_includes_head=True, 
        head_width=0.15,
        head_length=1,
        fc='r',
        ec='r',
        alpha=1.0-j*0.1)

    ii = 0
    lines = []
    while ii < n_laser:
        lines.append(scan_intersection[ii])
        ii = ii + 36
    lc = mc.LineCollection(lines, alpha=1.0-j*0.1)
    ax1.add_collection(lc)

plt.figure(figsize=(9, 4))

grid = features[feature_idx].reshape((1, 50))
# print(grid)
# Create the plot
plt.figure(figsize=(9, 4))
plt.imshow(grid, cmap='viridis', interpolation='none')

# Add grid lines
# plt.grid(which='both', color='gray', linestyle='-', linewidth=2)

# Set grid line positions
# plt.xticks(np.arange(-0.5, 10, 1), [])
# plt.yticks(np.arange(-0.5, 10, 1), [])
# Remove y-axis ticks
plt.yticks([])

# Show color bar
plt.colorbar(location='bottom')

# Display the plot
plt.show()