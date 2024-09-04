import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from math import fabs

ours_full = np.load('evaluation_trajectory/ours_full.npz')
ours_lip = np.load('evaluation_trajectory/ours_lip.npz')
drl_vo = np.load('evaluation_trajectory/drl_vo.npz')
rgl = np.load('evaluation_trajectory/rgl.npz')
lndnl = np.load('evaluation_trajectory/td3_lstm.npz')
dwa = np.load('evaluation_trajectory/dwa.npz')

N = 5
radius = 0.3
goal = ours_full['goal'][0]
human_color = ['b', 'c', 'y', 'm']

robot_ours_full = ours_full['robot']
steps_ours_full = 0
while steps_ours_full < robot_ours_full.shape[0]:
    if fabs(robot_ours_full[steps_ours_full][0]) > 50:
        break
    steps_ours_full += 1
humans_ours_full = ours_full['humans']
human_num_ours_full = humans_ours_full.shape[1]
static_obstacles_ours_full = ours_full['static_obstacles']

robot_ours_lip = ours_lip['robot']
steps_ours_lip = 0
while steps_ours_lip < robot_ours_lip.shape[0]:
    if fabs(robot_ours_lip[steps_ours_lip][0]) > 50:
        break
    steps_ours_lip += 1
humans_ours_lip = ours_lip['humans']
human_num_ours_lip = humans_ours_lip.shape[1]
static_obstacles_ours_lip = ours_lip['static_obstacles']

robot_lndnl = lndnl['robot']
steps_lndnl = 0
while steps_lndnl < robot_lndnl.shape[0]:
    if fabs(robot_lndnl[steps_lndnl][0]) > 50:
        break
    steps_lndnl += 1
humans_lndnl = lndnl['humans']
human_num_lndnl = humans_lndnl.shape[1]
static_obstacles_lndnl = lndnl['static_obstacles']

robot_rgl = rgl['robot']
steps_rgl = 0
while steps_rgl < robot_rgl.shape[0]:
    if fabs(robot_rgl[steps_rgl][0]) > 50:
        break
    steps_rgl += 1
steps_rgl -= 70
humans_rgl = rgl['humans']
human_num_rgl = humans_rgl.shape[1]
static_obstacles_rgl = rgl['static_obstacles']

robot_drl_vo = drl_vo['robot']
steps_drl_vo = 0
while steps_drl_vo < robot_drl_vo.shape[0]:
    if fabs(robot_drl_vo[steps_drl_vo][0]) > 50:
        break
    steps_drl_vo += 1
humans_drl_vo = drl_vo['humans']
human_num_drl_vo = humans_drl_vo.shape[1]
static_obstacles_drl_vo = drl_vo['static_obstacles']

robot_dwa = dwa['robot']
steps_dwa = 0
while steps_dwa < robot_dwa.shape[0]:
    if fabs(robot_dwa[steps_dwa][0]) > 50:
        break
    steps_dwa += 1
humans_dwa = dwa['humans']
human_num_dwa = humans_dwa.shape[1]
static_obstacles_dwa = dwa['static_obstacles']

################## ours full ######################
fig1, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.add_artist(plt.Circle(goal, radius, fill=True, color='g'))
static_obstacles = static_obstacles_ours_full[0]
static_obstacle_num = static_obstacles.shape[0]
for static_i in range(static_obstacle_num):
    ax.add_patch(Rectangle([static_obstacles[static_i][0], static_obstacles[static_i][1]], 
                            static_obstacles[static_i][2], static_obstacles[static_i][3],
                            facecolor='k',
                            fill=True))
i = 0
count = int(steps_ours_full / N)
if steps_ours_full % N != 0:
    count += 1
count_c = 1
while i < steps_ours_full:
    ax.add_artist(plt.Circle((robot_ours_full[i][0], robot_ours_full[i][1]), radius, fill=True, color='r', alpha=count_c / count))
    human_position = []
    for human_i in range(human_num_ours_full):
        human_circle = plt.Circle(humans_ours_full[i][human_i], radius, fill=True, color=human_color[human_i], alpha=count_c / count)
        human_position.append(np.array([humans_ours_full[i][human_i][0], humans_ours_full[i][human_i][1]]))
        ax.add_artist(human_circle)
    i = i + N
    count_c += 1
if i - N != steps_ours_full - 1:
    ax.add_artist(plt.Circle((robot_ours_full[steps_ours_full - 1][0], robot_ours_full[steps_ours_full - 1][1]), radius, fill=True, color='r'))
    human_position = []
    for human_i in range(human_num_ours_full):
        human_circle = plt.Circle(humans_ours_full[steps_ours_full - 1][human_i], radius, fill=True, color=human_color[human_i])
        human_position.append(np.array([humans_ours_full[steps_ours_full - 1][human_i][0], humans_ours_full[steps_ours_full - 1][human_i][1]]))
        ax.add_artist(human_circle)
################## ours full ######################

################## ours lip ######################
fig2, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.add_artist(plt.Circle(goal, radius, fill=True, color='g'))
static_obstacles = static_obstacles_ours_lip[0]
static_obstacle_num = static_obstacles.shape[0]
for static_i in range(static_obstacle_num):
    ax.add_patch(Rectangle([static_obstacles[static_i][0], static_obstacles[static_i][1]], 
                            static_obstacles[static_i][2], static_obstacles[static_i][3],
                            facecolor='k',
                            fill=True))
i = 0
count = int(steps_ours_lip / N)
if steps_ours_lip % N != 0:
    count += 1
count_c = 1
while i < steps_ours_lip:
    ax.add_artist(plt.Circle((robot_ours_lip[i][0], robot_ours_lip[i][1]), radius, fill=True, color='r', alpha=count_c / count))
    human_position = []
    for human_i in range(human_num_ours_lip):
        human_circle = plt.Circle(humans_ours_lip[i][human_i], radius, fill=True, color=human_color[human_i], alpha=count_c / count)
        human_position.append(np.array([humans_ours_lip[i][human_i][0], humans_ours_lip[i][human_i][1]]))
        ax.add_artist(human_circle)
    i = i + N
    count_c += 1
if i - N != steps_ours_lip - 1:
    ax.add_artist(plt.Circle((robot_ours_lip[steps_ours_lip - 1][0], robot_ours_lip[steps_ours_lip - 1][1]), radius, fill=True, color='r'))
    human_position = []
    for human_i in range(human_num_ours_lip):
        human_circle = plt.Circle(humans_ours_lip[steps_ours_lip - 1][human_i], radius, fill=True, color=human_color[human_i])
        human_position.append(np.array([humans_ours_lip[steps_ours_lip - 1][human_i][0], humans_ours_lip[steps_ours_lip - 1][human_i][1]]))
        ax.add_artist(human_circle)
################## ours lip ######################


################## drl_vo ######################
fig3, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.add_artist(plt.Circle(goal, radius, fill=True, color='g'))
static_obstacles = static_obstacles_drl_vo[0]
static_obstacle_num = static_obstacles.shape[0]
for static_i in range(static_obstacle_num):
    ax.add_patch(Rectangle([static_obstacles[static_i][0], static_obstacles[static_i][1]], 
                            static_obstacles[static_i][2], static_obstacles[static_i][3],
                            facecolor='k',
                            fill=True))
i = 0
count = int(steps_drl_vo / N)
if steps_drl_vo % N != 0:
    count += 1
count_c = 1
while i < steps_drl_vo:
    ax.add_artist(plt.Circle((robot_drl_vo[i][0], robot_drl_vo[i][1]), radius, fill=True, color='r', alpha=count_c / count))
    human_position = []
    for human_i in range(human_num_drl_vo):
        human_circle = plt.Circle(humans_drl_vo[i][human_i], radius, fill=True, color=human_color[human_i], alpha=count_c / count)
        human_position.append(np.array([humans_drl_vo[i][human_i][0], humans_drl_vo[i][human_i][1]]))
        ax.add_artist(human_circle)
    i = i + N
    count_c += 1
if i - N != steps_drl_vo - 1:
    ax.add_artist(plt.Circle((robot_drl_vo[steps_drl_vo - 1][0], robot_drl_vo[steps_drl_vo - 1][1]), radius, fill=True, color='r'))
    human_position = []
    for human_i in range(human_num_drl_vo):
        human_circle = plt.Circle(humans_drl_vo[steps_drl_vo - 1][human_i], radius, fill=True, color=human_color[human_i])
        human_position.append(np.array([humans_drl_vo[steps_drl_vo - 1][human_i][0], humans_drl_vo[steps_drl_vo - 1][human_i][1]]))
        ax.add_artist(human_circle)
################## drl_vo ######################

################## rgl ######################
fig4, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.add_artist(plt.Circle(goal, radius, fill=True, color='g'))
static_obstacles = static_obstacles_rgl[0]
static_obstacle_num = static_obstacles.shape[0]
for static_i in range(static_obstacle_num):
    ax.add_patch(Rectangle([static_obstacles[static_i][0], static_obstacles[static_i][1]], 
                            static_obstacles[static_i][2], static_obstacles[static_i][3],
                            facecolor='k',
                            fill=True))
i = 0
count = int(steps_rgl / N)
if steps_rgl % N != 0:
    count += 1
count_c = 1
while i < steps_rgl:
    ax.add_artist(plt.Circle((robot_rgl[i][0], robot_rgl[i][1]), radius, fill=True, color='r', alpha=count_c / count))
    human_position = []
    for human_i in range(human_num_rgl):
        human_circle = plt.Circle(humans_rgl[i][human_i], radius, fill=True, color=human_color[human_i], alpha=count_c / count)
        human_position.append(np.array([humans_rgl[i][human_i][0], humans_rgl[i][human_i][1]]))
        ax.add_artist(human_circle)
    i = i + N
    count_c += 1
if i - N != steps_rgl - 1:
    ax.add_artist(plt.Circle((robot_rgl[steps_rgl - 1][0], robot_rgl[steps_rgl - 1][1]), radius, fill=True, color='r'))
    human_position = []
    for human_i in range(human_num_rgl):
        human_circle = plt.Circle(humans_rgl[steps_rgl - 1][human_i], radius, fill=True, color=human_color[human_i])
        human_position.append(np.array([humans_rgl[steps_rgl - 1][human_i][0], humans_rgl[steps_rgl - 1][human_i][1]]))
        ax.add_artist(human_circle)
################## rgl ######################

################## lndnl ######################
fig5, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.add_artist(plt.Circle(goal, radius, fill=True, color='g'))
static_obstacles = static_obstacles_lndnl[0]
static_obstacle_num = static_obstacles.shape[0]
for static_i in range(static_obstacle_num):
    ax.add_patch(Rectangle([static_obstacles[static_i][0], static_obstacles[static_i][1]], 
                            static_obstacles[static_i][2], static_obstacles[static_i][3],
                            facecolor='k',
                            fill=True))
i = 0
count = int(steps_lndnl / N)
if steps_lndnl % N != 0:
    count += 1
count_c = 1
while i < steps_lndnl:
    ax.add_artist(plt.Circle((robot_lndnl[i][0], robot_lndnl[i][1]), radius, fill=True, color='r', alpha=count_c / count))
    human_position = []
    for human_i in range(human_num_lndnl):
        human_circle = plt.Circle(humans_lndnl[i][human_i], radius, fill=True, color=human_color[human_i], alpha=count_c / count)
        human_position.append(np.array([humans_lndnl[i][human_i][0], humans_lndnl[i][human_i][1]]))
        ax.add_artist(human_circle)
    i = i + N
    count_c += 1
if i - N != steps_lndnl - 1:
    ax.add_artist(plt.Circle((robot_lndnl[steps_lndnl - 1][0], robot_lndnl[steps_lndnl - 1][1]), radius, fill=True, color='r'))
    human_position = []
    for human_i in range(human_num_lndnl):
        human_circle = plt.Circle(humans_lndnl[steps_lndnl - 1][human_i], radius, fill=True, color=human_color[human_i])
        human_position.append(np.array([humans_lndnl[steps_lndnl - 1][human_i][0], humans_lndnl[steps_lndnl - 1][human_i][1]]))
        ax.add_artist(human_circle)
################## lndnl ######################

################## dwa ######################
fig6, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-5.0, 5.0)
ax.set_ylim(-5.0, 5.0)
ax.add_artist(plt.Circle(goal, radius, fill=True, color='g'))
static_obstacles = static_obstacles_dwa[0]
static_obstacle_num = static_obstacles.shape[0]
for static_i in range(static_obstacle_num):
    ax.add_patch(Rectangle([static_obstacles[static_i][0], static_obstacles[static_i][1]], 
                            static_obstacles[static_i][2], static_obstacles[static_i][3],
                            facecolor='k',
                            fill=True))
i = 0
count = int(steps_dwa / N)
if steps_dwa % N != 0:
    count += 1
count_c = 1
while i < steps_dwa:
    ax.add_artist(plt.Circle((robot_dwa[i][0], robot_dwa[i][1]), radius, fill=True, color='r', alpha=count_c / count))
    human_position = []
    for human_i in range(human_num_dwa):
        human_circle = plt.Circle(humans_dwa[i][human_i], radius, fill=True, color=human_color[human_i], alpha=count_c / count)
        human_position.append(np.array([humans_dwa[i][human_i][0], humans_dwa[i][human_i][1]]))
        ax.add_artist(human_circle)
    i = i + N
    count_c += 1
if i - N != steps_dwa - 1:
    ax.add_artist(plt.Circle((robot_dwa[steps_dwa - 1][0], robot_dwa[steps_dwa - 1][1]), radius, fill=True, color='r'))
    human_position = []
    for human_i in range(human_num_dwa):
        human_circle = plt.Circle(humans_dwa[steps_dwa - 1][human_i], radius, fill=True, color=human_color[human_i])
        human_position.append(np.array([humans_dwa[steps_dwa - 1][human_i][0], humans_dwa[steps_dwa - 1][human_i][1]]))
        ax.add_artist(human_circle)
################## dwa ######################


plt.show()