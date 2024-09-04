import numpy as np
import matplotlib.pyplot as plt


ours_mujoco = np.loadtxt('training_data/ours_mujoco.txt')
ours_lip = np.loadtxt('training_data/ours_lip.txt')
drl_vo = np.loadtxt('training_data/drl_vo.txt')
rgl = np.loadtxt('training_data/gcn.txt')
lndnl = np.loadtxt('training_data/lndnl.txt')


# Use a different font, such as 'serif'
plt.rcParams['font.family'] = 'DejaVu Serif'

axis_x = np.linspace(1, 100, num=100)
axis_y = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
legend_font = {"size" : 16}


fig1, ax1 = plt.subplots(figsize=(10, 7.5))
ax1.set_xlabel('training steps (2e4)', fontsize=16)
ax1.set_ylabel('success rate', fontsize=16)
ax1.plot(axis_x, ours_mujoco, '-r', label='Ours_Full', linewidth=2)
ax1.plot(axis_x, ours_lip, '-c', label='Ours_LIP')
ax1.plot(axis_x, drl_vo, '-m', label='DRL_VO')
ax1.plot(axis_x, rgl, '-b', label='RGL')
ax1.plot(axis_x, lndnl, '-y', label='LNDNL')
# plt.xticks(axis_x, fontproperties = 'Times New Roman', size = 16)
plt.yticks(axis_y, size = 16)
plt.legend(ncol=2, prop=legend_font, bbox_to_anchor=(0.55, 0.15))

plt.show()