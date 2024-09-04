import numpy as np
from pathlib import Path
import pysocialforce as psf
# import matplotlib.pyplot as plt
import time

'''
robot_state, (px, py, vx, vy, gx, gy)
human_states: (px, py, vx, vy, gx, gy) x human_number
obstacles: (x1, x2, y1, y2) x obstacle number
'''
def sfm_humans(robot_state, human_states, obstacle_set, domain=False):
    # randomize sfm config
    user_config = {}
    user_config['scene'] = {
        'enable_group': True,
        'agent_radius': 0.3,
        'step_width': 0.2 + 0.02 * np.random.random() * (float(domain)),
        'max_speed_acc': 3.0 - 0.5 * np.random.random() * (float(domain)),
        'max_speed': 1.0 - 0.2 * np.random.random() * (float(domain))
    }

    user_config['desired_force'] = {
        'factor': 2.0,
        # The relaxation distance of the goal
        'goal_threshold': 0.2,
        # How long the relaxation process would take
        'relaxation_time': 0.5
    }

    user_config['social_force'] = {
        'factor': 5.1 + 3.0 * np.random.random() * (float(domain)),
        # Moussaid-Helbing 2009
        # relative importance of position vs velocity vector
        'lambda_importance': 2.0,
        # define speed interaction
        'gamma': 0.35,
        'n': 2,
        # define angular interaction
        'n_prime': 3,
    }

    user_config['obstacle_force'] = {
        'factor': 2.0 + np.random.random() * (float(domain)),
        # the standard deviation of obstacle force
        'sigma': 0.2,
        # threshold to trigger this force
        'threshold': 0.15 + 0.15 * np.random.random() * (float(domain))
    }

    user_config['group_coherence_force'] = {
        'factor': 3.0
    }
    

    user_config['group_repulsive_force'] = {
        'factor': 1.0,
        # threshold to trigger this force
        'threshold': 0.55
    }

    user_config['group_gaze_force'] = {
        'factor': 4.0,
        # fielf of view
        'fov_phi': 90.0
    }
   
    human_robot_states = np.vstack((human_states, robot_state))

    sim = psf.Simulator(
          human_robot_states,
          obstacles=obstacle_set,
          sfm_config=user_config,
    )
    sim.step_once()
    next_states, next_group_states = sim.get_states()
    return next_states[-1]

