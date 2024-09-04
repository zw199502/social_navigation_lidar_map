import logging
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.patches import Rectangle
import numpy as np

from numpy.linalg import norm
from utils.human import Human
from utils.robot import Robot
from utils.state import ObservableState
from policy.policy_factory import policy_factory
from info import *
from math import atan2, hypot, sqrt, cos, sin, fabs, inf, ceil
from time import sleep, time
from C_library.motion_plan_lib import *
import threading
from collections import deque

class CrowdSim:
    def __init__(self, args, action_range, digit_dim, 
                 action_choices=None, digit_env=None):
        self.n_laser = args.lidar_dim
        self.laser_angle_resolute = args.laser_angle_resolute
        self.laser_min_range = args.laser_min_range
        self.laser_max_range = args.laser_max_range
        self.square_width = args.square_width
        self.human_policy_name = 'orca' # human policy is fixed orca policy
        self.robot_policy = args.policy
        self.robot_model = args.robot_model
        self.robot_test_model = args.robot_test_model
        self.robot_goal_state_dim = args.robot_goal_state_dim
        
        # last-time distance from the robot to the goal
        self.goal_distance_last = None

        
        # scan_intersection, each line connects the robot and the end of each laser beam
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32) # used for visualization

        # laser state
        self.scan_current = self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        
        self.global_time = 0.0
        self.global_step = 0
        self.time_limit = 50
        self.time_step = 0.4
        self.y_range = 7.0
        self.x_range = 7.0
        self.v_min = 0.1
        self.max_episode_step = int(self.time_limit / self.time_step)
        self.randomize_attributes = False
        self.success_reward = 0.5
        self.collision_penalty = -0.5
        self.discomfort_dist = 0.5
        self.discomfort_penalty_factor = 0.4
        self.goal_distance_factor = 1.4
        # self.digit_reward_factor = 0.2 # with torque penalty
        self.digit_reward_factor = 1.0 # without torque penalty
        self.digit_crazy_penalty = -0.5  
        self.angular_penalty = -0.1

        # here, more lines can be added to simulate obstacles
        self.lines = None
        self.circle_radius = 4.0 # human distribution margin
        self.static_obstacle_area_x = 3.0 # static obstacle distribution area
        self.static_obstacle_area_y = 1.5 
        self.human_num_max = 4
        self.static_obstacle_num_max = 3

        self.human_num = None
        self.static_obstacle_num = None


        self.humans = None
        self.human_v_pref = 1.0
        self.static_obstacles = None
        self.rectangles = None
        self.action_range = action_range
        self.action_choices = action_choices
        self.robot = Robot()
        self.robot.time_step = self.time_step
        self.robot.v_pref = action_range[1, 0]
        self.action_last = np.zeros(2)
        self.acceleration = [1.0, 1.0]

        self.memory_lock = threading.Lock()

        # dwa parameters
        self.acc_linear_max = self.acceleration[0]
        self.acc_angular_max = self.acceleration[1]
        self.dwa_resolution_linear_v = 0.02
        self.dwa_resolution_angular_v = 0.02
        self.dwa_look_forward_steps = 5
        self.dwa_dist_goal_cost = 0.4
        self.delta_linear_v_max = self.acc_linear_max * self.time_step
        self.delta_angular_v_max = self.acc_angular_max * self.time_step
        self.delta_linear_v = 0.05
        self.delta_angular_v = 0.05

        # LIPM
        self.w = np.sqrt(9.81/1.02)
        self.cosh_wt = np.cosh(self.w * self.time_step)
        self.sinh_wt = np.sinh(self.w * self.time_step)
        
        # mujoco digit model
        self.digit_env = digit_env
        self.digit_dim = digit_dim
        self.mujoco_visualize = False
        
        # lidar to image
        self.frame_stack = args.frame_stack
        self.frames = deque([], maxlen=self.frame_stack)
        self.image_size = args.image_size
        self.single_frame = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        self.r_resolution = self.laser_max_range / self.image_size
        self.theta_resolution = self.n_laser / self.image_size

        # visualization on 2D plane
        plt.ion()
        plt.show()
        self.fig, self.ax = plt.subplots(figsize=(7, 7))

        # log lidar, robot, and humans
        self.log_env = {}

    def get_all_obstacles(self):
        self.static_obstacle_num = 0
        self.human_num = 0
        self.humans = []
        self.lines = []
        margin = 35.0  
        self.lines.append(np.array([[-margin, -margin],
                                    [-margin,  margin]], dtype=np.float32))
        self.lines.append(np.array([[-margin, margin],
                                    [margin,  margin]], dtype=np.float32))
        self.lines.append(np.array([[margin,  margin],
                                    [margin, -margin]], dtype=np.float32))
        self.lines.append(np.array([[margin, -margin],
                                    [-margin, -margin]], dtype=np.float32))
        for i in range(self.digit_env.markers.shape[0]):
            if fabs(self.digit_env.markers[i][0]) > 50:
                break
            if fabs(self.digit_env.markers[i][5]) > 0.6:
                self.human_num += 1
                human = Human()
                human.time_step = self.time_step
                human.radius = 0.3
                human.v_pref = self.human_v_pref
                human.set(self.digit_env.markers[i][0], self.digit_env.markers[i][1], self.square_width, self.square_width, 0, 0, 0)
                self.humans.append(human)
            else:
                self.static_obstacle_num += 1
                position_x = self.digit_env.markers[i][0] - self.digit_env.markers[i][3] / 2.0
                position_y = self.digit_env.markers[i][1] - self.digit_env.markers[i][4] / 2.0
                self.lines.append(np.array([[position_x, position_y],
                                            [position_x + self.digit_env.markers[i][3], position_y]], dtype=np.float32))
                self.lines.append(np.array([[position_x, position_y],
                                            [position_x, position_y + self.digit_env.markers[i][4]]], dtype=np.float32))
                self.lines.append(np.array([[position_x + self.digit_env.markers[i][3], position_y],
                                            [position_x + self.digit_env.markers[i][3], position_y + self.digit_env.markers[i][4]]], dtype=np.float32))
                self.lines.append(np.array([[position_x + self.digit_env.markers[i][3], position_y + self.digit_env.markers[i][4]],
                                            [position_x, position_y + self.digit_env.markers[i][4]]], dtype=np.float32))
        self.lines = np.array(self.lines)

    def generate_random_static_obstacle(self):
        self.static_obstacles = {}
        self.static_obstacle_num = int(np.random.randint(self.static_obstacle_num_max, size=1)[0] + 1)
        # rectangles, 2 center position, 2 side length
        self.rectangles = np.zeros((self.static_obstacle_num, 4), dtype=np.float32)
        # here, more lines can be added to simulate obstacles
        # margin = square_width / 2.0
        margin = 35.0  
        self.lines = np.zeros((4 + self.static_obstacle_num * 4, 2, 2), dtype=np.float32)
        self.lines[0, :, :] = np.array([[-margin, -margin],
                                        [-margin,  margin]], dtype=np.float32) 
        self.lines[1, :, :] = np.array([[-margin,  margin],
                                        [margin,  margin]], dtype=np.float32) 
        self.lines[2, :, :] = np.array([[margin,  margin],
                                        [margin, -margin]], dtype=np.float32) 
        self.lines[3, :, :] = np.array([[margin, -margin],
                                        [-margin, -margin]], dtype=np.float32) 
     
        while True:
            # positions = np.random.uniform(-self.static_obstacle_area, self.static_obstacle_area, 
            #                             (self.static_obstacle_num, 2))
            positions_x = np.random.uniform(-self.static_obstacle_area_x, self.static_obstacle_area_x, 
                                            (self.static_obstacle_num, 1))
            positions_y = np.random.uniform(-self.static_obstacle_area_y, self.static_obstacle_area_y, 
                                            (self.static_obstacle_num, 1))
            positions = np.hstack((positions_x, positions_y))
            shapes = np.random.uniform(0.3, 0.4, (self.static_obstacle_num, 2))
            collision = False
            for i in range(self.static_obstacle_num):
                temp = False
                for j in range(i+1, self.static_obstacle_num):
                    if hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]) <= 0.5:
                        collision = True
                        temp = True
                        break
                if temp:
                    break
            if not collision:
                centers = positions + shapes / 2.0
                radiuses = np.sqrt(np.sum(np.square(shapes), axis=1)) / 2.0
                self.static_obstacles['positions'] = positions
                self.static_obstacles['shapes'] = shapes
                self.static_obstacles['centers'] = centers
                self.static_obstacles['radiuses'] = radiuses
                self.rectangles = np.hstack((positions, shapes))
                # add lines
                for k in range(self.static_obstacle_num):

                    self.lines[0 + 4 * (k + 1), :, :] = np.array([[positions[k][0], positions[k][1]],
                                                                  [positions[k][0] + shapes[k][0], positions[k][1]]], dtype=np.float32) 
                    self.lines[1 + 4 * (k + 1), :, :] = np.array([[positions[k][0], positions[k][1]],
                                                                  [positions[k][0], positions[k][1] + shapes[k][1]]], dtype=np.float32) 
                    self.lines[2 + 4 * (k + 1), :, :] = np.array([[positions[k][0] + shapes[k][0], positions[k][1]],
                                                                  [positions[k][0] + shapes[k][0], positions[k][1] + shapes[k][1]]], dtype=np.float32) 
                    self.lines[3 + 4 * (k + 1), :, :] = np.array([[positions[k][0], positions[k][1] + shapes[k][1]],
                                                                  [positions[k][0] + shapes[k][0], positions[k][1] + shapes[k][1]]], dtype=np.float32) 
                    # print(self.lines)
                break

        
    def generate_random_human_position(self):
        self.human_num = int(np.random.randint(self.human_num_max, size=1)[0] + 1)
        self.humans = [None] * self.human_num
        for i in range(self.human_num):
            self.humans[i] = self.generate_circle_crossing_human()

        for i in range(len(self.humans)):
            human_policy = policy_factory[self.human_policy_name]()
            human_policy.time_step = self.time_step
            human_policy.max_speed = self.humans[i].v_pref
            human_policy.radius = self.humans[i].radius
            human_policy.max_robot_speed = self.robot.v_pref
            self.humans[i].set_policy(human_policy)

    def generate_circle_crossing_human(self):
        if self.static_obstacles is None:
            raise NotImplementedError(self.static_obstacles)
        human = Human()
        human.time_step = self.time_step

        if self.randomize_attributes:
            # Sample agent radius and v_pref attribute from certain distribution
            human.sample_random_attributes()
        else:
            human.radius = 0.3
            human.v_pref = self.human_v_pref
        while True:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (np.random.random() - 0.5) * human.v_pref
            py_noise = (np.random.random() - 0.5) * human.v_pref
            px = self.circle_radius * np.cos(angle) + px_noise
            py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for agent in [self.robot] + self.humans:
                if agent is None:
                    continue
                min_dist = human.radius + agent.radius + self.discomfort_dist
                if norm((px - agent.px, py - agent.py)) < min_dist or \
                        norm((px - agent.gx, py - agent.gy)) < min_dist:
                    collide = True
                    break
            for static_obs in range(self.static_obstacle_num):
                min_dist = human.radius + self.static_obstacles['radiuses'][static_obs] + self.discomfort_dist
                if norm((px - self.static_obstacles['centers'][static_obs][0], 
                         py - self.static_obstacles['centers'][static_obs][1])) < min_dist:
                    collide = True
                    break
            if not collide:
                break
        # px, py, gx, gy, vx, vy, theta
        human.set(px, py, -px, -py, 0, 0, 0)
        return human
        
    '''use c library to generate lidar scan, which is fast'''
    def get_lidar(self):
        scan = np.zeros(self.n_laser, dtype=np.float32)
        # robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta])
        robot_pose = np.array([self.robot.px, self.robot.py, self.robot.theta], dtype=np.float32)
        num_line = self.lines.shape[0]
        num_circle = self.human_num
        InitializeEnv(num_line, num_circle, self.n_laser, self.laser_angle_resolute)
        for i in range (num_line):
            set_lines(4 * i    , self.lines[i][0][0])
            set_lines(4 * i + 1, self.lines[i][0][1])
            set_lines(4 * i + 2, self.lines[i][1][0])
            set_lines(4 * i + 3, self.lines[i][1][1])
        for i in range (num_circle):
            set_circles(3 * i    , self.humans[i].px)
            set_circles(3 * i + 1, self.humans[i].py)
            set_circles(3 * i + 2, self.humans[i].radius)
        set_robot_pose(robot_pose[0], robot_pose[1], robot_pose[2])
        cal_laser()
        self.scan_intersection = np.zeros((self.n_laser, 2, 2), dtype=np.float32)
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            ### used for visualization
            self.scan_intersection[i, 0, 0] = self.robot.px
            self.scan_intersection[i, 0, 1] = self.robot.py
            self.scan_intersection[i, 1, 0] = get_scan_line(4 * i + 2)
            self.scan_intersection[i, 1, 1] = get_scan_line(4 * i + 3)
            ### used for visualization
        
        self.scan_current = np.clip(scan, self.laser_min_range, self.laser_max_range)
        ReleaseEnv()
        
        self.single_frame = np.zeros((1, self.image_size, self.image_size), dtype=np.uint8)
        for i in range(self.n_laser):
            if self.scan_current[i] == self.laser_max_range:
                continue
            j = int(i / self.theta_resolution)
            if j >= self.image_size:
                j = self.image_size - 1
            k = int(self.scan_current[i] / self.r_resolution)
            if k >= self.image_size:
                k = self.image_size - 1
            self.single_frame[0, j, k] = 255
    
    '''use python to generate lidar scan, which is far slow and not recommended'''
    def get_lidar_python(self):
        if self.human_num > 0:
            circle_flag = True
        else:
            circle_flag = False

        line_vector_1 = np.zeros(2, dtype=np.float32)
        line_vector_2 = np.zeros(2, dtype=np.float32)
        scans = self.laser_max_range * np.ones(self.n_laser, dtype=np.float32)
        robot_pose = np.array([self.robot.px, self.robot.py, 0.0], dtype=np.float32)
        line_vector = np.zeros(2, dtype=np.float32)
        line_unit_vector = np.zeros(2, dtype=np.float32)
        for i in range(self.n_laser):
            min_range = 999.9
            angle_rel = (float(i) - (float(self.n_laser) - 1.0) / 2.0) * self.laser_angle_resolute
            # laser angle range: [-pi, pi]
            if angle_rel < -np.pi:
                angle_rel = -np.pi
            if angle_rel > np.pi:
                angle_rel = np.pi
            angle_abs = angle_rel + robot_pose[2]
            line_unit_vector[0] = cos(angle_abs)
            line_unit_vector[1] = sin(angle_abs)
            # a1,b1,c1 are the parameters of line
            a1 = line_unit_vector[1]
            b1 = -line_unit_vector[0]
            c1 = robot_pose[1] * line_unit_vector[0] - robot_pose[0] * line_unit_vector[1]
            _intersection_x = 0.0
            _intersection_y = 0.0

            for j in range(self.lines.shape[0]):
                x0 = self.lines[j][0][0]
                y0 = self.lines[j][0][1]
                x1 = self.lines[j][1][0]
                y1 = self.lines[j][1][1]
                f0 = a1 * x0 + b1 * y0 + c1
                f1 = a1 * x1 + b1 * y1 + c1
                if f0 * f1 > 0: # the two points are located on the same side
                    continue
                else:
                    dx = x1 - x0
                    dy = y1 - y0
                    a2 = dy
                    b2 = -dx
                    c2 = y0 * dx - x0 * dy
                    if fabs(a1 * b2 - a2 * b1) < 0.00001:
                        continue
                    intersection_x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
                    intersection_y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
                    # intersection is always in front of the robot
                    line_vector[0] = intersection_x - robot_pose[0]
                    line_vector[1] = intersection_y - robot_pose[1]
                    # the intersection point must be located in the one direction of the line
                    if (line_vector[0] * line_unit_vector[0] > 0) or (line_vector[1] * line_unit_vector[1] > 0):
                        scan_range = hypot(intersection_x - robot_pose[0], intersection_y - robot_pose[1])
                        if scan_range < min_range:
                            _intersection_x = intersection_x
                            _intersection_y = intersection_y
                            min_range = scan_range

            if circle_flag:
                for k in range(self.human_num):
                    x2 = self.humans[k].px
                    y2 = self.humans[k].py
                    r = self.humans[k].radius
                    f2 = a1 * x2 + b1 * y2 + c1
                    
                    d = fabs(f2) / hypot(a1, b1)
                    if d > r:
                        continue
                    else:
                        a3 = b1
                        b3 = - a1
                        c3 = -(a3 * x2 + b3 * y2)
                        if fabs(a1 * b3 - a3 * b1) < 0.00001:
                            continue
                        intermediate_x = (b1 * c3 - b3 * c1) / (a1 * b3 - a3 * b1)
                        intermediate_y = (a3 * c1 - a1 * c3) / (a1 * b3 - a3 * b1)
                        if d == r:
                            _intersection_x = intermediate_x
                            _intersection_y = intermediate_y
                        else:
                            if d >= 0 and d < r:
                                temp = r * r - d * d
                                if temp < 0.0 or temp > 999.0:
                                    continue
                                l_vector = sqrt(temp)
                                intersection_x_1 = l_vector * line_unit_vector[0] + intermediate_x
                                intersection_y_1 = l_vector * line_unit_vector[1] + intermediate_y
                                intersection_x_2 = -l_vector * line_unit_vector[0] + intermediate_x
                                intersection_y_2 = -l_vector * line_unit_vector[1] + intermediate_y
                                line_vector_1[0] = intersection_x_1 - robot_pose[0]
                                line_vector_1[1] = intersection_y_1 - robot_pose[1]
                                # the intersection point must be located in the one direction of the line
                                if (line_vector_1[0] * line_unit_vector[0]) > 0 or (line_vector_1[1] * line_unit_vector[1] > 0):
                                    scan_range = hypot(intersection_x_1 - robot_pose[0], intersection_y_1 - robot_pose[1])
                                    if scan_range < min_range:
                                        min_range = scan_range
                                        _intersection_x = intersection_x_1
                                        _intersection_y = intersection_y_1
                                line_vector_2[0] = intersection_x_2 - robot_pose[0]
                                line_vector_2[1] = intersection_y_2 - robot_pose[1]
                                if (line_vector_2[0] * line_unit_vector[0]) > 0 or (line_vector_2[1] * line_unit_vector[1] > 0):
                                    scan_range = hypot(intersection_x_2 - robot_pose[0], intersection_y_2 - robot_pose[1])
                                    if scan_range < min_range:
                                        min_range = scan_range
                                        _intersection_x = intersection_x_2
                                        _intersection_y = intersection_y_2

            scans[i] = min_range
            self.scan_intersection[i, 0, 0] = self.robot.px
            self.scan_intersection[i, 0, 1] = self.robot.py
            self.scan_intersection[i, 1, 0] = _intersection_x
            self.scan_intersection[i, 1, 1] = _intersection_y
        self.scan_current = np.clip(scans, self.laser_min_range, self.laser_max_range)

    def cal_dwa_action(self):
        # dwa calculate action
        action_cost = 99999.9
        action_dwa = np.zeros(2)
        dwa_horizon = 5
        for i in range(self.action_choices.shape[0]):
            robot_vel = self.action_choices[i]
            
            robot_x = self.robot.px
            robot_y = self.robot.py
            robot_theta = self.robot.theta
            collision = False
            dis_human_and_obstacle = 99999.9
            robot_theta = robot_theta + robot_vel[1] * self.time_step
            if robot_theta > np.pi:
                robot_theta -= (2.0 * np.pi)
            elif robot_theta < -np.pi:
                robot_theta += (2.0 * np.pi)
            x_d = self.action_last[0]
            for j in range(dwa_horizon):
                # differential model
                # robot_x = robot_x + robot_vel[0] * self.time_step * cos(robot_theta)
                # robot_y = robot_y + robot_vel[0] * self.time_step * sin(robot_theta)
                # differential model

                # LIP model
                pf_x = (x_d * self.cosh_wt - robot_vel[0]) / (self.w * self.sinh_wt)
                x_n =  pf_x - pf_x * self.cosh_wt + x_d * self.sinh_wt / self.w
                x_d = robot_vel[0]
                robot_x = robot_x + x_n * cos(robot_theta)
                robot_y = robot_y + x_n * sin(robot_theta)
                # LIP model

                # distance to humans
                for k in range(self.human_num):
                    human_x = self.humans[k].px + (j + 1) * self.humans[k].vx * self.time_step
                    human_y = self.humans[k].py + (j + 1) * self.humans[k].vy * self.time_step
                    dis_human_temp = hypot(human_x - robot_x, human_y - robot_y)
                    if dis_human_temp <= self.humans[k].radius + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_human_temp)

                if collision:
                    break

                # distance to obstacles
                for k in range(self.static_obstacle_num):
                    obstacle_x = self.static_obstacles['centers'][k][0]
                    obstacle_y = self.static_obstacles['centers'][k][1]
                    dis_obstacle_temp = hypot(obstacle_x - robot_x, obstacle_y - robot_y)
                    if dis_obstacle_temp <= self.static_obstacles['radiuses'][k] + self.robot.radius:
                        collision = True
                        break
                    dis_human_and_obstacle = min(dis_human_and_obstacle, dis_obstacle_temp)

                if collision:
                    break

            if collision:
                continue
            dis_goal = hypot(self.robot.gx - robot_x, self.robot.gy - robot_y)
         
            action_cost_temp = 1.0 / (dis_human_and_obstacle + 0.8) + dis_goal * 0.2
            if action_cost > action_cost_temp:
                action_cost = action_cost_temp
                action_dwa = robot_vel
        return action_dwa
        
    def step(self, action, eval=False, save_data=False):
        if self.robot_test_model == 'digit_hardware_real_obstacle':
            self.get_all_obstacles()
        else:
            human_actions = np.zeros((self.human_num, 2), dtype=np.float32)
            for i in range(self.human_num):
                # observation for humans is always coordinates
                ob = [other_human.get_observable_state() for other_human in self.humans if other_human != self.humans[i]]
                for k in range(self.static_obstacle_num):
                    ob.append(ObservableState(
                            self.static_obstacles['centers'][k][0], 
                            self.static_obstacles['centers'][k][1], 
                            0.0, 0.0, self.static_obstacles['radiuses'][k])
                            )
                if 1.2 * hypot(self.robot.vx, self.robot.vy) < hypot(self.humans[i].vx, self.humans[i].vy):
                    ob.append(ObservableState(self.robot.px, self.robot.py, self.robot.vx, self.robot.vy, self.robot.radius))
                    action_temp = self.humans[i].act(ob, has_robot=True)
                    human_actions[i] = np.array([action_temp[0], action_temp[1]], dtype=np.float32)
                else:
                    action_temp = self.humans[i].act(ob)
                    human_actions[i] = np.array([action_temp[0], action_temp[1]], dtype=np.float32)

        # update robot states
        digit_penalty = 1e5
        crazy_digit = False
        action_copy = np.array([action[0], action[1]])
        if self.digit_env is None:
            pf_x = (self.action_last[0] * self.cosh_wt - action[0]) / (self.w * self.sinh_wt)
            x_n =  pf_x - pf_x * self.cosh_wt + self.action_last[0] * self.sinh_wt / self.w
            robot_theta = self.robot.theta + action[1] * self.time_step
            if robot_theta > np.pi:
                robot_theta -= (2.0 * np.pi)
            elif robot_theta < -np.pi:
                robot_theta += (2.0 * np.pi)
            robot_x = self.robot.px + x_n * cos(robot_theta)
            robot_y = self.robot.py + x_n * sin(robot_theta)
        else:
            vel_command_to_digit = {
                'x_vel': action[0],
                'y_vel': 0.0,
                'yaw_vel': action[1]
            }
            self.digit_env.set_vel_command(vel_command_to_digit)
            if self.robot_test_model == 'digit_arsim' or self.robot_test_model == 'digit_hardware' \
               or self.robot_test_model == 'digit_hardware_real_obstacle':
                while True:
                    sleep(self.digit_env.cfg.control.control_dt)
                    if self.digit_env.get_domain_switch():
                        break
                crazy_digit = False
                digit_penalty = 0.0
                self.digit_env.get_robot_states()
            else:
                small_t = 0
                while True:
                    st_time = time()
                    digit_env_step = self.digit_env.step(np.zeros(12))
                    reward_info = digit_env_step['env_info']['reward_info']
                    # digit_penalty = min(digit_penalty, 
                    #                     reward_info['z_vel_penalty'] 
                    #                     + reward_info['roll_pitch_penalty'] 
                    #                     + reward_info['torque_penalty'])
                    digit_penalty = min(digit_penalty, 
                                        reward_info['z_vel_penalty'] 
                                        + reward_info['roll_pitch_penalty'] 
                                    )
                    if digit_env_step['env_info']['crazy_digit']:
                        crazy_digit = True
                        break
                    if self.mujoco_visualize:
                        end_time = time()
                        if (end_time - st_time) < self.digit_env.cfg.control.control_dt:
                            sleep(self.digit_env.cfg.control.control_dt - (end_time - st_time))
                    small_t += 1
                    if self.digit_env.domain_switch:
                        # print('small_t: ', small_t)
                        break
            robot_x = self.digit_env.root_xy_pos[0]
            robot_y = self.digit_env.root_xy_pos[1]
            robot_theta = self.digit_env.root_rpy[2]
            action_copy[0] = hypot(robot_y - self.robot.py, robot_x - self.robot.px) / self.time_step
            
        # update states
        self.robot.update_states(robot_x, robot_y, robot_theta, action_copy, differential=True)

        if self.robot_test_model == 'digit_hardware_real_obstacle':
            self.get_all_obstacles()
        else:
            for i in range(self.human_num):
                self.humans[i].update_states(human_actions[i])

        # get new laser scan and grid map
        self.get_lidar() 
        
        self.frames.append(self.single_frame)
        assert len(self.frames) == self.frame_stack
        lidar_image = np.concatenate(list(self.frames), axis=0)
        
        self.global_time += self.time_step
        
        # if reaching goal
        goal_dist = hypot(robot_x - self.robot.gx, robot_y - self.robot.gy)
        if eval:
            reaching_goal = goal_dist < (self.robot.radius + 0.2)
        else:
            reaching_goal = goal_dist < (self.robot.radius + 0.1)

        # collision detection between the robot and humans
        collision = False
        dmin = self.scan_current.min()
        if dmin <= self.robot.radius:
            collision = True
            
        out_area = fabs(self.robot.py) > self.y_range or fabs(self.robot.px) > self.x_range
        reward = 0
        
        if ((dmin - self.robot.radius) < self.discomfort_dist):
            dis_obstacle_reward = (dmin - self.robot.radius - self.discomfort_dist) * self.discomfort_penalty_factor
        else:
            dis_obstacle_reward = 0.0
            
        dis_goal_reward = self.goal_distance_factor * (self.goal_distance_last - goal_dist)
        # dis_goal_reward = 0.0
        self.goal_distance_last = goal_dist
        
        # angular_reward = fabs(action[1]) * self.angular_penalty
        angular_reward = 0.0
        
        if self.digit_env is not None:
            digit_reward = self.digit_reward_factor * digit_penalty
        else:
            digit_reward = 0.0
        
        self.action_last = action
        reward = dis_obstacle_reward + dis_goal_reward \
                 + angular_reward + digit_reward 
        if collision or out_area:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif crazy_digit:
            reward = self.digit_crazy_penalty
            done = True
            info = DigitCrazy()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif ((dmin - self.robot.radius) < self.discomfort_dist):
            done = False
            info = Danger(dmin)
        else:
            done = False
            info = Nothing()
  
        for i, human in enumerate(self.humans):
            # let humans move circularly from two points
            if human.reached_destination():
                self.humans[i].gx = -self.humans[i].gx
                self.humans[i].gy = -self.humans[i].gy

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)

        # get the observation
        robot_goal_state = np.array([r / self.square_width, t / np.pi, 
                                     self.action_last[0] / self.action_range[1, 0],
                                     self.action_last[1] / self.action_range[1, 1]], dtype=np.float32)
        if self.digit_dim > 0 and (self.digit_env is not None):
            robot_goal_state = np.concatenate([robot_goal_state, self.digit_env.actor_obs]).astype(np.float32).flatten()
        
        if save_data:
            self.global_step += 1
            if self.digit_env is not None and self.robot_test_model == 'digit_mujoco':
                self.log_env['ypr'][self.global_step] = self.digit_env.root_ang_vel
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, action[0], action[1], self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_position = np.zeros((self.human_num, 2), dtype=np.float32)
            for i in range(self.human_num):
                humans_position[i] = np.array([self.humans[i].px, self.humans[i].py], dtype=np.float32)
            self.log_env['humans'][self.global_step] = humans_position
            static_obstacles_info = np.zeros((self.static_obstacle_num, 4), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles['positions'][i][0], 
                                                    self.static_obstacles['positions'][i][1],
                                                    self.static_obstacles['shapes'][i][0],  
                                                    self.static_obstacles['shapes'][i][1]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers

        if self.robot_policy == 'td3_lstm':
            return self.scan_current / self.laser_max_range, robot_goal_state, reward, done, info
        else:
            return lidar_image, robot_goal_state, reward, done, info
    
    def save_video(self, steps, episodes):
        filename = 'eval_' + str(steps) + '_' + str(episodes)
        if self.digit_env is None:
            raise NotImplementedError(self.digit_env)
        self.digit_env.save_video(filename)
    
    def reset(self, eval=False, save_data=False):
        self.global_time = 0.0
        self.global_step = 0
        self.action_last = np.zeros(2)
        self.static_obstacles = None
        self.log_env = {}
        # px, py, gx, gy, vx, vy, theta
        self.robot.set(-self.circle_radius, 0.0, self.circle_radius, 0.0, 0.0, 0.0, 0.0)
        
        if self.digit_env is not None:    
            if self.robot_test_model == 'digit_mujoco':
                # for initializing
                self.digit_env.reset(robot=np.array([self.robot.px, self.robot.py], dtype=np.float32))
                sleep(self.digit_env.cfg.control.control_dt)
                # initialize the locomotion for 2 seconds to let the robot step in place
                for i in range(int(2.0 / self.digit_env.cfg.control.control_dt)):
                    st_time = time()
                    self.digit_env.step(np.zeros(12))
                    if self.mujoco_visualize:
                        end_time = time()
                        if (end_time - st_time) < self.digit_env.cfg.control.control_dt:
                            sleep(self.digit_env.cfg.control.control_dt - (end_time - st_time))
                # make sure the command sending moment is the contact switching 
                while True:
                    st_time = time()
                    self.digit_env.step(np.zeros(12))
                    if self.mujoco_visualize:
                        end_time = time()
                        if (end_time - st_time) < self.digit_env.cfg.control.control_dt:
                            sleep(self.digit_env.cfg.control.control_dt - (end_time - st_time))
                    if self.digit_env.domain_switch:
                        break
            elif self.robot_test_model == 'digit_arsim' or self.robot_test_model == 'digit_hardware' \
                 or self.robot_test_model == 'digit_hardware_real_obstacle':
                sleep(20.0)
                while True:
                    sleep(self.digit_env.cfg.control.control_dt)
                    if self.digit_env.get_domain_switch():
                        break
                self.digit_env.get_robot_states()
            robot_x = self.digit_env.root_xy_pos[0]
            robot_y = self.digit_env.root_xy_pos[1]
            robot_theta = self.digit_env.root_rpy[2] 
            # update states
            self.robot.update_states(robot_x, robot_y, robot_theta, np.zeros(2), differential=True)
        
        self.goal_distance_last = self.robot.get_goal_distance()

        # 3,5 save
        # np.random.seed(5)
        
        if self.robot_test_model == 'digit_hardware_real_obstacle':
            self.get_all_obstacles()
        else:
            self.generate_random_static_obstacle()

            self.generate_random_human_position()

        self.get_lidar() 
        
        for _ in range(self.frame_stack):
            self.frames.append(self.single_frame)
        assert len(self.frames) == self.frame_stack
        lidar_image = np.concatenate(list(self.frames), axis=0)

        # get the observation

        dx = self.robot.gx - self.robot.px
        dy = self.robot.gy - self.robot.py
        theta = self.robot.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel)
        t = atan2(y_rel, x_rel)
        
        robot_goal_state = np.array([r / self.square_width, t / np.pi, 
                                     self.action_last[0] / self.action_range[1, 0],
                                     self.action_last[1] / self.action_range[1, 1]], dtype=np.float32)
        if self.digit_dim > 0 and (self.digit_env is not None):
            robot_goal_state = np.concatenate([robot_goal_state, self.digit_env.actor_obs]).astype(np.float32).flatten()
            
        if save_data:
            self.log_env['ypr'] = -100.0 * np.ones((self.max_episode_step + 1, 3), dtype=np.float32)
            self.log_env['robot'] = -100.0 * np.ones((self.max_episode_step + 1, 5), dtype=np.float32)
            self.log_env['goal'] =  -100.0 * np.ones((self.max_episode_step + 1, 2), dtype=np.float32)
            self.log_env['humans'] = -100.0 * np.ones((self.max_episode_step + 1, self.human_num, 2), dtype=np.float32)
            self.log_env['static_obstacles'] = -100.0 * np.ones((self.max_episode_step + 1, self.static_obstacle_num, 4), dtype=np.float32)
            self.log_env['laser'] = -100.0 * np.ones((self.max_episode_step + 1, self.n_laser, 4), dtype=np.float32)

            if self.digit_env is not None and self.robot_test_model == 'digit_mujoco':
                self.log_env['ypr'][self.global_step] = self.digit_env.root_ang_vel
            self.log_env['robot'][self.global_step] = np.array([self.robot.px, self.robot.py, 0.0, 0.0, self.robot.theta])
            self.log_env['goal'][self.global_step] = np.array([self.robot.gx, self.robot.gy])
            humans_position = np.zeros((self.human_num, 2), dtype=np.float32)
            for i in range(self.human_num):
                humans_position[i] = np.array([self.humans[i].px, self.humans[i].py], dtype=np.float32)
            self.log_env['humans'][self.global_step] = humans_position
            static_obstacles_info = np.zeros((self.static_obstacle_num, 4), dtype=np.float32)
            for i in range (self.static_obstacle_num):
                static_obstacles_info[i] = np.array([self.static_obstacles['positions'][i][0], 
                                                    self.static_obstacles['positions'][i][1],
                                                    self.static_obstacles['shapes'][i][0],  
                                                    self.static_obstacles['shapes'][i][1]])
            self.log_env['static_obstacles'][self.global_step] = static_obstacles_info
            lasers = np.zeros((self.n_laser, 4), dtype=np.float32)
            for i in range(self.n_laser):
                laser = self.scan_intersection[i]
                lasers[i] = np.array([laser[0][0], laser[0][1], laser[1][0], laser[1][1]], dtype=np.float32)
            self.log_env['laser'][self.global_step] = lasers

        if self.robot_policy == 'td3_lstm':
            return self.scan_current / self.laser_max_range, robot_goal_state
        else:
            return lidar_image, robot_goal_state

    def render(self, mode='laser'):
        if mode == 'laser':
            self.ax.set_xlim(-5.0, 5.0)
            self.ax.set_ylim(-5.0, 5.0)
            for human in self.humans:
                human_circle = plt.Circle(human.get_position(), human.radius, fill=False, color='b')
                self.ax.add_artist(human_circle)
            self.ax.add_artist(plt.Circle(self.robot.get_position(), self.robot.radius, fill=True, color='r'))
            for i in range(self.static_obstacle_num):
                self.ax.add_patch(Rectangle(self.rectangles[i, :2], 
                                            self.rectangles[i, 2], self.rectangles[i, 3],
                                            facecolor='green',
                                            fill=True))
            plt.text(-4.5, -4.5, str(round(self.global_time, 2)), fontsize=20)
            x, y, theta = self.robot.px, self.robot.py, self.robot.theta
            dx = cos(theta)
            dy = sin(theta)
            self.ax.arrow(x, y, dx, dy,
                width=0.01,
                length_includes_head=True, 
                head_width=0.15,
                head_length=1,
                fc='r',
                ec='r')
            ii = 0
            lines = []
            while ii < self.n_laser:
                lines.append(self.scan_intersection[ii])
                ii = ii + 36
            lc = mc.LineCollection(lines)
            self.ax.add_collection(lc)
            plt.draw()
            plt.pause(0.001)
            plt.cla()


