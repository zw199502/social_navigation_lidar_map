Social navigation with SAC_AE algorithm

# environment
- Ubuntu 20.04
- NVIDIA GeForce RTX 4090
- Driver Version: 535.171.04   CUDA Version: 12.2
- Anaconda install
- Create Anaconda env, ```conda create -n torch python=3.10```
- ```conda activate torch```
- Install [pytorch](https://pytorch.org/get-started/locally/)

# c/c++ to compile related libraries
- ```sudo apt update```
- ```sudo apt install cmake build-essential```
- if error like depends libc6-dev and g++, change ubuntu source to [tsinghua](https://mirror.tuna.tsinghua.edu.cn/help/ubuntu/), ```sudo apt install libc6=2.31-0ubuntu9.15```, then ```sudo apt install cmake build-essential```.

# simulation with simplified LIP model and digit_mujoco
- ```git clone https://github.com/sybrenstuvel/Python-RVO2.git```
- conda activate torch
- pip install Cython
- inside Python-RVO2, ```python setup.py build```, and then ```python setup.py install```. If error happened, please check cmake and build-essential are installed correctly.
- clone the project, ```git clone https://github.gatech.edu/GeorgiaTechLIDARGroup/sac_ae_lidar_map.git```
- inside C_library, ```python setup.py build_ext --inplace```
- in digit_mujoco, ```pip install -e .```. If segmentation fault, repeat ```pip install -e .```.
- test if it's correctly compiled, inside the main directory, ```python locomotion.py```
- run the training code with lip model, set robot_model as lip, then ```python sac_ae_main.py```.
- run the training code with digit mujoco model, set robot_model as digit_mujoco, then ```python sac_ae_main.py```.
- if training with lip model and testing with digit_mujoco, set robot_model as lip, and robot_test_model as digit_mujoco or digit_sim. At the meantime, specify load_test_model to load pre-trained networks. e.g. step_2720000_success_91.
- if training with digit mujoco model and testing with digit_mujoco, set robot_model as digit mujoco, and robot_test_model as digit_mujoco or digit_sim. At the meantime, specify load_test_model to load pre-trained networks. e.g. step_2720000_success_91.
- replay saved trajectory, ```python replay_episode.py```, please revise file path in ./logs/XXX
- train with digit_mujoco, set robot_model as digit_mujoco, then ```python sac_ae_main.py```. Currently, directly training with digit in mujoco is not good.
- baselines: dwa_main.py, rgl_main.py, lndnl_main.py, drl_vo_main.py

