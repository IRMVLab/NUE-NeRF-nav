import numpy as np
from trainer import work
from model import E2E_model, NeRF_proc, model_train, nerf_train, NeRF_pi, E2E_model_without_exploration, E2E_model_only_exploration
from model_qkv import E2E_model_qkv
from util import render_heatmap
from dataset import dataset

import os
from trainer import Transition
import warnings
import torch
import yaml
import cv2 as cv
from datetime import datetime
from copy import deepcopy
from random import choice
from matplotlib import pyplot as plt
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe, Queue

warnings.filterwarnings("ignore", category=UserWarning)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# mode configuration
training = 1 # 1 for train

# RL
episodeLimit = 24000
batchSize = 256
initialLearningRate = 5e-4
initial_epsilon_greedy = 0.5
useGreedy = True and training
epsilonGreedyDecayStep = 2e6
lrDecayStep = 1e7  # 0  1e7
save_model_every_n_step = 10000

date=str(datetime.today())[:10]
model_path = './model/' + date + '/'
summary_path = './summary/' + date + '/'
model_load = ''

if training:
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if training:
    mode = "headless"
else:
    mode = "gui_non_interactive"

config_data = yaml.load(open("turtlebot_nav.yaml", "r"), Loader=yaml.FullLoader)

episodeCount = 0
last_save_step = 0
success_num = 0
spl_alpha_sum = 0
print('training')
global_step = 0

with open('cameras.txt', 'r') as f:
    K = f.readline()
    K = K.split(' ')
    H, W = int(K[-2]), int(K[-1])
    K = np.array([
        [float(K[0]), 0, float(K[2])],
        [0, float(K[1]), float(K[3])],
        [0, 0, 1]
    ], dtype=np.float32)

if __name__ == '__main__':
    model = E2E_model(3)
    #model.load_state_dict(torch.load(model_load, map_location='cpu'))
    nerf = NeRF_pi(3)
    nerf_tmp = NeRF_pi(3)

    mp.set_start_method('spawn')  # 设置multiprocessing模块的启动方法为'spawn'，这将启动新的进程
    manager = mp.Manager()  # 创建一个Manager对象，用于管理共享内存和进程间通信
    source = manager.list()  # 创建一个共享的可变列表source，可以在多个进程之间共享和修改
    print(len(source))
    _flag = manager.list()  # 创建一个共享的可变列表_flag
    _flag.append(1)  # 将值1添加到_flag列表中
    source_lock = mp.Lock()  # 创建一个共享锁对象source_lock，用于在多个进程之间同步对source列表的访问

    lock = mp.Lock()
    reset_list = manager.list()
    reset_list.append(False)
    parent_conn, child_conn = Pipe()  # 创建一个双向管道，parent_conn和child_conn分别是管道的两个端点
    nerf_list = manager.list()
    # nerf_list = Queue(maxsize=100)
    queue = Queue(maxsize=200)

    nerf_proc = NeRF_proc(nerf_tmp, device_0, nerf_list, N_sample=64)

    process = []
    p0 = Process(target=model_train, args=(model, device_1, source_lock, source, summary_path, model_path, _flag,))
    p0.start()
    process.append(p0)
    p1 = Process(target=nerf_train, args=(nerf, device_0, lock, queue, nerf_list, reset_list, child_conn,))
    p1.start()
    process.append(p1)
    data = dataset('./dataset', device_0, 30)
    cache = []
    if training:
        while episodeCount < episodeLimit:
            # 遍历data中的数据（使用一次导航的数据进行一次训练）
            for observation, robot_T, action, label, target, step, done in data:


                if step == 0:
                    nerf_proc.change_target(target)  # 更换目标图像
                prd_map, uncertainty_map, alpha = work(nerf_proc, observation, robot_T, lock, queue, step, 10800, device_0)  # 只渲染，不更新参数
                # 图像元素归一化
                observation[0, 0] *= 10  # 将observation的第一个元素的第一个子元素乘以10
                observation[0, 0][observation[0, 0] >= 5] = 5  # 将observation的第一个元素的第一个子元素中大于等于5的元素设置为5
                observation[0, 0] /= 5

                trans = Transition(observation, action, prd_map, label, uncertainty_map.unsqueeze(0))  # 创建一个命名元组
                cache.append(trans)  # trans存到cache中

                if done:
                    break

            if len(cache) > 1300:
                source_lock.acquire()  # 获取锁
                source.extend(cache)  # 将cache中数据添加到source
                source_lock.release()  # 释放锁
                print(len(cache))
                cache.clear()  # 清空cache

            reset_list[-1] = True  # nerf_reset
            print(parent_conn.recv())  # 接收并打印parent_conn收到的消息

        print('Training Ended!')

