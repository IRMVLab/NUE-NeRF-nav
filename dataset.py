import numpy as np
import os
from random import choice
import torch


def read_episode(root, sc, index):
    path = os.path.join(root, sc, index)
    observation = np.fromfile(path+'/observation.npy', dtype=np.float32).reshape(-1, 4, 180, 240)
    observation[:, 0] /= 10
    action = np.fromfile(path+'/action.npy', dtype=np.uint8).reshape(-1, 1)
    label = np.fromfile(path+'/target_theta.npy', dtype=np.float32).reshape(-1, 1)
    robot_T = np.fromfile(path+'/robot_pos_ori.npy', dtype=np.float32).reshape(-1, 4, 4)
    return observation, action, label, robot_T

def dataset(root, device, low_limit=30):
    scene = []
    # 获取所有场景
    for sc in os.listdir(root):
        # 若此场景中的训练样本数足够多，将此场景加入训练集
        if len(os.listdir(root+'/'+sc)) >= low_limit:
            scene.append(sc)

    while True:
        sc = choice(scene)  # 随机选择一个场景
        observation, action, label, robot_T = read_episode(root, sc, choice(os.listdir(root+'/'+sc)))  # 随机选择一个场景下的一次导航的训练数据
        target = observation[-1][1:].transpose([1,2,0])
        target = torch.from_numpy(target).to(device)
        # observation = torch.from_numpy(observation).to(device)
        # robot_T = torch.from_numpy(robot_T).to(device)
        for i in range(observation.shape[0]-1):
            done = i==observation.shape[0]-2
            yield observation[i:i+1], robot_T[i], action[i], label[i], target, i, done

