import json
import os
import torch
import numpy as np
import yaml
import igibson
from trainer import work, Rotation2Quat, Quat2Rotation,writeSummary
from model import E2E_model, NeRF_proc, nerf_train_for_test, NeRF_pi, E2E_model_without_exploration
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Pipe, Queue
from igibson.envs.igibson_env import iGibsonEnv
from copy import deepcopy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from time import time
from model_qkv import E2E_model_qkv


dt = [
            np.array([
                [np.cos(-0.08334 * np.pi), -np.sin(-0.08334 * np.pi), 0, 0],
                [np.sin(-0.08334 * np.pi), np.cos(-0.08334 * np.pi), 0, 0],
                [0, 0, 1, 0.007],
                [0, 0, 0, 1]
            ]),
            np.array([
                [np.cos(0.08334 * np.pi), -np.sin(0.08334 * np.pi), 0, 0],
                [np.sin(0.08334 * np.pi), np.cos(0.08334 * np.pi), 0, 0],
                [0, 0, 1, 0.007],
                [0, 0, 0, 1]
            ]),
        ]

def action_step(env, act, robot_T):
    if act == 0:
        state, reward, done, info = env.step(np.array([0.7, 0]))

    elif act == 1 or act == 2:
        T = np.matmul(robot_T, dt[act - 1])
        pos = T[:3, -1]
        ori = Rotation2Quat(T[:3, :3])
        env.robots[0].set_position_orientation(pos, ori)
        state, reward, done, info = env.step(np.array([0.0, 0.0]))
    return state, reward, done, info

def load_config(map_list, map_index):
    config_filename = "./turtlebot_nav.yaml"
    config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config_data["scene"] = "gibson"
    config_data["scene_id"] = map_list[map_index]
    config_data["enable_shadow"] = True
    config_data["enable_pbr"] = True
    return config_data


max_step_num = 500
device_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
config_data = yaml.load(open("turtlebot_nav.yaml", "r"), Loader=yaml.FullLoader)

model_index = ''
model_load = 'model/' + model_index + '.pkl'

if not os.path.exists('test_result'):
    os.mkdir('test_result')
if not os.path.exists('test_result/' + '2024-01-27_6015'):
    os.mkdir('test_result/' + '2024-01-27_6015')
if not os.path.exists('test_result/' + model_index):
    os.mkdir('test_result/' + model_index)

if __name__=='__main__':

    map_list = ['Avonia', 'Azusa', 'Crandon', 'Lathrup', 'Mosinee', 'Nicut', 'Rabbit', 'Sawpit', 'Scioto', 'Shelbiana', 'Silas', 'Sisters', 'Spencerville', 'Swormville']
    map_index = 0
    dataset_path = 'test_dataset'
    config_data = load_config(map_list, map_index)
    map_path = os.path.join(dataset_path, map_list[map_index], 'floor_trav_0.png')

    model = E2E_model(3)
    #model.load_state_dict(torch.load(model_load, map_location=device_0))
    model.to(device_0)
    model.eval()
    nerf = NeRF_pi(3)
    nerf_tmp = NeRF_pi(3)

    mp.set_start_method('spawn')
    manager = mp.Manager()

    lock = mp.Lock()
    reset_list = manager.list()
    reset_list.append(False)
    parent_conn, child_conn = Pipe()
    nerf_list = manager.list()
    # nerf_list = Queue(maxsize=100)
    queue = Queue(maxsize=200)
    nerf_proc = NeRF_proc(nerf_tmp, device_0, nerf_list, N_sample=192)
    process = []
    p1 = Process(target=nerf_train_for_test, args=(nerf, device_1, lock, queue, nerf_list, reset_list, child_conn,))
    p1.start()
    process.append(p1)
    test_scenes = ['Avonia.json', 'Lathrup.json', 'Mosinee.json', 'Rabbit.json', 'Swapit.json', 'Ribera.json'
    ]

    _alpha = 1
    print(test_scenes)

    result_save_path ='test_result/' + model_index
    writer = SummaryWriter(result_save_path)
    stats = {'SR': []}
    _count = 0
    success_total = 0

    # 遍历测试场景
    for scene in test_scenes:
        print(scene)
        env = iGibsonEnv(config_file=deepcopy(config_data), scene_id=scene[:-5], mode='headless', action_timestep=2 / 5)
        env.reset()
        with open('test_dataset/'+scene, 'r') as f:
            content = json.load(f)
        result = []
        # 遍历每一个episode
        for episode in content:

            env.robots[0].set_position_orientation(np.array(episode['endXYZ']), np.array(episode['endQuat']))  # 将机器人放置到终点
            _,_,_,_ = env.step(np.array([0,0]))  # 执行无效动作，使环境更新状态
            state, _, _, _ = env.step(np.array([0, 0]))  # 获取当前状态
            nerf_proc.change_target(torch.from_numpy(state['rgb'].copy()).to(device_0))  # 目标图像设置为终点的RGB图像

            env.robots[0].set_position_orientation(np.array(episode['startXYZ']), np.array(episode['startQuat']))  # 将机器人放置到起点
            _, _, _, _ = env.step(np.array([0, 0]))
            state, _, _, _ = env.step(np.array([0, 0]))

            robot_pos = env.robots[0].get_position()
            delta_dist = 0
            closed_dist = 999
            distance = 999
            success = False
            end_pos = np.array(episode['endXYZ'])
            total_dt = 0
            total_count = 0
            total_cog = 0
            total_pol = 0

            for step in range(max_step_num):
                total_count += 1

                rgb = state["rgb"].transpose([2, 0, 1])
                depth = state["depth"].transpose([2, 0, 1])
                observation = np.concatenate([depth, rgb], 0)
                observation = np.expand_dims(observation, axis=0)
                robot_T = np.eye(4)  # 构建机器人的变换矩阵
                x, y, z, w = env.robots[0].get_orientation()
                robot_T[:3, :3] = Quat2Rotation(x, y, z, w)
                robot_T[:3, -1] = robot_pos
                robot_T_ = deepcopy(robot_T)

                t0 = time()
                prd_map, uncertainty_map, alpha = work(nerf_proc, observation, robot_T_, lock, queue, step, 10800, device_0, device_1)  # 通过nerf渲染图像

                with torch.no_grad():
                    observation[0, 0] *= 10
                    observation[0, 0][observation[0, 0] >= 5] = 5
                    observation[0, 0] /= 5
                    observation = torch.from_numpy(observation).to(device_0)

                    action_prob, dt_cog, dt_pol = model(observation=observation, out_pred=prd_map, uncertainty_map=uncertainty_map.unsqueeze(0).to(device_0), type='gathering')
                act = np.argmax(action_prob.cpu()).item()

                t1 = time()

                state, _, _, _ = action_step(env, act, robot_T)  # 执行选择的动作
                delta_dist += np.linalg.norm(env.robots[0].get_position()[:2]-robot_pos[:2], ord=2)  # 增加的距离
                robot_pos = env.robots[0].get_position()  # 更新robot_pos
                distance = np.linalg.norm(robot_pos[:2]-end_pos[:2], ord=2)  # 计算当前位置到目标位置的距离
                closed_dist = min(closed_dist, distance)

                t = t1-t0
                total_dt += t
                total_cog += dt_cog
                total_pol += dt_pol


                if distance <= 1:  # 如果距离小于等于0.8，则标记成功，并跳出循环
                    success = True
                    break

            print('Total Time', total_dt / total_count)
            print('Cognition Extraction', total_cog / total_count)
            print('Policy Generation', total_pol / total_count)

            reset_list[-1] = True  # 设置重置列表的最后一个元素为True，重置nerf
            print(parent_conn.recv())

            label = []
            if episode['pathDist'] / episode['dist'] < 1.2:
                label.append('straight')
            else:
                label.append('curved')

            if 1.5 <= episode['dist'] < 3:
                label.append('easy')
            elif 3 <= episode['dist'] < 5:
                label.append('medium')
            elif 5 <= episode['dist']:
                label.append('hard')

            result.append(
                {
                    'if success': success,  # 是否成功
                    'path dist': delta_dist,  # 机器人行走的路径长度
                    'closed_dist': closed_dist,  # 导航过程中距离目标点最近距离
                    'optimal dist': episode["pathDist"],  # 最优路径长度
                    'dist': episode["dist"],  # 起始点与目标点距离
                    'label': label,
                    'border': 0.8,
                    'endDist': distance  # 导航结束时与目标点的距离
                }
                )

            with open(os.path.join(result_save_path, scene), 'w') as savetxt:
                json.dump(deepcopy(result), savetxt)
            _count += 1
            success_total += 1*success
            stats['SR'].append(success_total/_count)  # 成功率
            writeSummary(writer, stats, _count)
            print('over')
    writer.close()


