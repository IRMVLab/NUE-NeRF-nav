import numpy as np
from copy import deepcopy
import time
from igibson.envs.igibson_env import iGibsonEnv
from matplotlib import pyplot as plt
import os
import yaml
import json

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

def Quat2Rotation(x,y,z,w):
    l1 = np.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],axis=0)
    l2 = np.stack([2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x],axis=0)
    l3 = np.stack([2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2], axis=0)
    T_w = np.stack([l1,l2,l3],axis=0)
    return T_w

def Rotation2Quat(pose):
    m11,m22,m33 = pose[0][0],pose[1][1],pose[2][2]
    m12,m13,m21,m23,m31,m32 = pose[0][1],pose[0][2],pose[1][0],pose[1][2],pose[2][0],pose[2][1]
    x,y,z,w = np.sqrt(m11-m22-m33+1)/2,np.sqrt(-m11+m22-m33+1)/2,np.sqrt(-m11-m22+m33+1)/2,np.sqrt(m11+m22+m33+1)/2
    Quat_ = np.array([
        [x,(m12+m21)/(4*x),(m13+m31)/(4*x),(m23-m32)/(4*x)],
        [(m12+m21)/(4*y),y,(m23+m32)/(4*y),(m31-m13)/(4*y)],
        [(m13 + m31) / (4 * z), (m23 + m32) / (4 * z), z,(m12 - m21) / (4 * z)],
        [(m23 - m32) / (4 * w), (m31 - m13) / (4 * w), (m12 - m21) / (4 * w),w]
    ], dtype=np.float32)
    index = np.array([x,y,z,w]).argmax()
    Quat = Quat_[index]
    return Quat


def control(theta, scan):
    if 0.458334 <= theta <= 0.541667:  # 0.41667  0.58334
        act = 0
    elif -0.5 <= theta < 0.458334:
        act = 2
    else:
        act = 1

    if not hasattr(control, 'turn'):
        control.turn = -1
        control.turn_count = 0
        control.stright_count = 0

    # 如果距离障碍物过近或control.turn > 0
    if scan.min()*5.6 < 0.3 or control.turn > 0:
        # print('control')
        if control.turn < 0:
            index = scan.argmin()
            index_ = scan.argmax()
            portion = (index/len(scan)) * 90
            portion = portion//15
            por_ = (index_/len(scan)) * 90
            por_ = por_/15

            control.turn = 2 if por_ < 3 else 1
            control.turn_count = portion + 2
            control.stright_count = 2
            act = control.turn
        else:
            if not control.turn_count <= 0:
                control.turn_count -= 1
                act = control.turn
            elif not control.stright_count <= 0:
                control.stright_count -= 1
                act = 0
            else:
                control.turn = -1
    return act




def step(env, act, robot_T):

    if act == 0:
        state, reward, done, info = env.step(np.array([0.7, 0]))

    elif act == 1 or act == 2:
        T = np.matmul(robot_T, dt[act - 1])
        pos = T[:3, -1]
        ori = Rotation2Quat(T[:3, :3])
        env.robots[0].set_position_orientation(pos, ori)
        state, reward, done, info = env.step(np.array([0.0, 0.0]))

    return state, reward, done, info

# 保存数据
def save(count, sc, observation, target_theta, action, robot_pos_ori, dist, pathdist):
    if not os.path.exists('dataset'):
        os.mkdir('dataset')
    if not os.path.exists('dataset/'+sc):
        os.mkdir('dataset/'+sc)
    if not os.path.exists('dataset/'+sc+'/'+'%04i'%count):
        os.mkdir('dataset/'+sc+'/'+'%04i'%count)
    root = 'dataset/'+sc+'/'+'%04i'%count  # 将整型count转为四位字符串


    observation.tofile(os.path.join(root, 'observation.npy'))
    target_theta.tofile(os.path.join(root, 'target_theta.npy'))
    action.tofile(os.path.join(root, 'action.npy'))
    robot_pos_ori.tofile(os.path.join(root, 'robot_pos_ori.npy'))
    with open(os.path.join(root, 'info.txt'), 'w') as f:
        f.write('distance:%f pathdist:%f'%(dist, pathdist))


# 数据集的目录
root = './navigation_scenarios/waypoints/full+'


with open('scenes_for_training.txt', 'r') as f:
    scene_list = f.readlines()


scene = []
for sc in scene_list:
    scene.append(sc[:-1])

# 遍历场景
for sc in scene:

    with open(os.path.join(root, sc+'.json'), 'r') as f:
        content = json.load(f)

    config_data = yaml.load(open("turtlebot_nav.yaml", "r"), Loader=yaml.FullLoader)
    env = iGibsonEnv(config_file=deepcopy(config_data), scene_id=sc, mode='headless', action_timestep=2 / 5)

    count = 0
    for episode in content:
        observation = []
        action = []
        target_theta = []
        robot_pos_ori = []

        flag = False

        state = env.reset()
        # 起始位置
        start_pos = np.array([episode['startX'], episode['startY'], episode['startZ']])
        # 起始位姿
        start_ori = np.array([0, 0, -np.sin(episode['startAngle']/2), np.cos(episode['startAngle']/2)])
        # 终止位置
        end_pos = np.array([episode['goalX'], episode['goalY'], episode['goalZ']])
        # robot初始化
        env.robots[0].set_position_orientation(start_pos, start_ori)
        state,_,_,_ = env.step(np.array([0,0]))
        episode['waypoints'].append([episode['goalX'], episode['goalY'], episode['goalZ']])
        waypoints = np.array(episode['waypoints'])
        collison_num = 0
        map = deepcopy(env.scene.floor_map[episode['level']])/255
        step_num=0
        # 遍历路点中的位置
        for pos in waypoints:
            target_pos = np.array([pos[0], pos[1], pos[2], 1])
            collison_count = 0
            while True:
                robot_pos = env.robots[0].get_position()

                xxx,yyy = env.scene.world_to_map(robot_pos[:2])
                map[xxx,yyy] = 3
                plt.imshow(map)
                plt.pause(0.01)
                plt.clf()

                # 若与当前的目标路点距离较近，则结束对当前目标路点的导航
                if np.linalg.norm(robot_pos[:2] - target_pos[:2], ord=2) <= 0.2:
                    break


                x, y, z, w = env.robots[0].get_orientation()
                T = np.concatenate([Quat2Rotation(-x, -y, -z, w), robot_pos.reshape(3, 1)], axis=1)
                T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)

                # 与下一个路点的角度（用于计算下一步执行的动作）
                theta = np.matmul(np.linalg.inv(T), target_pos.reshape(4, 1))
                theta = theta.reshape(4)[:2]
                theta /= np.linalg.norm(theta, ord=2)
                theta = np.arctan2(theta[0], -theta[1]) / np.pi

                # 计算与终点的角度（用于记录）
                tar_theta = np.matmul(np.linalg.inv(T), np.concatenate([end_pos, np.array([1])]).reshape(4, 1))
                tar_theta = tar_theta.reshape(4)[:2]
                tar_theta /= np.linalg.norm(tar_theta, ord=2)
                tar_theta = np.arctan2(tar_theta[0], -tar_theta[1]) / np.pi

                T[:3, :3] = Quat2Rotation(x, y, z, w)
                robot_T = T

                # 计算下一步执行的动作
                act = control(theta, state['scan'])
                # 记录rgbd图
                rgb = state["rgb"].transpose([2, 0, 1])
                depth = state["depth"].transpose([2, 0, 1]) * 10.
                observation.append(np.concatenate([depth, rgb],0))
                # 记录与终点角度
                target_theta.append(tar_theta)
                # 记录动作
                action.append(act)
                # 记录位姿
                robot_pos_ori.append(deepcopy(robot_T))

                # 执行动作
                state, reward, done, info = step(env, act, robot_T)
                step_num += 1
                # print(state['scan'].min()*5.6, state['scan'].argmin())

                if info['collision_step'] > collison_num:
                    collison_num = info['collision_step']
                    collison_count += 1
                # 若碰撞次数过多或行走步数超过500则结束本次导航任务
                if collison_count > 5 or step_num > 500:
                    flag = True
                    break
            # 结束本次导航任务
            if flag == True:
                break

        # 如果完整结束导航，则保存数据
        if not flag:
            observation = np.stack(observation, 0).astype(np.float32)
            target_theta = np.array(target_theta).astype(np.float32)
            action = np.array(action).astype(np.uint8)
            robot_pos_ori = np.stack(robot_pos_ori, 0).astype(np.float32)
            # 保存数据
            save(count, sc, observation, target_theta, action, robot_pos_ori, episode['dist'], episode['pathDist'])
            count += 1
            print(sc, episode['dist'], episode['pathDist'])


