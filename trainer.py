import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from collections import namedtuple
from time import time

Transition = namedtuple('Transition', ['state', 'action', 'prd_map', 'label', 'uncertainty_map'])  # 创建一个命名元组


def work(nerf, observation, robot_T, lock, queue, step, nerf_batch, device, other_device=None):
    observation = torch.from_numpy(observation).to(device)
    robot_T = torch.from_numpy(robot_T).to(device)
    with torch.no_grad():
        prd_map, uncertainty_map, alpha = nerf.memory_process(observation, robot_T, lock, queue, step, nerf_batch, other_device)

        if other_device==None:
            return prd_map.cpu(), uncertainty_map.cpu(), alpha.cpu()#,dt0,dt1
        else:
            return prd_map, uncertainty_map, alpha#,dt0,dt1

def adjust_learning_rate(lr_decay_step,global_step,initial_lr,optimizer):
    if lr_decay_step > 0:
        learning_rate = 0.9 * initial_lr * (
                lr_decay_step - global_step) / lr_decay_step + 0.1 * initial_lr
        if global_step > lr_decay_step:
            learning_rate = 0.1 * initial_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        learning_rate = initial_lr

    return learning_rate


def writeSummary(writer,stats,episode_num):
    for key in stats:
        if len(stats[key]) > 0:
            stat_mean = float(np.mean(stats[key]))
            writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stat_mean, global_step=episode_num)
            stats[key] = []
    writer.flush()


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
#
# class Trainer(object):
#     def __init__(self, env,
#                  nerf,
#                  use_greedy=False,
#                  device='cpu',
#                  clip_param=0.2,
#                  max_grad_norm=0.4,
#                  ppo_update_iters=5,
#                  batch_size=128,
#                  gamma=0.99,
#                  lr_decay_step=5e6,
#                  epsilon_greedy_decay_step=5e6,
#                  initial_lr=1e-6,
#                  initial_epsilon_greedy=0.5
#                  ):
#         super(Trainer, self).__init__()
#
#         self.initial_lr = initial_lr
#         self.initial_epsilon_greedy = initial_epsilon_greedy
#         self.clip_param = clip_param
#         self.max_grad_norm = max_grad_norm
#         self.ppo_update_iters = ppo_update_iters
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.lr_decay_step = lr_decay_step
#         self.epsilon_greedy_decay_step = epsilon_greedy_decay_step
#
#         self.device = device
#         self.use_greedy = use_greedy
#         self.nerf = nerf
#         self.env = env
#
#         # Training stats
#         self.global_step = 0
#         self.episode_num=0
#         self.buffer = []
#         self.stats = {'cumulative_reward': [], 'episode_length': [], 'value_loss': [], 'distance' :[],
#                       'policy_loss': [], 'learning_rate': [], "global_step": [],'SR':[],'SR_self_ctrl':[],'SPL':[],'DTS':[],'pred_loss':[]}
#
#     def _adjust_learning_rate(self):
#         if self.lr_decay_step > 0:
#             learning_rate = 0.9 * self.initial_lr * (
#                     self.lr_decay_step - self.global_step) / self.lr_decay_step + 0.1 * self.initial_lr
#             if self.global_step > self.lr_decay_step:
#                 learning_rate = 0.1 * self.initial_lr
#             for param_group in self.optimizer.param_groups:
#                 param_group['lr'] = learning_rate
#         else:
#             learning_rate = self.initial_lr
#         self.stats['learning_rate'].append(learning_rate)
#
#     def work(self, observation, robot_T, nerf_batch, theta, is_expert, type_="selectAction"):
#         def control(theta):
#             if 0.44445 <= theta <= 0.55556:  # 0.41667  0.58334
#                 return 0
#             elif -0.5 <= theta < 0.44445:
#                 return 2
#             else:
#                 return 1
#
#         """
#         Forward pass of the PPO agent. Depending on the type_ argument, it either explores by sampling its actor's
#         softmax output, or eliminates exploring by selecting the action with the maximum probability (argmax).
#         """
#         observation = from_numpy(observation).float().unsqueeze(0).to(self.device[1])
#         robot_T = from_numpy(robot_T).float().to(self.device)
#
#         with torch.no_grad():
#             prd_map = self.model.memory_process(observation, robot_T, nerf_batch)
#             if not is_expert:
#                 observation[0, 0] *= 10
#                 observation[0, 0][observation[0, 0] >= 5] = 5
#                 observation[0, 0] /= 5
#                 action_prob = self.model(x=observation, type='gathering')
#             else:
#                 return control(theta), None, prd_map
#
#         epsilon_greedy = self.initial_epsilon_greedy * (
#                 self.epsilon_greedy_decay_step - self.global_step) / self.epsilon_greedy_decay_step
#         if self.global_step > self.epsilon_greedy_decay_step:
#             epsilon_greedy = 0
#
#         if type_ == "selectAction":
#             if self.use_greedy and np.random.rand() < epsilon_greedy:
#                 return int(np.random.choice(range(self.env.action_space))), (1 / self.env.action_space),prd_map,control(theta)
#             else:
#                 c = Categorical(action_prob)
#                 action = c.sample()
#                 return int(action.item()), action_prob[:, action.item()].item(),prd_map,control(theta)
#         elif type_ == "selectActionMax":
#             return np.argmax(action_prob.cpu()).item(), 1.0,prd_map,control(theta)
#
#     def save(self, path):
#         """
#         Save actor and critic models in the path provided.
#
#         :param path: path to save the models
#         :type path: str
#         """
#         save(self.model.state_dict(), path + str(self.episode_num)+"_"+str(self.global_step) + '.pkl')
#
#     def load(self, path):
#         """
#         Load actor and critic models from the path provided.
#
#         :param path: path where the models are saved
#         :type path: str
#         """
#         model_state_dict = load(path)
#         self.model.load_state_dict(model_state_dict, strict=True)
#
#     def writeSummary(self, writer):
#         """
#         Write training metrics and data into tensorboard.
#
#         :param writer: pre-defined summary writer
#         :type writer: TensorBoard summary writer
#         """
#         for key in self.stats:
#             if len(self.stats[key]) > 0:
#                 stat_mean = float(np.mean(self.stats[key]))
#                 writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stat_mean, global_step=self.episode_num)
#                 self.stats[key] = []
#         writer.flush()
#
#     def storeTransition(self, transition):
#         """
#         Stores a transition in the buffer to be used later.
#
#         :param transition: contains state, action, action_prob, reward, next_state
#         :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
#         """
#         self.buffer.append(transition)
#
#     def trainStep(self, is_expert, batchSize_=None):
#         """
#         Performs a training step for the actor and critic models, based on transitions gathered in the
#         buffer. It then resets the buffer.
#
#         :param batchSize: Overrides agent set batch size, defaults to None
#         :type batchSize: int, optional
#         """
#         # Default behaviour waits for buffer to collect at least one batch_size of transitions
#         if batchSize_ is None:
#             if len(self.buffer) < self.batch_size:
#                 batchSize = len(self.buffer)
#             else:
#                 batchSize = self.batch_size
#         else:
#             batchSize=batchSize_
#         # Extract states, actions, rewards and action probabilities from transitions in buffer
#         state = np.stack([tt.state for tt in self.buffer],axis=0)
#         state = torch.from_numpy(state).float().to(self.device)
#         action = tensor([tt.action for tt in self.buffer], dtype=torch.long).view(-1, 1).to(self.device)
#         if not is_expert: reward = [tt.reward for tt in self.buffer]
#         if not is_expert: old_action_log_prob = tensor([tt.a_log_prob for tt in self.buffer], dtype=torch.float32).view(-1, 1).to(self.device)
#         out_pred = torch.cat([tt.prd_map for tt in self.buffer], dim=0).to(self.device)
#         label = torch.from_numpy(np.stack([tt.label for tt in self.buffer],0)).to(self.device)
#         #tcn_output = torch.stack([tt.tcn_output for tt in self.buffer],dim=0)
#         # learning rate decay
#         self._adjust_learning_rate()
#         # Unroll rewards
#         total_v, total_p, total_t = 0, 0, 0
#         _count = 0
#         if is_expert:
#             for i in range(self.ppo_update_iters):
#                 for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batchSize, False):
#                     V, action_prob, pred = self.model(x=state[index], out_pred=out_pred[index], type='training')
#                     action_loss = F.cross_entropy(action_prob, action[index].view(-1))
#                     theta_loss = F.cross_entropy(pred, label[index])
#                     loss = action_loss + theta_loss
#                     self.optimizer.zero_grad()  # Delete old gradients
#                     loss.backward()  # Perform backward step to compute new gradients
#                     nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # Clip gradients
#                     self.optimizer.step()  # Perform training step based on gradients
#                     total_p += action_loss
#                     total_t += theta_loss
#                     _count += 1
#         else:
#             R = 0
#             Gt = []
#             for r in reward[::-1]:
#                 R = r + self.gamma * R
#                 Gt.insert(0, R)
#             Gt = tensor(Gt, dtype=torch_float).to(self.device)
#             # Repeat the update procedure for ppo_update_iters
#             for i in range(self.ppo_update_iters):
#                 # Create randomly ordered batches of size batchSize from buffer
#                 #for index in range(len(self.buffer)):
#                 for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), batchSize, False):
#                     # Calculate the advantage at each step
#                     Gt_index = Gt[index].view(-1, 1)
#                     V, action_prob, pred = self.model(x=state[index], out_pred=out_pred[index], type='training')
#                     delta = Gt_index - V
#                     advantage = delta.detach()
#
#                     # Get the current probabilities
#                     # Apply past actions with .gather()
#                     action_prob = action_prob.gather(1, action[index])  # new policy
#                     # PPO
#                     ratio = (action_prob / old_action_log_prob[index])  # Ratio between current and old policy probabilities
#                     surr1 = ratio * advantage
#                     surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
#
#                     # update main network
#                     action_loss = -torch_min(surr1, surr2).mean()  # MAX->MIN descent
#                     value_loss = F.mse_loss(Gt_index, V)
#                     theta_loss = F.cross_entropy(pred, label[index])
#                     loss = action_loss + value_loss + theta_loss
#
#                     self.optimizer.zero_grad()  # Delete old gradients
#                     loss.backward()  # Perform backward step to compute new gradients
#                     nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # Clip gradients
#                     self.optimizer.step()  # Perform training step based on gradients
#                     _count += 1
#                     total_v += value_loss
#                     total_p += action_loss
#                     total_t += theta_loss
#
#         if not is_expert: self.stats['value_loss'].append(total_v.cpu().item()/_count)
#         self.stats['policy_loss'].append(total_p.cpu().item()/_count)
#         self.stats['pred_loss'].append(total_t.cpu().item()/_count)
#         # After each training step, the buffer is cleared
#         if not is_expert or len(self.buffer) >1000:
#             del self.buffer
#             self.buffer = []
#
#
