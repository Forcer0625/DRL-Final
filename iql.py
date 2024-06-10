import torch
import torch.nn as nn
import numpy as np
from modules import MLPAgent
from utilis import ReplayBuffer
from runner import *
from envs import BaseMPE
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from qmix import QMIX

class IQL(QMIX):
    def __init__(self, env:BaseMPE, config):
        self.env = env
        self.n_agents = self.env.n_agents
        if type(self.env.n_actions) == int:
            self.n_actions = self.env.n_actions
        else:
            self.n_actions = self.env.n_actions[0]
        state, observation, _ = self.env.reset()

        self.batch_size = config['batch_size']
        self.memory_size = config['memory_size']
        self.memory = ReplayBuffer(self.memory_size, state.shape, observation[0].shape, self.n_agents)

        self.lr = config['lr']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.device = config['device']
        self.loss = nn.MSELoss()
        
        self.policy = MLPAgent(observation[0].reshape(-1).shape[0], self.n_actions, device=self.device)
        self.target_policy = deepcopy(self.policy)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.runner = EGreedyRunner(env, self.policy, self.memory,\
                                    config['eps_start'], config['eps_end'], config['eps_dec'])
        

        self.logger = SummaryWriter('./runs/'+config['logdir'])
        self.infos = []
        self.config = config

    def update(self, ):
        states, observations, actions, rewards,\
            dones, states_, observations_ = self.memory.sample(self.batch_size)
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device).view(-1, *observations[0][0].shape)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).view(-1)
        rewards = np.stack([rewards, rewards, rewards], axis=1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).view(-1)
        dones = np.stack([dones, dones, dones], axis=1)
        dones = torch.as_tensor(dones, dtype=torch.int, device=self.device).view(-1)
        states_= torch.as_tensor(states_, dtype=torch.float32, device=self.device)
        observations_= torch.as_tensor(observations_, dtype=torch.float32, device=self.device).view(-1, *observations_[0][0].shape)

        action_values = self.policy(observations).reshape(-1, self.n_actions)
        action_values = action_values.gather(1, actions.unsqueeze(1))
        action_values = action_values.reshape(-1)

        # double-q
        with torch.no_grad():
            estimate_action_values = self.policy(observations_).reshape(-1, self.n_actions)
            next_action = torch.max(estimate_action_values, dim=1).indices
            next_action_values = self.target_policy(observations_).reshape(-1, self.n_actions)
            next_action_values = next_action_values.gather(1, next_action.unsqueeze(1))
            next_action_values = next_action_values.reshape(-1)

        # calculate loss
        target = rewards + self.gamma * (1 - dones) * next_action_values
        loss = self.loss(action_values, target.detach())

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def sync(self, ):
        target_net_weights = self.target_policy.state_dict()
        q_net_weights = self.policy.state_dict()
        for key in q_net_weights:
            target_net_weights[key] = q_net_weights[key]*self.tau + target_net_weights[key]*(1-self.tau)
        self.target_policy.load_state_dict(target_net_weights)

    def hard_sync(self):
        self.target_policy.load_state_dict(self.policy.state_dict())