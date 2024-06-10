from pettingzoo.mpe import simple_spread_v3, simple_reference_v3, simple_world_comm_v3
from EnergyHarvest.env import EnergyHarvest
from utilis import MultiFrameStack, FrameStack
from copy import deepcopy
import numpy as np

class BaseMPE():
    def __init__(self, render_mode=None):
        self.env = self.n_agents = self.n_actions = None
        self.frame_buffer = None
    
    def step(self, actions):
        '''returns global_state, observations, reward, terminations, truncations, infos'''
        dict_actions = {}
        a = 0
        for agent in self.env.agents:
            dict_actions[agent] = actions[a]
            a+=1
        single_frame_observatons, reward, terminations, truncations, infos = self.env.step(dict_actions)
        self.frame_buffer.push(single_frame_observatons)
        terminations = terminations['agent_0']
        truncations = truncations['agent_0']
        reward = reward['agent_0']
        return self.global_state(), self.frame_buffer.get(), reward, terminations, truncations, infos
    
    def simulate(self, actions):
        dict_actions = {}
        a = 0
        for agent in self.env.agents:
            dict_actions[agent] = actions[a]
            a+=1
        mirror = deepcopy(self.env)
        single_frame_observatons, reward, terminations, truncations, infos = mirror.step(dict_actions)
        global_state , observations = self.frame_buffer.next_frame(single_frame_observatons)
        terminations = terminations['agent_0']
        truncations = truncations['agent_0']
        reward = reward['agent_0']
        return global_state, observations, reward, terminations, truncations, infos

    def reset(self, seed=None):
        self.frame_buffer.clear()
        if seed is not None:
            single_frame_observatons, infos = self.env.reset(seed=seed)
        else:    
            single_frame_observatons, infos = self.env.reset()
        self.frame_buffer.push(single_frame_observatons)
        observatons = self.frame_buffer.get()
        return self.global_state(), observatons, infos

    def global_state(self):
        '''global state is concated from all agents' local observations'''
        return self.frame_buffer.top().reshape(-1)

class EnergyHarvest_v1():
    def __init__(self, render_mode=None, n_agents=3, n_sensors=10, n_actions=32,\
                    max_steps=50, max_distance=50, alpha=3.0, beam_width=np.pi/4, power=50.0, n_frame_stacks=2):
        
        env = EnergyHarvest(n_agents=n_agents, n_sensors=n_sensors, n_actions=n_actions,\
                                 max_steps=max_steps, max_distance=max_distance, alpha=alpha,\
                                 beam_width=beam_width, power=power)
        self.env = env
        self.reward_scale = 1.0
        self.n_agents = n_agents
        self.n_actions = [n_actions for _ in range(self.n_agents)]
        self.agents = ['power_station_'+str(i) for i in range(self.n_agents)]
    
        self.n_frame_stacks = n_frame_stacks
        self.frame_buffer = MultiFrameStack(self.agents, n_frame_stacks)

    def step(self, actions):
        state, single_frame_observatons, reward, terminations, truncations, infos = self.env.step(actions)
        self.frame_buffer.push(single_frame_observatons[:,:,2])
        observations = np.zeros((self.n_agents, self.env.n_sensors+1, 2+self.n_frame_stacks))
        observations[:,:,0:2] = single_frame_observatons[:,:,0:2]
        directions_and_capacities = self.frame_buffer.get()
        directions_and_capacities = directions_and_capacities.reshape(self.n_agents, self.n_frame_stacks, self.env.n_sensors+1)
        directions_and_capacities = directions_and_capacities.transpose((0,2,1))
        observations[:,:,2:] = directions_and_capacities
        return state.reshape(-1), observations.reshape(self.n_agents, -1), reward*self.reward_scale, terminations, truncations, infos

    def reset(self, seed=None):
        self.frame_buffer.clear()
        state, single_frame_observatons, infos = self.env.reset(seed=seed)
        self.frame_buffer.push(single_frame_observatons[:,:,2])
        observations = np.zeros((self.n_agents, self.env.n_sensors+1, 2+self.n_frame_stacks))
        observations[:,:,0:2] = single_frame_observatons[:,:,0:2]
        directions_and_capacities = self.frame_buffer.get()
        directions_and_capacities = directions_and_capacities.reshape(self.n_agents, self.n_frame_stacks, self.env.n_sensors+1)
        directions_and_capacities = directions_and_capacities.transpose((0,2,1))
        observations[:,:,2:] = directions_and_capacities
        return state.reshape(-1), observations.reshape(self.n_agents, -1), infos
    
    def concate(self, positions, frames):
        pass
    
    def np2dict(self, original_obs):
        i = 0
        obs = {}
        for agent in self.agents:
            obs[agent] = original_obs[i]
            i+=1
        return obs
    
class Spread(BaseMPE):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.env = simple_spread_v3.parallel_env(render_mode=render_mode)
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.n_actions = [self.env.action_space(agent).n for agent in self.env.agents]
        self.frame_buffer = MultiFrameStack(self.env.agents, 4)

class Reference(BaseMPE):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.env = simple_reference_v3.parallel_env(render_mode=render_mode)
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.n_actions = [self.env.action_space(agent).n for agent in self.env.agents]
        self.frame_buffer = MultiFrameStack(self.env.agents, 4)

class Comm(BaseMPE):
    '''inhomogeneous agents : [leadadversary_0, adversary_0, adversary_1, adversary_3, agent_0, agent_1]'''
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.env = simple_world_comm_v3.parallel_env(render_mode=render_mode)
        self.env.reset()
        self.n_agents = len(self.env.agents)
        self.frame_buffer = MultiFrameStack(self.env.agents, 4)

    def step():
        pass
    def reset():
        pass
    def global_state():
        pass

        