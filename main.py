from qmix import *
import torch
from envs import *
from EnergyHarvest.env import EnergyHarvest

ep_steps = 50
total_steps = int(1e6)*ep_steps
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'eps_start':0.99,
    'eps_end':0.05,
    'eps_dec':total_steps*0.2, # more leads to slower decay
    'gamma':0.99,
    'lr': 1e-4,
    'tau':0.005, # more is harder
    'batch_size':256,
    'memory_size':1000000,
    'device':device,
    'logdir':'energy_harvest_4frame_sensor20_power20_rewardv2',
}

if __name__ == '__main__':
    print(device)
    env = EnergyHarvest_v1()
    qmix = QMIX(env, config)
    qmix.learn(total_steps)
    
    qmix.save_model()