import numpy as np
import random

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def radius_clip(rad):
    '''normalize radius into [0~2*pi]'''
    while(rad < 0):
        rad += np.pi*2

    while(rad >= np.pi*2):
        rad -= np.pi*2

    return rad

def radius_scale(normalize_value):
    return normalize_value*2*np.pi

class EnergyHarvest():
    def __init__(self, n_agents=3, n_sensors=10, n_actions=32,\
                    max_steps=50, max_distance=50, alpha=3.0, beam_width=np.pi/4, power=50.0):
        self.n_agents = n_agents
        self.n_sensors = n_sensors
        self.n_actions = n_actions
        self.action_space = tuple(range(n_actions))
        self.max_setps = max_steps
        self.max_distance = max_distance
        self.beam_width = beam_width
        self.power = power
        self.scale = np.array([max_distance, np.pi*2, 1.0])

        #pass loss cofficient
        self.alpha = alpha

    def is_terminated(self):
        return np.all(self.sensors[:,2]==1.0)

    def _is_truncated(self):
        self.step_counter += 1
        if self.step_counter < self.max_setps:
            return False
        return True

    def global_state(self):
        '''[n_agents+n_sensors, 3]'''
        '''power station:   [rho, phi, direction]'''
        '''sensor:          [rho, phi, capacity]'''
        state = np.zeros((self.n_agents+self.n_sensors, 3))

        for i in range(self.n_agents):
            pstate = self.power_stations[i].state()
            pscale = np.array([self.max_distance, np.pi*2, np.pi*2])
            pstate = pstate/pscale
            state[i] = pstate
        
        state[self.n_agents:] = self.sensors

        return state

    def step(self, actions):
        
        if type(actions) != np.array:
            actions = np.array(actions)

        directions = actions*2*np.pi/self.n_actions
        diff = np.zeros(self.n_sensors)
        observations = np.zeros((self.n_agents, self.n_sensors + 1, 3))
        full_sensors = (self.sensors[:,2]==1.0).sum()
        for i in range(self.n_agents):
            self.power_stations[i].set_direction(directions[i])
            diff += self.power_stations[i].charge(self.sensors, self.alpha)
        for i in range(self.n_agents):
            observations[i] = self.power_stations[i].make_observation(self.sensors, self.scale)

        full_sensors_ = (self.sensors[:,2]==1.0).sum()

        reward = (full_sensors_ - full_sensors)

        termination = self.is_terminated()
        truncation = self._is_truncated()

        if termination:
            reward = 5.0
        else:
            reward -= 1.0/self.max_setps

        info = {
            'full_sensors': full_sensors_,
            'mean_transmit': diff.mean(),
        }
        
        return self.global_state(), observations, reward, termination, truncation, info
            

    def reset(self, seed=None):
        self.step_counter = 0
        if seed is not None:
            np.random.seed(seed)
        self.sensors = np.random.random((self.n_sensors, 3)) # [rho, phi, capacity]
        self.sensors[:,2] = 0.0

        interval = np.pi*2/self.n_agents
        rad_diff = np.pi*2*random.uniform(0,1)
        self.power_stations = []
        for i in range(self.n_agents):
            if seed is not None:
                np.random.seed(seed)
            distance = random.gauss(0.5, 0.1)*self.max_distance

            if seed is not None:
                np.random.seed(seed)
            rad = random.gauss(0, 0.2)+i*interval+rad_diff

            power_station = PowerStation(distance, rad, power=self.power, beam_width=self.beam_width)
            power_station.set_observation(self.sensors, self.scale)
            self.power_stations.append(power_station)

        observations = np.zeros((self.n_agents, self.n_sensors + 1, 3))
        for i in range(self.n_agents):
            observations[i] = self.power_stations[i].make_observation(self.sensors, self.scale)

        info = {
            'full_sensors':(self.sensors[:,2]==1.0).sum(),
            'mean_transmit':0.0
        }

        return self.global_state(), observations, info

        


class PowerStation():
    def __init__(self, rho, phi, power=1.0, beam_width=np.pi/4):
        # control parameters
        self.beam_width = beam_width
        self.power = power
        self.set_direction(0.0)
        # location of power station
        self.rho = rho
        self.phi = phi
        self.x, self.y = pol2cart(rho, phi)

    def _in_range(self, phi):
        phi = radius_clip(phi)

        if self.anghi > self.anglo:
            if phi <= self.anghi and phi >= self.anglo:
                return True
        else:
            if phi <= self.anghi or  phi >= self.anglo:
                return True
        return False

    def set_direction(self, direction:float):
        value = radius_clip(direction)

        self.direction=value
        self.anglo=self.direction-self.beam_width
        self.anghi=self.direction+self.beam_width

        while(self.anglo<=0):
            self.anglo += (np.pi*2)

        while(self.anghi>2*np.pi):
            self.anghi -= (np.pi*2)

    def set_observation(self, sensors:np.array, scale:np.array):
        self.n_sensors = sensors.shape[0]
        self.observation = np.zeros((self.n_sensors, 2))# [rel_distance, rel_radius]
        for i in range(self.n_sensors):
            rho, phi, _ = sensors[i]*scale
            x, y = pol2cart(rho, phi)

            rel_x, rel_y = x - self.x, y - self.y
            rel_rho, rel_phi = cart2pol(rel_x, rel_y)

            self.observation[i] = rel_rho, rel_phi
        
    def state(self):
        return np.array([self.rho, self.phi, self.direction])

    def make_observation(self, sensors:np.array, scale):
        observation = np.zeros((self.n_sensors + 1, 3))
        # self state
        observation[0] = self.state()/np.array([scale[0], scale[1], np.pi*2])
        # sensors 
        position = self.observation/scale[0:2]

        observation[1:, 0:2] = position
        observation[1:, 2] = sensors[:,2]
        
        return observation

    def transmit(self, sensor_index, sensor, alpha):
        '''charge sensor based on local observation'''
        if sensor[2] == 1.0 or (not self._in_range(self.observation[sensor_index][1])):
            return 0.0
        # calculate pass loss
        capacity = sensor[2]
        pass_loss = self.power*(4*(np.pi**2)/(self.beam_width*2*np.pi))/(self.observation[sensor_index][0]**alpha)
        capacity_ = capacity + pass_loss

        if capacity_ > 1.0:
            capacity_ = 1.0
        
        sensor[2] = capacity_

        return capacity_ - capacity

    def charge(self, sensors, alpha:float):
        diffs = np.zeros(self.n_sensors)

        for i in range(self.n_sensors):
            diffs[i] = self.transmit(i, sensors[i], alpha)
        
        return diffs
            
