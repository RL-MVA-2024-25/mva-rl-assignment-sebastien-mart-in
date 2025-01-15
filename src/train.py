from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_agent

from functools import partial

import random
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 200
DOMAIN_RANDOMIZATION = False
GAMMA = 0.99
BATCH_SIZE = 512
BUFFER_SIZE = 1e5
EPSILON_MAX = 1.
EPSILON_MIN = 0.01
EPSILON_DECAY_PERIOD = 1e4
EPSILON_DELAY_DECAY = 100
LEARNING_RATE = 1e-3
GRADIENT_STEPS = 3
UPDATE_TARGET_FREQ = 200
NEURONS = 256
MODEL_PATH = "dqn.pt"

SCALE_REWARD = 10*np.log(5)
SCALE_OBSERVATION = True

DOUBLE_DQN = False
FAST_ENV = False

OBSERVATION_SPACE = 6
ACTION_SPACE = 4

MAX_EPISODES_STEPS = 200
NB_EPSIODES_TEST = 1



class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
        self.size = 0
        self.obs_means = np.zeros(OBSERVATION_SPACE)
        self.obs_stds = np.ones(OBSERVATION_SPACE)
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity
        
        # Update the running mean and std
        observations = np.array([item[0] for item in self.data[:self.size]])
        self.obs_means = np.mean(observations, axis=0)
        self.obs_stds = np.std(observations, axis=0)

        
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    

class ProjectAgent:
    
    def __init__(self, args=None):
        self.episodes = args.episodes if args is not None else EPISODES
        self.nb_episodes_test = args.nb_episodes_test if args is not None else NB_EPSIODES_TEST
        self.gamma = args.gamma if args is not None else GAMMA                                      # 0.99 default value     ## discount factor
        self.batch_size = args.batch_size if args is not None else BATCH_SIZE                       # 512 default value      ## batch size
        buffer_size = args.buffer_size if args is not None else BUFFER_SIZE                       # 100_000 default value   ## buffer size
        self.epsilon_max = args.epsilon_max if args is not None else EPSILON_MAX                   # 1. default value       ## epsilon greedy
        self.epsilon_min = args.epsilon_min if args is not None else EPSILON_MIN                 # 0.01 default value     ## epsilon greedy
        self.epsilon_stop = args.epsilon_decay_period if args is not None else EPSILON_DECAY_PERIOD        # 10_000 default value    ## epsilon decay period
        self.epsilon_delay = args.epsilon_delay_decay if args is not None else EPSILON_DELAY_DECAY        # 100 default value      ## epsilon delay decay
        lr = args.learning_rate if args is not None else LEARNING_RATE                             # 0.001 default value    ## learning rate
        self.nb_gradient_steps = args.gradient_steps if args is not None else GRADIENT_STEPS           # 3 default value        ## gradient steps
        self.update_target_freq = args.update_target_freq if args is not None else UPDATE_TARGET_FREQ    # 200 default value      ## update target NET frequency
        self.neurons = args.neurons if args is not None else NEURONS                          # 256 default value      ## number of neurons
        self.criterion = torch.nn.SmoothL1Loss()               # Loss function
        
        self.path = os.path.join(os.path.dirname(__file__), args.model if args is not None else MODEL_PATH)
        self.nb_actions =  ACTION_SPACE # env.action_space.n,                           # 4 
        self.observation_space =OBSERVATION_SPACE # env.observation_space.shape[0],        # 6
        
        
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.memory = ReplayBuffer(buffer_size, DEVICE)
        self.model = self.get_DQN()
        self.target_model = deepcopy(self.model).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # Standard optimizer
        
        self.scale_reward = args.scale_reward if args is not None else SCALE_REWARD
        self.scale_observation = args.scale_observation if args is not None else SCALE_OBSERVATION
        self.obs_means = np.zeros(self.observation_space)
        self.obs_stds = np.ones(self.observation_space)
        
              
    def act(self, observation, use_random=False):
        observation = torch.Tensor(observation).unsqueeze(0).to(DEVICE)
        if self.scale_observation:
            observation = self._normalize_state(observation)
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        cpu_device = torch.device("cpu")
        DQN = self.get_DQN()
        self.model = DQN.to(cpu_device)
        self.model.load_state_dict(torch.load(self.path, map_location=cpu_device, weights_only=True))
        self.model.eval()
        return


    def get_DQN(self):
        # Defining an instance of the DQN model tested
        state_dim = self.observation_space
        n_action = self.nb_actions
        nb_neurons = self.neurons
        DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons*2),
            nn.ReLU(),
            nn.Linear(nb_neurons*2, nb_neurons*4),
            nn.ReLU(),
            nn.Linear(nb_neurons*4, nb_neurons*8),
            nn.ReLU(),
            nn.Linear(nb_neurons*8, n_action)
        ).to(DEVICE)
        return DQN 
    
    def _normalize_state(self, state):
        # Standardisation avec moyenne et écart-type (ajuster ces valeurs)
        # running mean and std
        state_mean = self.memory.obs_means
        state_std = self.memory.obs_stds
        state_mean = torch.Tensor(state_mean).to(DEVICE)
        state_std = torch.Tensor(state_std).to(DEVICE)
        return (state - state_mean) / state_std
    
    def _rescale_reward(self, reward):
        # Deal with tensor and gpu
        reward = torch.Tensor(reward).to(DEVICE)
        return torch.log(reward + torch.tensor(2.5e4) + torch.tensor(20000.0) * (0.3**2 + 0.7**2)) / torch.log(
            torch.tensor(353200.0 * 1000) + torch.tensor(2.5e4) + torch.tensor(20000.0) * (0.3**2 + 0.7**2)
        )

