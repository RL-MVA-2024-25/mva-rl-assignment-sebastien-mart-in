import random
import torch
import numpy as np
import torch.nn as nn
import os


class ProjectAgent:
    def __init__(self):
        buffer_capacity = int(1e5)
        self.model_path = os.path.join(os.path.dirname(__file__), "mymodel.pt")  
        self.action_space = 4  
        self.state_space = 6 
        self.policy_model = self.model_nn()

    def act(self, state, use_random=False):
        state_tensor = torch.Tensor(state).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        if use_random:
            return random.randint(0, self.action_space - 1)

        with torch.no_grad():
            q_values = self.policy_model(state_tensor)
        return torch.argmax(q_values).item()

    def save(self, path):
        torch.save(self.policy_model.state_dict(), path)

    def load(self):
        self.policy_model = self.model_nn()
        self.policy_model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu")))
        self.policy_model.eval()

    def model_nn(self):
        model = nn.Sequential(
            nn.Linear(self.state_space, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 2),
            nn.ReLU(),
            nn.Linear(256 * 2, 256 * 4),
            nn.ReLU(),
            nn.Linear(256 * 4, 256 * 8),
            nn.ReLU(),
            nn.Linear(256 * 8, self.action_space)
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model

