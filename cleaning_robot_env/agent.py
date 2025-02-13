import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from cleaning_robot_env.env import CleaningRobotEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)

class DQN(nn.Module):
    def __init__(self, input_size, output_size, kl=True):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.kl = kl
        if self.kl:
            self.fc_kl = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = {'Q_reward': self.fc3(x)}
        if self.kl:
            output['Q_KL'] = self.fc_kl(x)
        return output

class Agent:
    def __init__(self, env, temp=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.use_eps_greedy = True
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.temp = temp
        self.temp_decrement = 0.001
        self.temp_min = 0.5

        self.batch_size = 64
        self.target_update_interval = 10
        
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.q_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)
        self.criterion = nn.MSELoss()
    
    def get_action(self, state, training=True):
        if training and self.use_eps_greedy and random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
            
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                outputs = self.q_net(state_tensor)
                q_values = outputs['Q_reward']
                q_values -= self.temp * outputs.get('Q_KL', 0)
                
            return q_values.argmax().item()
        
        # boltzmann exploration
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = self.q_net(state_tensor)
        action_probs = self.get_probs(outputs)
        action = torch.multinomial(action_probs, 1)
        return action

    def get_probs(self, outputs):
        logits = outputs['Q_reward'] / self.temp
        if 'Q_KL' in outputs:
            logits -= outputs['Q_KL']
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs