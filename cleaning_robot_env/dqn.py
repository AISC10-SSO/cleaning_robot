import numpy as np
import random
import torch
from collections import deque
from cleaning_robot_env.env import CleaningRobotEnv
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, grid_size=3):
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.memory = deque(maxlen=2000)
        
        # Create environment and get state space size
        self.env = CleaningRobotEnv(grid_size=grid_size)
        self.state_space_size = self.env.observation_space.n
        self.action_space_size = self.env.action_space.n
        
        # Initialize Q-network
        self.q_network = QNetwork(self.state_space_size, self.action_space_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return np.argmax(q_values.numpy())
    
    def train(self, episodes=1000):
        rewards_history = []
        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                
                if len(self.memory) > self.batch_size:
                    self.replay()
            
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards_history.append(total_reward)
        return rewards_history
    
    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.q_network(next_states).max(1)[0]
        target_q_values = rewards + (self.discount_factor * next_q_values * (1 - dones))
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def test(self, episodes=1):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                state = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = np.argmax(self.q_network(state).numpy())
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.env.render()
            
            print(f"Test Episode {episode + 1}, Total Reward: {total_reward}")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    agent = Agent(grid_size=2)
    rewards = agent.train()
    agent.test()