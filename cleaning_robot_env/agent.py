import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from cleaning_robot_env.env import CleaningRobotEnv
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, input_size, output_size, kl=False):
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
    def __init__(self, grid_size=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.learning_rate = 0.001
        self.discount_factor = 0.95
        self.use_eps_greedy = True
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.temp = 1
        self.temp_decrement = 0.001
        self.temp_min = 0.5

        self.batch_size = 64
        self.target_update_interval = 10
        
        self.env = CleaningRobotEnv(grid_size=grid_size)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Networks directly on GPU
        self.q_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=10000)
        self.criterion = nn.MSELoss()
    
    def get_action(self, state):
        if self.use_eps_greedy:
            if random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.action_size)
            
            # Direct GPU tensor creation
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
        logits -= outputs.get('Q_KL', 0)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs

    def train(self, episodes=100):
        rewards_history = []
        for episode in tqdm(range(episodes)):
            state = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transitions directly on GPU
                self.memory.append((
                    torch.as_tensor(state, dtype=torch.float32, device=self.device),
                    torch.as_tensor([action], dtype=torch.long, device=self.device),
                    torch.as_tensor([reward], dtype=torch.float32, device=self.device),
                    torch.as_tensor(next_state, dtype=torch.float32, device=self.device),
                    torch.as_tensor([done], dtype=torch.bool, device=self.device)
                ))
                
                total_reward += reward
                state = next_state
                self.learn()
            if self.use_eps_greedy:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            else:
                self.temp = max(self.temp_min, self.temp - self.temp_decrement)

            rewards_history.append(total_reward)
            
            if episode % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
        
        return rewards_history
    
    def kl_divergence(self, p, q):
        p += torch.finfo(p.dtype).tiny
        q += torch.finfo(q.dtype).tiny
        return (p * (p / q).log()).sum()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # All tensors already on GPU - no need for device moves
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)
        
        outputs = self.q_net(states)
        current_q = outputs['Q_reward'].gather(1, actions.unsqueeze(1))

        next_outputs = self.target_net(next_states)
        next_probs = self.get_probs(next_outputs)
        next_preds = {k: torch.einsum("ba,ba->b", next_probs, v) for k, v in next_outputs.items()}
        targets = {'Q_reward': rewards + (1 - dones.float()) * self.discount_factor * next_preds['Q_reward']}
        if 'Q_KL' in outputs:
            targets['Q_KL'] = self.kl_divergence(torch.full_like(next_probs, 1 / self.action_size), next_probs)
        
        loss = self.criterion(current_q.squeeze(), targets['Q_reward'])
        if 'Q_KL' in outputs:
            current_q_kl = outputs['Q_KL'].gather(1, actions.unsqueeze(1))
            loss += self.criterion(current_q_kl, targets['Q_KL'])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes=100):
        average_true_reward = 0
        average_reward = 0
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            total_true_reward = 0
            while not done:
                # Direct GPU tensor creation
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_state, reward, done, info = self.env.step(action)
                true_reward = info['true_reward']
                total_true_reward += true_reward
                total_reward += reward
                state = next_state
                self.env.render()
            average_true_reward += total_true_reward
            average_reward += total_reward
            
            print(f"Test Episode {episode + 1}, Total Reward: {total_reward}, Total True Reward: {total_true_reward}")
        average_true_reward /= episodes
        average_reward /= episodes
        print(f"Average Reward: {average_reward}, Average True Reward: {average_true_reward}")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    agent = Agent(grid_size=3)
    rewards = agent.train(episodes=100)
    agent.test()