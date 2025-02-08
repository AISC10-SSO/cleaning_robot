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
    def __init__(self, grid_size=5, temp=1):
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
        
        self.env = CleaningRobotEnv(grid_size=grid_size)
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
            
            if episode % self.target_update_interval == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
    
    
    def kl_divergence(self, p, q):
        p += torch.finfo(p.dtype).tiny
        q += torch.finfo(q.dtype).tiny
        return (p * (p / q).log()).sum()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.stack(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        next_states = torch.stack(next_states)
        dones = torch.cat(dones)
        
        outputs = self.q_net(states)
        current_q = outputs['Q_reward'].gather(1, actions.unsqueeze(1))

        # soft Q-learning:
        next_outputs = self.target_net(next_states)
        next_probs = self.get_probs(next_outputs)
        next_preds = {k: torch.einsum("ba,ba->b", next_probs, v) for k, v in next_outputs.items()}
        targets = {'Q_reward': rewards + (1 - dones.float()) * self.discount_factor * next_preds['Q_reward']}
        loss = self.criterion(current_q.squeeze(), targets['Q_reward'])
        if 'Q_KL' in outputs:
            targets['Q_KL'] = self.kl_divergence(torch.full_like(next_probs, 1 / self.action_size), next_probs)
            current_q_kl = outputs['Q_KL'].gather(1, actions.unsqueeze(1))
            loss += self.criterion(current_q_kl, targets['Q_KL'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes=100):
        average_true_return = 0
        average_return = 0
        for episode in range(episodes):
            state = self.env.reset()
            _return = 0
            done = False
            true_return = 0
            while not done:
                action = self.get_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                true_reward = info['true_reward']
                true_return += true_reward
                _return += reward
                state = next_state
                # self.env.render()
            average_true_return += true_return
            average_return += _return
            
            # print(f"Test Episode {episode + 1}, Return: {_return}, Total True Reward: {true_return}")
        average_true_return /= episodes
        average_return /= episodes
        print(f"Average Reward: {average_return}, Average True Reward: {average_true_return}")
        return average_return, average_true_return

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    average_returns = []
    average_true_returns = []
    for temp in np.logspace(-5, 1, 10):
        print(f"Using temperature: {temp}")
        agent = Agent(grid_size=3, temp=temp)
        agent.train(episodes=1000)
        average_return, average_true_return = agent.test()
        average_returns.append(average_return)
        average_true_returns.append(average_true_return)

    print(f"Returns: {average_returns}")
    print(f"True Returns: {average_true_returns}")

    # # plot true returns vs returns
    # import matplotlib.pyplot as plt
    # plt.scatter(average_returns, average_true_returns)
    # plt.show()