from agent import Agent
from new_env import CleaningRobotEnv
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train(self, episodes=100):
    for episode in tqdm(range(episodes)):
        state, _ = self.env.reset()  # Gymnasium returns (obs, info)
        total_reward = 0
        done = False
        while not done:
            action = self.get_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            agent.memory.append((
                torch.as_tensor(state, dtype=torch.float32, device=agent.device),
                torch.as_tensor([action], dtype=torch.long, device=agent.device),
                torch.as_tensor([reward], dtype=torch.float32, device=agent.device),
                torch.as_tensor(next_state, dtype=torch.float32, device=agent.device),
                torch.as_tensor([done], dtype=torch.bool, device=agent.device)
            ))
            
            total_reward += reward
            state = next_state
            learn(agent)
        if agent.use_eps_greedy:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        else:
            agent.temp = max(agent.temp_min, agent.temp - agent.temp_decrement)
        
        if episode % agent.target_update_interval == 0:
            agent.target_net.load_state_dict(agent.q_net.state_dict())
    
def kl_divergence(p, q):
    p += torch.finfo(p.dtype).tiny
    q += torch.finfo(q.dtype).tiny
    return (p * (p / q).log()).sum(-1)

def learn(agent):
    if len(agent.memory) < agent.batch_size:
        return

    transitions = random.sample(agent.memory, agent.batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)
    
    states = torch.stack(states)
    actions = torch.cat(actions)
    rewards = torch.cat(rewards)
    next_states = torch.stack(next_states)
    dones = torch.cat(dones)
    
    outputs = agent.q_net(states)
    current_q = outputs['Q_reward'].gather(1, actions.unsqueeze(1)).squeeze(1)

    # soft Q-learning:
    with torch.no_grad():
        next_outputs = agent.target_net(next_states)
        next_probs = agent.get_probs(next_outputs)
        next_preds = {k: torch.einsum("ba,ba->b", next_probs, v) for k, v in next_outputs.items()}
    
    targets = {'Q_reward': rewards + (1 - dones.float()) * agent.discount_factor * next_preds['Q_reward']}
    loss = agent.criterion(current_q.squeeze(), targets['Q_reward'])
    if 'Q_KL' in outputs:
        targets['Q_KL'] = kl_divergence(torch.full_like(next_probs, 1 / agent.action_size), next_probs) 
        + (1 - dones.float()) * agent.discount_factor * next_preds['Q_KL']

        current_q_kl = outputs['Q_KL'].gather(1, actions.unsqueeze(1)).squeeze(1)
        loss += agent.criterion(current_q_kl, targets['Q_KL'])

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

def test(agent, episodes=100):
    avg_true_return = 0
    avg_return = 0
    for episode in range(episodes):
        state = agent.env.reset()
        _return = 0
        done = False
        true_return = 0
        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, done, info = agent.env.step(action)
            true_reward = info['true_reward']
            true_return += true_reward
            _return += reward
            state = next_state
            # self.env.render()
        avg_true_return += true_return
        avg_return += _return
        
        # print(f"Test Episode {episode + 1}, Return: {_return}, Total True Reward: {true_return}")
    avg_true_return /= episodes
    avg_return /= episodes
    print(f"Average Return: {avg_return}, Average True Return: {avg_true_return}")
    return avg_return, avg_true_return

def seed(x):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)

if __name__ == "__main__":
    grid_size = 3
    misspecs = [1, 0.75, 0.5]
    temps = np.logspace(-5, 1, 10)
    misspec_to_avg_returns = {}
    misspec_to_avg_true_returns = {}

    for misspec in misspecs:
        print(f"Misspecification Degree: {misspec}")
        env = CleaningRobotEnv(grid_size=grid_size, misspec=misspec)
        avg_returns = []
        avg_true_returns = []
        for temp in temps:
            seed(42)
            print(f"Temperature: {temp}")
            agent = Agent(env, temp=temp)
            train(agent, episodes=1000)
            seed(420)
            avg_return, avg_true_return = test(agent)
            avg_returns.append(avg_return)
            avg_true_returns.append(avg_true_return)

        misspec_to_avg_returns[misspec] = avg_returns
        misspec_to_avg_true_returns[misspec] = avg_true_returns

    print(f"{misspec_to_avg_returns=}")
    print(f"{misspec_to_avg_true_returns=}")
    for misspec in misspecs:
        avg_returns = misspec_to_avg_returns[misspec]
        avg_true_returns = misspec_to_avg_true_returns[misspec]

        plt.scatter(avg_returns, avg_true_returns, label=f"Misspec: {misspec}")

    plt.xlabel("Returns")
    plt.ylabel("True Returns")
    plt.xlim(0, 150)
    plt.ylim(0, 150)
    plt.plot([0, 150], [0, 150], color='black', linestyle='--')
    plt.legend()
    plt.show()