import gym
import torch
import torch.nn as nn
import numpy as np
import os

# Environment setup
env = gym.make('LunarLander-v2', render_mode="human")


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
q_network = QNetwork(state_size, action_size)

# Load the saved model
# model_filename = '/home/janak/Documents/Pytorch_CPU/SQL/soft_q_learning_model.pth'
model_filename = '/home/janak/Documents/Pytorch_CPU/SQL/SQL_lunar_2.pth'

def load_model(model, filename):
    if os.path.isfile(filename):
        model.load_state_dict(torch.load(filename))
        print(f"Loaded model from {filename}")
    else:
        raise FileNotFoundError(f"No model found at {filename}")

load_model(q_network, model_filename)

# Epsilon-Greedy Action Selection for Testing (epsilon = 0 for greedy policy)
def select_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state_tensor)
    return torch.argmax(q_values).item()

# Testing the agent
def test_agent(num_episodes=10):
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        print(f'Episode {episode}, Total Reward: {total_reward}')

    average_reward = np.mean(total_rewards)
    print(f'Average Reward over {num_episodes} episodes: {average_reward}')
    return total_rewards

# Test the agent
test_agent(num_episodes=10)
