import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torch.nn.init as init

# Define the ActorCritic class with convolutional layers
class ActorCritic(nn.Module):
    def __init__(self, height=10, width=10, hidden_dim=128, action_dim=4):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        # self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        # init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        
        # Calculate the size of the output from the convolutional layers
        def conv2d_size_out(size, kernel_size=3, stride=1, padding=0):
            return (size - (kernel_size - 1) - 1 + 2 * padding) // stride + 1
        
        convw = conv2d_size_out(width)
        convh = conv2d_size_out(height)
        self.linear_input_size = convw * convh * 3  # 3 is the number of channels after conv2
        # self.linear_input_size = 432
        print(f"Linear input size: {self.linear_input_size}")
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.linear_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, action_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
        # LeakyReLU activation function
        self.elu = nn.GELU()

    def swish(self,x):
        return x*torch.sigmoid(x)


    def pi(self, x):
        # print("*****************************")
        # print("Starting Value of x in pi")
        # print(x[0])

        x = self.swish(self.conv1(x))
        # print("Value of x in pi after conv1 ")
        # print(x[0])

        # x = self.swish(self.conv2(x))
        # print("Value of x in pi after conv2")
        # print(x[0])

        # print("************ X ******************")
        # print(x.shape)
        # print(x)

        # x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        # print(x.size(0))
        # print("*****************before resize *  **************")
        # print(x.shape)
        x = x.reshape(x.size(0),self.linear_input_size)
        # print(f"Shape after flattening in pi: {x.shape}")
        # print("*************************************")
        # print("Shape of x after conv layer")
        # print(x.shape)
        # print(x)
        # print("Value of x in pi")
        # print(x)
        

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc_pi(x)
        # print("Shape of x after fc_pi layer")
        # print(x.shape)
        
        return Categorical(logits=x)

    def v(self, x):
        x = self.swish(self.conv1(x))
        # x = self.swish(self.conv2(x))
        print("*************************************")
        # print("Shape of x after conv layer in v")
        # print(x.shape)

        # x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layers
        # print(x.size(0))
        # print("*****************before resize *  **************")
        # print(x)
        x = x.reshape(x.size(0),self.linear_input_size)
        # print("Value of x in v")
        # print(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc_v(x)
        return v




# Define the PPO class
# Define the PPOAgent class
class PPOAgent:
    def __init__(self, height, width, action_dim=4, buffer_size=10000, gamma=0.99,
                  K_epochs=4, eps_clip=0.2, hidden_dim=128, device=None):
        self.policy = ActorCritic(height, width, hidden_dim, action_dim)
        self.policy_old = ActorCritic(height, width, hidden_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = self.policy.optimizer
        self.MseLoss = nn.MSELoss()
        self.memory = deque(maxlen=buffer_size)
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.rewards = []

    def update(self):
        states, actions, logprobs, rewards, is_terminals = zip(*self.memory)

        discounted_rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        print(f"Discounted Rewards: {discounted_rewards}")
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        

        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)
        


        old_states = torch.cat(states).detach()
        old_actions = torch.cat(actions).detach()
        old_logprobs = torch.cat(logprobs).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = discounted_rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, discounted_rewards) - 0.01 * dist_entropy
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, discounted_rewards) - 0.02 * dist_entropy  # Example adjustment
            # print(f"Discounted rewards: {discounted_rewards}")
            # print(f"State values: {state_values}")
            # print(f"ratios: {ratios}")
            # print(f"Advantages: {advantages}")
            # print(f"Surr1 : {surr1}")
            # print(f"Surr2 : {surr2}")
            # print(f"Loss: {loss}")

            self.optimizer.zero_grad()
            loss.mean().backward()
            if torch.isnan(loss).any():
                print("NaN detected in loss!")
                raise ValueError("NaN in loss detected.")

            
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)    

            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    
    def evaluate(self, state, action):
        state_value = self.policy.v(state)
        # print("*************************************")
        # print(f"State: {state}")
        dist = self.policy.pi(state)

        # Debugging: Print logits and check for NaNs
        # print("Logits: ", dist.logits)
        
        if torch.isnan(dist.logits).any():
            print("NaN detected in logits!")
            dist.logits = torch.where(torch.isnan(dist.logits), torch.zeros_like(dist.logits), dist.logits)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy


    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=self.device))
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"Model loaded from {filename}")

    def normalize_state(self, state):
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def train(self, env, num_episodes, early_stopping=None, checkpoint_path=None):
        for episode in range(1, num_episodes + 1):
            total_reward = 0
            state = env.reset()
            state = self.normalize_state(state)
            done = False
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add channel dimension
                
                dist = self.policy_old.pi(state_tensor)
                action = dist.sample()

                next_state, reward, done, _ = env.step(action.item())
                # next_state = self.normalize_state(next_state)

                self.memory.append((state_tensor, action, dist.log_prob(action), reward, done))

                state = next_state
                total_reward += reward

                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    break
            # print(self.memory)
            # print(len(self.memory))
            if len(self.memory)>1000:
                print("Updating")
                self.update()
                self.memory.clear()
            self.rewards.append(total_reward)

            if early_stopping and early_stopping(self.rewards):
                print("Early stopping criterion met")
                if checkpoint_path:
                    self.save(checkpoint_path)
                break
            if (episode+1) % 1000 == 0:
                self.save(checkpoint_path)

        env.close()

    def test(self, env, num_episodes=10):
        for episode in range(num_episodes):
            state = env.reset()
            # state = self.normalize_state(state)
            done = False
            total_reward = 0
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add channel dimension
                dist = self.policy_old.pi(state_tensor)
                action = dist.sample()
                state, reward, done, _ = env.step(action.item())
                # state = self.normalize_state(state)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            self.rewards.append(total_reward)
        env.close()

    def plot(self, plot_path):
        data = self.rewards

        # Calculate the moving average
        window_size = 10
        moving_avg = pd.Series(data).rolling(window=window_size).mean()

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot the moving average line
        sns.lineplot(data=moving_avg, color='red')

        # Shade the area around the moving average line to represent the range of values
        plt.fill_between(range(len(moving_avg)),
                         moving_avg - np.std(data),
                         moving_avg + np.std(data),
                         color='blue', alpha=0.2)

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Moving Average of Rewards')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as a PNG file
        plt.savefig(plot_path)
        # Show the plot
        plt.show()
