import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from wordle_env import WordleEnv

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class WordleAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)

        # Replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration-exploitation balance
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = 0.001  # soft update parameter

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

        # Learn every time step
        if len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

    def act(self, state):
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Get the expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Get the target Q values from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Example usage
if __name__ == "__main__":
    word_list = ['apple', 'grape', 'berry', 'melon', 'lemon']
    env = WordleEnv(word_list)
    agent = WordleAgent(state_size=5, action_size=len(word_list), seed=0)
    
    num_episodes = 1000
    for i_episode in range(1, num_episodes+1):
        state = env.reset()[0]  # Gymnasium returns (observation, info), so take the first element
        total_reward = 0
        for t in range(env.max_attempts):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        print(f"Episode {i_episode}, Total Reward: {total_reward}")

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}: Epsilon = {agent.epsilon}")

    print("Training complete!")
