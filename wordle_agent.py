import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

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
        
        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
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
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

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
