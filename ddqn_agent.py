import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)  # fc3 activation
        # Chọn activation cuối: tanh hoặc softmax
        # x = torch.tanh(self.fc4(x))        # option 1: tanh
        # # x = F.softmax(self.fc4(x), dim=1)  # option 2: softmax nếu muốn xác suất
        x = self.fc4(x)
        return x


class DDQNAgent:
    def __init__(self, state_size, action_size,
                 lr=1e-4, gamma=0.95,
                 epsilon=1.0, eps_decay=0.995, eps_min=0.01,
                 memory_size=10000, batch_size=64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)

        self.q_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict()) 

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

 
    def remember(self, state, action, next_state,reward, done=False):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            action_idx = torch.argmax(q_values).item()

        return action_idx

    def update(self):
        if len(self.memory) < self.batch_size:
            return  
        
        print("update ------ model")
        batch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([[b[1]] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])

        q_values = self.q_net(states).gather(1, actions)

        next_actions = torch.argmax(self.q_net(next_states), dim=1, keepdim=True)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss & update
        loss = self.criterion(q_values.squeeze(1), target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def save(self, filepath="ddqn_agent.pth"):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    def load(self, filepath="ddqn_agent.pth"):
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        print(f"Model loaded from {filepath}")
