import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import OUNoise
from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent:
    def __init__(self, state_size, action_size, num_agents, random_seed, buffer_size=int(1e6), batch_size=128, gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-3):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # === Actor Networks (separate per agent) ===
        self.actors_local = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actors_target = [Actor(state_size, action_size, random_seed).to(device) for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors_local]

        # === Shared Critic Network ===
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0)

        # === Noise process
        self.noise = OUNoise(action_size, random_seed)

        # === Replay Buffer (shared) ===
        self.memory = ReplayBuffer(buffer_size, batch_size, random_seed)

        # === Initialize targets ===
        for i in range(num_agents):
            self.soft_update(self.actors_local[i], self.actors_target[i], 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience and train."""
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        if len(self.memory) > self.batch_size:
            for i in range(self.num_agents):
                experiences = self.memory.sample()
                self.learn(experiences, i)

    def act(self, states, noise=0.0):
        """Returns actions for each agent given their states."""
        actions = []
        for i in range(self.num_agents):
            state = torch.from_numpy(states[i]).float().to(device)
            self.actors_local[i].eval()
            with torch.no_grad():
                action = self.actors_local[i](state).cpu().data.numpy()
            self.actors_local[i].train()
            action += noise * np.random.randn(self.action_size)
            actions.append(np.clip(action, -1, 1))
        return actions
    
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, agent_index):
        """Update actor and critic networks for a specific agent."""
        states, actions, rewards, next_states, dones = experiences

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # --- Critic update (shared) ---
        with torch.no_grad():
            next_actions = self.actors_target[agent_index](next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update (individual) ---
        predicted_actions = self.actors_local[agent_index](states)
        actor_loss = -self.critic_local(states, predicted_actions).mean()
        self.actor_optimizers[agent_index].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_index].step()

        # --- Soft update of target networks ---
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actors_local[agent_index], self.actors_target[agent_index], self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class ReplayBuffer:
    """Shared replay buffer for all agents."""

    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
