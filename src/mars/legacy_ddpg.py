from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass
class DDPGBetaConfig:
    state_dim: int = 24
    hidden_dim: int = 64
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.95
    tau: float = 0.02
    replay_capacity: int = 1000
    batch_size: int = 16
    exploration_noise: float = 0.05
    min_beta: float = 0.0
    max_beta: float = 1.0


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.items: deque[tuple[np.ndarray, float, float, np.ndarray]] = deque(maxlen=capacity)

    def append(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray) -> None:
        self.items.append((state.astype(np.float32), float(action), float(reward), next_state.astype(np.float32)))

    def sample(self, batch_size: int):
        batch = random.sample(self.items, min(batch_size, len(self.items)))
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32).view(-1, 1),
            torch.tensor(rewards, dtype=torch.float32).view(-1, 1),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.items)


class DDPGBetaAgent:
    """Minimal DDPG agent for continuous beta_explore selection."""

    def __init__(self, config: DDPGBetaConfig, seed: int = 0) -> None:
        self.config = config
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.actor = Actor(config.state_dim, config.hidden_dim)
        self.critic = Critic(config.state_dim, config.hidden_dim)
        self.target_actor = Actor(config.state_dim, config.hidden_dim)
        self.target_critic = Critic(config.state_dim, config.hidden_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.replay = ReplayBuffer(config.replay_capacity)
        self.previous_state: np.ndarray | None = None
        self.previous_action: float | None = None
        self.last_actor_loss: float | str = "NA"
        self.last_critic_loss: float | str = "NA"

    def select_beta(self, state: np.ndarray, train: bool = True) -> float:
        state = self._shape_state(state)
        with torch.no_grad():
            action = float(self.actor(torch.tensor(state, dtype=torch.float32).view(1, -1)).item())
        if train and self.config.exploration_noise > 0:
            action += float(np.random.normal(0.0, self.config.exploration_noise))
        action = float(np.clip(action, self.config.min_beta, self.config.max_beta))
        self.previous_state = state
        self.previous_action = action
        return action

    def observe(self, reward: float, next_state: np.ndarray) -> dict:
        if self.previous_state is None or self.previous_action is None:
            return {"actor_loss": "NA", "critic_loss": "NA", "replay_size": len(self.replay)}
        next_state = self._shape_state(next_state)
        self.replay.append(self.previous_state, self.previous_action, reward, next_state)
        self._train_step()
        return {"actor_loss": self.last_actor_loss, "critic_loss": self.last_critic_loss, "replay_size": len(self.replay)}

    def _train_step(self) -> None:
        if len(self.replay) < max(2, self.config.batch_size):
            self.last_actor_loss = "NA"
            self.last_critic_loss = "NA"
            return
        states, actions, rewards, next_states = self.replay.sample(self.config.batch_size)
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + self.config.gamma * self.target_critic(next_states, next_actions)
        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        self.last_actor_loss = float(actor_loss.item())
        self.last_critic_loss = float(critic_loss.item())

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1.0 - self.config.tau) * target_param.data + self.config.tau * source_param.data)

    def _shape_state(self, state: np.ndarray) -> np.ndarray:
        state = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        if state.size >= self.config.state_dim:
            return state[: self.config.state_dim]
        return np.pad(state, (0, self.config.state_dim - state.size), mode="constant")
