import numpy as np
import torch
from torch import nn

from .ac import ActorCritic


class NonLinearActorCritic(ActorCritic):
    n_hidden: int = 24

    def __init__(self, env, device=None):
        super(NonLinearActorCritic, self).__init__(env)
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(self.n_varphi, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_actions * self.n_nodes),
            nn.Softmax(dim=-1),
        )
        self.actor.to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(self.n_phi, 1),
        )
        self._actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.01)
        self._critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)
        # Only for ensuring all gradients are zeroed
        self.critic.to(self.device)

    def update(self, state, actions, reward, next_state, next_actions):
        """Updates actor and critic parameters

        Parameters:
        -----------
        * state: tuple<np.array<n_phi>, np.array<n_actions, n_nodes, n_varphi>>
            features representing the state where
            state[0]: phi represents the state at time t as seen by the critic.
            state[1]: varphi represents the state at time t as seen by the actor.
        * actions: np.array<n_nodes>
            Actions for each agent at time t.
        * rewards: np.array<n_nodes>
            Instantaneous rewards for each of the agents.
        * next_state: tuple<np.array<n_phi>, np.array<n_actions, n_nodes, n_varphi>>
            features representing the state where
            next_state[0]: phi represents the state at time t+1 as seen by the critic.
            next_state[1]: varphi represents the state at time t+1 as seen by the actor.
        * next_actions: tuple(float<>, float<>)
            Actions for each agent at time t+1.
        """
        # Common knowledge at timestep-t
        phi, varphi = self.env.get_features(state, actions)
        next_phi, _ = self.env.get_features(next_state, next_actions)
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        mu = self.mu
        with torch.no_grad():
            delta = np.mean(reward) - mu + self.q(next_phi) - self.q(phi)
        q = self._q(phi)
        q.backward()  # propagate gradients
        alpha = self.alpha
        beta = self.beta

        # Log variables.
        advantages = []
        deltas = []
        grad_ws = []
        grad_thetas = []
        scores = []

        # Capture before updates.
        ws = [self.w.tolist()]
        thetas = [self.theta.tolist()]

        # 3.2 Critic step
        # [n_phi,]
        # grad_w = alpha * delta * dq
        # next_w = self.w + grad_w
        for param in self.critic.parameters():
            param.data += param.grad * alpha * delta

        # 3.3 Actor step
        for i in range(self.n_nodes):
            prob = self._policy(varphi, i)
            # ksi = self.grad_log_policy(varphi, actions, i)  # [n_phi]
            # print(prob.shape)
            log_prob = torch.log(prob[actions[i]])
            log_prob.backward()
        # Will accumulate gradients
        for param in self.actor.parameters():
            param.data += beta * delta * param.grad

        self.n_steps += 1
        self.mu = self.next_mu

        return advantages, float(delta), ws, grad_ws, thetas, grad_thetas, scores

    def _policy(self, varphi, i) -> torch.Tensor:
        # varphi is the state vector, in this case
        # [n_actions, n_nodes, n_varphi]
        # print(f"{varphi[:, i, :].shape = }")
        _varphi = torch.tensor(varphi[:, i, :], dtype=torch.float32, device=self.device)
        probs = self.actor(_varphi)
        return probs

    def policy(self, varphi, i):
        with torch.no_grad():
            return self._policy(varphi, i)[i].cpu().numpy()

    def _q(self, phi) -> torch.Tensor:
        _phi = torch.tensor(phi, dtype=torch.float32, device=self.device)
        q_value = self.critic(_phi)
        return q_value

    def q(self, phi):
        with torch.no_grad():
            return self._q(phi).cpu().numpy()
