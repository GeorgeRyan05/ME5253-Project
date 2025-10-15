import numpy as np

import torch

from torch import nn
from dist_ac import DistributedActorCritic
from cooperative_environment import CooperativeNavigationEnvironment


class Predictor:
    def __init__(self, n_phi, n_hidden, n_actions, device) -> None:
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(n_phi, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_phi, 1),
        )
        self.actor.to(self.device)
        self.critic.to(self.device)
        self._actor_optimizer = torch.optim.SGD(self.actor.parameters(), lr=0.01)
        self._critic_optimizer = torch.optim.SGD(self.critic.parameters(), lr=0.01)


class DistributedActorCriticNonLinear(DistributedActorCritic):
    n_hidden: int = 24

    def __init__(self, env: CooperativeNavigationEnvironment, device=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        super(DistributedActorCriticNonLinear, self).__init__(env)
        self.predictors: list[Predictor] = []
        for agent in range(self.n_agents):
            self.predictors.append(
                Predictor(self.n_phi, self.n_hidden, self.n_actions, self.device)
            )
        # Only for ensuring all gradients are zeroed

    def update(self, state, actions, rewards, next_state, next_actions, C):
        """Updates actor and critic parameters

        Parameters:
        -----------
        * state: tuple<np.array<n_phi>, np.array<n_actions, n_agents, n_varphi>>
            features representing the state where
            state[0]: phi represents the state at time t as seen by the critic.
            state[1]: varphi represents the state at time t as seen by the actor.
        * actions: np.array<n_agents>
            Actions for each agent at time t.
        * rewards: np.array<n_agents>
            Instantaneous rewards for each of the agents.
        * next_state: tuple<np.array<n_phi>, np.array<n_actions, n_agents, n_varphi>>
            features representing the state where
            next_state[0]: phi represents the state at time t+1 as seen by the critic.
            next_state[1]: varphi represents the state at time t+1 as seen by the actor.
        * next_actions: tuple(float<>, float<>)
            Actions for each agent at time t+1.
        """
        # NOTE - varphi should be replaced with phi for these functions, as the actions are handled on the output side, not the input layer
        # 1. Common knowledge at timestep-t
        phi, varphi = self.env.get_features(state, actions)
        next_phi, _ = self.env.get_features(next_state, next_actions)

        # dq = self.grad_q(phi)
        alpha = self.alpha
        beta = self.beta
        mu = self.mu
        advantages = []
        deltas = []
        grad_ws = []
        grad_thetas = []
        scores = []

        ws = [self.w.tolist()]
        thetas = [self.theta.tolist()]
        # wtilde = np.zeros_like(self.w)
        # 2. Iterate agents on the network.
        for i in range(self.n_agents):
            # 2.1 Compute time-difference delta
            with torch.no_grad():
                delta = rewards[i] - mu[i] + self.q(next_phi, i) - self.q(phi, i)
            predictor = self.predictors[i]
            predictor._critic_optimizer.zero_grad()
            predictor._actor_optimizer.zero_grad()
            q = self._q(phi, i)
            q.backward()

            # 2.2 Critic step
            for param in predictor.critic.parameters():
                param.data += param.grad * alpha * delta
            # grad_w = alpha * delta * dq
            # wtilde[i, :] = self.w[i, :] + grad_w  # [n_phi,]

            # 3.3 Actor step
            adv = self.advantage(phi, varphi, state, actions, i)  # [n_varphi,]
            # ksi = self.grad_log_policy(varphi, actions, i)  # [n_varphi,]
            prob = self._policy(varphi, i)
            log_prob = torch.log(prob[actions[i]])
            log_prob.backward()
            # grad_theta = beta * adv * ksi
            # self.theta[i, :] += grad_theta  # [n_varphi,]

            # Log step
            # grad_thetas.append(grad_theta.tolist())
            # scores.append(ksi.tolist())

            advantages.append(adv)
            deltas.append(float(delta))

        print(f"{C = }")
        # Consensus step.
        self.w = C @ wtilde

        # Each agent is independent, so backward pass can be done independently
        for param in predictor.actor.parameters():
            param.data += beta * delta * param.grad
        # Log.
        ws.append(self.w.tolist())
        thetas.append(self.theta.tolist())

        self.n_steps += 1
        self.mu = self.next_mu
        return advantages, deltas, ws, grad_ws, thetas, grad_thetas, scores

    def _q(self, phi: np.ndarray, i: int) -> torch.Tensor:
        """Q-function

        Parameters:
        -----------
        * phi: np.array<n_phi>
            critic features

        Returns:
        --------
        * q: float
            q-value for agent i
        """
        _phi = torch.tensor(phi, dtype=torch.float32).to(self.device)
        predictor = self.predictors[i]
        q = predictor.critic(_phi)
        return q

    def q(self, phi: np.ndarray, i: int):
        """Q-function

        Parameters:
        -----------
        * phi: np.array<n_phi>
            critic features

        Returns:
        --------
        * q: float
           Q-value for agent i
        """
        with torch.no_grad():
            q = self._q(phi, i)
        return q.cpu().numpy()

    def grad_q(self, phi):
        raise NotImplementedError

    def _policy(self, phi: np.ndarray, i: int) -> torch.Tensor:
        """
        Compute policy for state phi and agent i.
        """
        _phi = torch.tensor(phi, dtype=torch.float32).to(self.device)
        predictor = self.predictors[i]
        probs = predictor.actor(_phi)
        return probs

    def policy(self, varphi, i):
        """Computes gibbs distribution / Boltzman policies

        Parameters:
        -----------
        * varphi: np.array<n_actions, n_agents, n_varphi>
            actor features


        * i: integer
            index of the agent on the interval {0,N-1}

        Returns:
        -------
        * probabilities: np.array<n_actions>
            Stochastic policy
        """
        with torch.no_grad():
            z = self._policy(varphi, i)[i].cpu().numpy()

        return z

    def grad_log_policy(self, varphi, actions, i):
        raise NotImplementedError
