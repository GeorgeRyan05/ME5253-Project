"""
Distributed Actor-Critic Algorithm (Algorithm 1) Implementation
Based on action-value function approximation with networked agents.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import json
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)


class NetworkedEnvironment:
    """Multi-agent networked MDP environment"""

    def __init__(self, n_agents: int = 20, n_states: int = 20):
        self.N = n_agents  # Number of agents
        self.S = n_states  # Number of states
        self.A_i = 2  # Binary action space for each agent {0, 1}
        self.A_total = 2**n_agents  # Total joint action space

        # Generate transition probability matrix P
        # P[s, a, s'] = probability of transitioning from s to s' under joint action a
        self.P = self._generate_transition_probabilities()

        # Generate reward functions for each agent
        # R_i[s, a] = mean reward for agent i at state s under joint action a
        self.R = self._generate_rewards()

        # Current state
        self.current_state = np.random.randint(0, self.S)

    def _generate_transition_probabilities(self) -> np.ndarray:
        """Generate transition probability matrix"""
        # For computational efficiency, we'll use a simplified transition model
        # that only depends on state (not full joint action space)
        P = np.random.uniform(0, 1, (self.S, self.S))
        # Add small constant for ergodicity
        P += 1e-5
        # Normalize to make stochastic
        P = P / P.sum(axis=1, keepdims=True)
        return P

    def _generate_rewards(self) -> np.ndarray:
        """Generate mean rewards for each agent and state-action pair"""
        # R[i, s, a] where a is the joint action index
        # For computational efficiency, we simplify to R[i, s, a_i]
        R = np.random.uniform(0, 4, (self.N, self.S, self.A_i))
        return R

    def reset(self) -> int:
        """Reset environment to random initial state"""
        self.current_state = np.random.randint(0, self.S)
        return self.current_state

    def step(self, actions: np.ndarray) -> Tuple[int, np.ndarray, bool]:
        """
        Take a step in the environment
        Args:
            actions: array of shape (N,) with each agent's action
        Returns:
            next_state, rewards, done
        """
        s = self.current_state

        # Sample next state based on transition probabilities
        next_state = np.random.choice(self.S, p=self.P[s])

        # Sample rewards for each agent
        rewards = np.zeros(self.N)
        for i in range(self.N):
            mean_reward = self.R[i, s, actions[i]]
            rewards[i] = np.random.uniform(mean_reward - 0.5, mean_reward + 0.5)

        self.current_state = next_state
        done = False  # Continuing task

        return next_state, rewards, done


class BoltzmannPolicy:
    """Boltzmann policy for each agent"""

    def __init__(self, n_states: int, n_actions: int, feature_dim: int = 5):
        self.S = n_states
        self.A = n_actions
        self.m = feature_dim

        # Initialize policy parameters theta
        self.theta = np.random.randn(self.m) * 0.1

        # Generate feature vectors q_{s,a} for each state-action pair
        self.q = np.random.uniform(0, 1, (self.S, self.A, self.m))

    def get_probability(self, state: int, action: int) -> float:
        """Get probability of taking action at state"""
        logits = np.array([np.dot(self.q[state, a], self.theta) for a in range(self.A)])
        # Numerical stability: subtract max
        logits = logits - np.max(logits)
        logits = np.clip(logits, -50, 50)  # Prevent overflow
        exp_logits = np.exp(logits)
        return exp_logits[action] / (np.sum(exp_logits) + 1e-10)

    def get_action_probabilities(self, state: int) -> np.ndarray:
        """Get probability distribution over actions at state"""
        logits = np.array([np.dot(self.q[state, a], self.theta) for a in range(self.A)])
        # Numerical stability: subtract max
        logits = logits - np.max(logits)
        logits = np.clip(logits, -50, 50)  # Prevent overflow
        exp_logits = np.exp(logits)
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)
        return probs

    def sample_action(self, state: int) -> int:
        """Sample an action from the policy"""
        probs = self.get_action_probabilities(state)
        return np.random.choice(self.A, p=probs)

    def gradient_log_policy(self, state: int, action: int) -> np.ndarray:
        """Compute gradient of log policy"""
        probs = self.get_action_probabilities(state)

        # ∇_θ log π_θ(s,a) = q_{s,a} - Σ_b π_θ(s,b) q_{s,b}
        grad = self.q[state, action].copy()
        for a in range(self.A):
            grad -= probs[a] * self.q[state, a]

        return grad


class ActionValueFunction:
    """Linear action-value function approximation Q(s,a; ω)"""

    def __init__(self, n_states: int, n_agents: int, feature_dim: int = 10):
        self.S = n_states
        self.N = n_agents
        self.K = feature_dim

        # Initialize weight vector omega
        self.omega = np.random.randn(self.K) * 0.1

        # Generate feature vectors φ_{s,a} for state-joint action pairs
        # For computational efficiency, we use features based on state and individual actions
        self.phi = np.random.uniform(
            0, 1, (self.S, 2**n_agents if n_agents <= 5 else 100, self.K)
        )

        # For large action spaces, we'll use a hash function to map joint actions to features
        self.use_hash = n_agents > 5

    def _joint_action_to_index(self, actions: np.ndarray) -> int:
        """Convert joint action array to index"""
        if self.use_hash:
            # Use hash for large action spaces
            return hash(tuple(actions)) % 100
        else:
            # Binary encoding for small action spaces
            return sum(actions[i] * (2**i) for i in range(len(actions)))

    def get_value(self, state: int, actions: np.ndarray) -> float:
        """Get Q-value for state and joint action"""
        action_idx = self._joint_action_to_index(actions)
        return np.dot(self.phi[state, action_idx], self.omega)

    def get_gradient(self, state: int, actions: np.ndarray) -> np.ndarray:
        """Get gradient of Q with respect to omega"""
        action_idx = self._joint_action_to_index(actions)
        return self.phi[state, action_idx]


class DistributedActorCritic:
    """Distributed Actor-Critic Algorithm (Algorithm 1)"""

    def __init__(self, n_agents: int = 20, n_states: int = 20, discount: float = 0.99):
        self.N = n_agents
        self.S = n_states
        self.gamma = discount  # Discount factor

        # Initialize environment
        self.env = NetworkedEnvironment(n_agents, n_states)

        # Initialize policies for each agent
        self.policies = [
            BoltzmannPolicy(n_states, 2, feature_dim=20) for _ in range(n_agents)
        ]

        # Initialize action-value function (shared)
        self.Q = ActionValueFunction(n_states, n_agents, feature_dim=40)

        # Initialize consensus weights for action-value function
        self.omega_bars = [np.copy(self.Q.omega) for _ in range(n_agents)]

        # Initialize average reward baseline for each agent
        self.mu = [0.0 for _ in range(n_agents)]

        # Generate communication graph (random with connectivity ratio 4/N)
        self.C = self._generate_consensus_matrix()

        # Iteration counter
        self.t = 0

        # History for tracking
        self.returns_history = []
        self.avg_returns_history = []

    def _generate_consensus_matrix(self) -> np.ndarray:
        """Generate doubly stochastic consensus weight matrix"""
        # Generate random graph with connectivity ratio 4/N
        connectivity_ratio = 4.0 / self.N
        total_edges = int(connectivity_ratio * self.N * (self.N - 1) / 2)

        # Create adjacency matrix
        adj = np.zeros((self.N, self.N))
        edges_added = 0

        while edges_added < total_edges:
            i = np.random.randint(0, self.N)
            j = np.random.randint(0, self.N)
            if i != j and adj[i, j] == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges_added += 1

        # Create Laplacian matrix
        D = np.diag(adj.sum(axis=1))
        L = D - adj

        # Normalize to create doubly stochastic matrix
        # C = I - αL where α is chosen to make C doubly stochastic
        alpha = 0.5 / (np.max(D) + 1e-8)
        C = np.eye(self.N) - alpha * L

        # Ensure doubly stochastic (rows and columns sum to 1)
        C = np.abs(C)
        C = C / C.sum(axis=1, keepdims=True)

        return C

    def _get_stepsize_omega(self, t: int) -> float:
        """Get stepsize for critic update"""
        return 0.5 / ((t + 10) ** 0.6)

    def _get_stepsize_theta(self, t: int) -> float:
        """Get stepsize for actor update"""
        return 0.1 / ((t + 10) ** 0.75)

    def run_episode(self, max_steps: int = 1000) -> float:
        """Run one episode of the algorithm"""
        # Reset environment
        s_t = self.env.reset()

        # Sample initial actions from each agent's policy
        actions_t = np.array(
            [self.policies[i].sample_action(s_t) for i in range(self.N)]
        )

        episode_return = 0.0
        returns = []

        for step in range(max_steps):
            self.t += 1

            # Get stepsizes
            beta_omega = self._get_stepsize_omega(self.t)
            beta_theta = self._get_stepsize_theta(self.t)

            # --- Execute actions and observe next state and rewards ---
            s_next, rewards, _ = self.env.step(actions_t)

            # Sample next actions
            actions_next = np.array(
                [self.policies[i].sample_action(s_next) for i in range(self.N)]
            )

            # Store TD errors for actor update
            deltas = []

            # --- Critic Step (for each agent) ---
            for i in range(self.N):
                # Update average reward baseline
                # μ^i_{t+1} = (1 - β_{ω,t}) · μ^i_t + β_{ω,t} · r'^i_{t+1}
                self.mu[i] = (1 - beta_omega) * self.mu[i] + beta_omega * rewards[i]

                # Compute Q-values
                Q_current = self.Q.get_value(s_t, actions_t)
                Q_next = self.Q.get_value(s_next, actions_next)

                # Compute TD error: δ^i = r'^i_{t+1} - μ^i_t + Q_{t+1} - Q_t
                delta_i = rewards[i] - self.mu[i] + Q_next - Q_current
                deltas.append(delta_i)

                # Update critic: ω̄^i_{t+1} = ω̄^i_t + β_{ω,t} · δ^i_t · ∇_ω Q_t
                grad_Q = self.Q.get_gradient(s_t, actions_t)
                self.omega_bars[i] = self.omega_bars[i] + beta_omega * delta_i * grad_Q

            # --- Actor Step (for each agent) ---
            for i in range(self.N):
                # Compute eligibility trace: ψ^i_t = ∇_{θ^i} log π^i_{θ^i}(s_t, a^i_t)
                psi_i = self.policies[i].gradient_log_policy(s_t, actions_t[i])

                # Compute gradient with clipping
                grad = beta_theta * deltas[i] * psi_i
                grad = np.clip(grad, -1.0, 1.0)  # Gradient clipping

                # Update actor using TD error: θ^i_{t+1} = θ^i_t + β_{θ,t} · δ^i_t · ψ^i_t
                self.policies[i].theta = self.policies[i].theta + grad

                # Clip theta to prevent explosion
                self.policies[i].theta = np.clip(self.policies[i].theta, -10, 10)

            # --- Consensus Step ---
            # Share omega_bars with neighbors and perform consensus
            new_omega_bars = []
            for i in range(self.N):
                # ω^i_{t+1} = Σ_{j∈N} c(i,j) · ω̄^j_{t+1}
                omega_consensus = np.zeros_like(self.omega_bars[i])
                for j in range(self.N):
                    omega_consensus += self.C[i, j] * self.omega_bars[j]
                new_omega_bars.append(omega_consensus)

            self.omega_bars = new_omega_bars

            # Update state and actions for next iteration
            s_t = s_next
            actions_t = actions_next

            # Track returns
            avg_reward = np.mean(rewards)
            episode_return += avg_reward
            returns.append(avg_reward)

        # Compute average return for this episode
        avg_return = episode_return / max_steps
        self.returns_history.append(returns)
        self.avg_returns_history.append(avg_return)

        return avg_return

    def train(self, n_episodes: int = 20, steps_per_episode: int = 1000):
        """Train the distributed actor-critic algorithm"""
        print(f"Training Distributed Actor-Critic with {self.N} agents...")
        print(f"Episodes: {n_episodes}, Steps per episode: {steps_per_episode}")

        for episode in range(n_episodes):
            avg_return = self.run_episode(steps_per_episode)

            if (episode + 1) % 5 == 0:
                print(
                    f"Episode {episode + 1}/{n_episodes}, "
                    f"Avg Return: {avg_return:.4f}"
                )

        print("Training completed!")

    def plot_results(self, window_size: int = 100):
        """Plot the globally averaged returns over time"""
        # Flatten the returns history
        all_returns = []
        for episode_returns in self.returns_history:
            all_returns.extend(episode_returns)

        # Compute moving average for smoother plot
        if len(all_returns) > window_size:
            moving_avg = np.convolve(
                all_returns, np.ones(window_size) / window_size, mode="valid"
            )
            x_axis = np.arange(window_size - 1, len(all_returns))
        else:
            moving_avg = np.array(all_returns)
            x_axis = np.arange(len(all_returns))

        plt.figure(figsize=(10, 6))
        plt.plot(
            x_axis,
            moving_avg,
            linewidth=2,
            label="Algorithm 1 (Distributed AC)",
            color="#1f77b4",
        )
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Globally Averaged Return", fontsize=12)
        plt.title("Convergence of Globally Averaged Returns", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc="lower right")
        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"distributed_ac_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as: {filename}")

        plt.show()

        return all_returns


def main():
    """Main function to run the distributed actor-critic algorithm"""
    print("=" * 70)
    print("Distributed Actor-Critic Algorithm Implementation")
    print("Based on Action-Value Function Approximation")
    print("=" * 70)
    print()

    # Initialize the algorithm
    n_agents = 20
    n_states = 20

    algo = DistributedActorCritic(n_agents=n_agents, n_states=n_states)

    # Train the algorithm
    # Run for longer to see better convergence (similar to paper)
    n_episodes = 10
    steps_per_episode = 3000

    algo.train(n_episodes=n_episodes, steps_per_episode=steps_per_episode)

    # Plot results
    all_returns = algo.plot_results(window_size=100)

    # Print final statistics
    print("\n" + "=" * 70)
    print("Training Statistics")
    print("=" * 70)
    print(f"Total iterations: {len(all_returns)}")
    print(f"Final average return (last 100 steps): {np.mean(all_returns[-100:]):.4f}")
    print(f"Overall average return: {np.mean(all_returns):.4f}")
    print()


if __name__ == "__main__":
    main()
