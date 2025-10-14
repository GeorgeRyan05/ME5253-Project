import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append("d:/Workspace/GitRepos/ME5253/networked_agents")

from environment import Environment


class NetworkedActorCritic:
    def __init__(self, env, gamma=0.95, seed=42):
        self.env = env
        self.N = env.n_nodes  # Number of agents
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.n_phi = env.n_phi  # Feature dimension for Q
        self.n_varphi = env.n_varphi  # Feature dimension for policy
        self.gamma = gamma

        np.random.seed(seed)

        # Initialize parameters for each agent
        # Critic parameters: ω^i for Q-function (uses PHI features from environment)
        self.omega = np.zeros((self.N, self.n_phi))

        # Actor parameters: θ^i for policy (uses VARPHI features from environment)
        self.theta = np.random.randn(self.N, self.n_varphi) * 0.01

        # Average reward baseline for each agent
        self.mu = np.zeros(self.N)

        # Consensus weights (will be updated each iteration)
        self.C = np.eye(self.N)

        # Store omega_bar for consensus
        self.omega_bar = np.zeros((self.N, self.n_phi))

    def get_policy_features(self, state, action, agent_i):
        """Get policy features varphi(s, a_i) for agent i"""
        # VARPHI shape: [n_states, n_actions, n_nodes, n_varphi]
        varphi = self.env.VARPHI[state, action, agent_i, :]
        return varphi

    def get_q_features(self, state, joint_actions):
        """Get Q-function features phi(s, a) for joint actions"""
        # PHI shape: [n_states * n_action_space, n_phi]
        phi = self.env.get_phi(state, joint_actions)
        return phi

    def compute_policy(self, state, agent_i):
        """Compute Boltzmann policy π_θ(s, a_i) for agent i"""
        logits = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            varphi = self.get_policy_features(state, a, agent_i)
            logits[a] = np.dot(varphi, self.theta[agent_i])

        # Softmax with numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(np.clip(logits, -10, 10))
        probs = exp_logits / (np.sum(exp_logits) + 1e-10)

        return probs

    def sample_action(self, state, agent_i):
        """Sample action for agent i according to policy"""
        probs = self.compute_policy(state, agent_i)
        action = np.random.choice(self.n_actions, p=probs)
        return action

    def compute_Q_value(self, state, joint_actions, agent_i):
        """Compute Q(s, a; ω^i) = φ(s, a)^T · ω^i"""
        phi = self.get_q_features(state, joint_actions)
        Q_value = np.dot(phi, self.omega[agent_i])
        return Q_value

    def compute_policy_gradient(self, state, action, agent_i):
        """Compute ∇_θ log π_θ(s, a_i)"""
        # ∇_θ log π_θ(s, a_i) = varphi(s, a_i) - Σ_b π_θ(s, b) varphi(s, b)
        varphi_a = self.get_policy_features(state, action, agent_i)

        # Compute expectation
        probs = self.compute_policy(state, agent_i)
        expected_varphi = np.zeros(self.n_varphi)
        for b in range(self.n_actions):
            varphi_b = self.get_policy_features(state, b, agent_i)
            expected_varphi += probs[b] * varphi_b

        grad = varphi_a - expected_varphi
        return grad

    def run_episode(self, n_steps=1000):
        """Run one episode of the algorithm"""
        # Reset environment
        state = self.env.state

        # Initialize
        returns_history = []

        # Sample initial joint actions
        joint_actions = np.array([self.sample_action(state, i) for i in range(self.N)])

        for t in range(n_steps):
            # Stepsizes
            beta_omega = 1.0 / (t + 100) ** 0.65
            beta_theta = 1.0 / (t + 100) ** 0.85
            beta_mu = 1.0 / (t + 100) ** 0.65

            # Take action and observe next state and rewards
            rewards = self.env.get_rewards(joint_actions)
            self.env.next_step(joint_actions)
            next_state = self.env.state

            # Sample next joint actions
            next_joint_actions = np.array(
                [self.sample_action(next_state, i) for i in range(self.N)]
            )

            # Update consensus weights
            self.C = self.env.get_consensus()

            # Critic step: Update Q-function parameters for each agent
            for i in range(self.N):
                # Compute Q values
                Q_curr = self.compute_Q_value(state, joint_actions, i)
                Q_next = self.compute_Q_value(next_state, next_joint_actions, i)

                # TD error: δ^i = r'^i_{t+1} - μ^i + Q_{t+1} - Q_t
                delta_i = rewards[i] - self.mu[i] + Q_next - Q_curr

                # Update average reward: μ^i ← (1 - β_μ)μ^i + β_μ r'^i
                self.mu[i] = (1 - beta_mu) * self.mu[i] + beta_mu * rewards[i]

                # Gradient of Q w.r.t. ω: ∇_ω Q = φ(s, a)
                phi = self.get_q_features(state, joint_actions)

                # Update: ω̄^i ← ω^i + β_ω · δ^i · φ(s, a)
                self.omega_bar[i] = self.omega[i] + beta_omega * delta_i * phi

            # Actor step: Update policy parameters
            for i in range(self.N):
                # Compute current Q-value for eligibility trace
                Q_curr = self.compute_Q_value(state, joint_actions, i)

                # Compute policy gradient
                psi_i = self.compute_policy_gradient(state, joint_actions[i], i)

                # Update: θ^i ← θ^i + β_θ · Q(s, a; ω^i) · ψ^i
                self.theta[i] = self.theta[i] + beta_theta * Q_curr * psi_i

                # Clip theta to prevent explosion
                self.theta[i] = np.clip(self.theta[i], -10, 10)

            # Consensus step: ω^i ← Σ_j C(i,j) · ω̄^j
            self.omega = self.C @ self.omega_bar

            # Update for next iteration
            state = next_state
            joint_actions = next_joint_actions

            # Track average return
            avg_return = np.mean(rewards)
            returns_history.append(avg_return)

        return returns_history


def plot_results(returns_history, window=100):
    """Plot the learning curve"""
    plt.figure(figsize=(10, 6))

    # Smooth the curve
    smoothed = np.convolve(returns_history, np.ones(window) / window, mode="valid")

    plt.plot(smoothed, linewidth=2, label="Algorithm 1 (Decentralized)")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Globally Averaged Return", fontsize=12)
    plt.title("Convergence of Networked Actor-Critic Algorithm", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("convergence_corrected.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Final average return: {np.mean(returns_history[-100:]):.4f}")


def main():
    # Create environment with same parameters as the paper
    print("Creating environment...")
    env = Environment(
        n_states=20, n_actions=2, n_nodes=20, n_phi=10, n_varphi=5, seed=42
    )

    print(f"Environment created:")
    print(f"  States: {env.n_states}")
    print(f"  Actions per agent: {env.n_actions}")
    print(f"  Agents: {env.n_nodes}")
    print(f"  Q-features (phi): {env.n_phi}")
    print(f"  Policy features (varphi): {env.n_varphi}")
    print(f"  Total joint actions: {env.n_action_space}")

    # Create algorithm
    print("\nInitializing algorithm...")
    algo = NetworkedActorCritic(env, gamma=0.95, seed=42)

    # Run training
    print("\nRunning training...")
    n_steps = 10000
    returns = algo.run_episode(n_steps=n_steps)

    # Plot results
    print("\nPlotting results...")
    plot_results(returns, window=100)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
