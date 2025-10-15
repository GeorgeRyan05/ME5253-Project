from collections import defaultdict
from functools import cached_property
from mpe2 import simple_spread_v3
from mpe2._mpe_utils.core import Landmark, World, Entity, Agent
import numpy as np

from environment import Environment


class CooperativeNavigationEnvironment(Environment):
    """
    Cooperative Navigation task (based on Lowe et al. (2017),
    using the Multi Particle Environments package for
    unspecified hyperparameters).
    """

    agent_size = 0.15
    landmark_size = 0.050
    n_states = None
    n_phi = None
    n_varphi = None
    state = None
    # contradictory implementation to linear, due to continuous state space.
    # Leaving None to prevent confusion.
    # world size is [-1, 1] in both dimensions.

    def __init__(
        self,
        n_nodes=20,
        seed=0,
    ):
        self.n_actions = 5
        rng = np.random.default_rng(seed)
        self.n_nodes = n_nodes
        # self.n_phi = n_phi
        # self.n_varphi = n_varphi
        self.n_edges = int(n_nodes * (n_nodes - 1))
        self.n_action_space = self.n_actions**n_nodes
        self.seed = seed
        self._env = simple_spread_v3.env(render_mode="rgb_array", N=n_nodes)
        self._env.reset(seed=seed)
        self.log = defaultdict(list)
        self.reset()
        self._world: World = self._env.world
        self.landmarks: list[Landmark] = self._world.landmarks
        self.agents: list[Agent] = self._world.agents
        self._agent_name_to_index = {ag.name: i for i, ag in enumerate(self.agents)}
        self.targets = rng.permutation(n_nodes).tolist()

    def reset(self):
        self._env.reset(seed=self.seed)
        self.n_step = 0
        for key in ("actions", "steps", "state", "reward"):
            if key in self.log:
                del self.log[key]

    def get_features(self, state, actions):
        return self.get_action_vec(state, actions), self.get_varphi(state)

    def get_action_vec(self, state, actions):
        # state_vec = self.agent_positions.flatten()
        actions_vec = np.zeros((self.n_nodes, self.n_actions))
        for i, a in enumerate(actions):
            actions_vec[i, a] = 1.0
        actions_vec = actions_vec
        # print(f"{state_vec = }, {actions_vec = }")
        return actions_vec

    def get_varphi(self, state):
        """
        Here, returns the state vector (agent positions)

        Parameters
        ----------
        state : int
            Irrelevant, will be tracked

        Returns
        -------
        np.ndarray
            State vector (agent positions)
        """
        return self.agent_positions.flatten()
    @cached_property
    def landmark_positions(self):
        return np.array([lm.state.p_pos for lm in self.landmarks])

    @property
    def agent_positions(self):
        return np.array([ag.state.p_pos for ag in self.agents])

    def reward(self, i):
        "Reward at a particular time-step."
        agent = self.agents[i]
        # Distance to target landmark
        target = self.landmark_positions[self.targets[i]]
        dist2 = np.sum(np.square(agent.state.p_pos - target))
        rew = -dist2 - int(agent.collide)
        return rew

    def get_rewards(self, actions):
        # self.next_step(actions)
        rewards = np.array([self.reward(i) for i in range(self.n_nodes)])
        print(f"{rewards = }")
        return rewards

    def next_step(self, actions):
        """Takes a step in the environment.

        Parameters:
        -----------
        * actions: list<int<n_actions>, n_nodes>
            List of actions for each agent.

        Returns:
        --------
        * observations: list<np.array, n_nodes>
            List of observations for each agent.
        * rewards: np.array<float, n_nodes>
            Array of rewards for each agent.
        * done: bool
            Whether the episode has ended.
        * info: dict
            Additional information.
        """
        for agent in self.agents:
            if agent.collide:
                print(f"Collision detected for agent {agent.name}")
            action_space = self._env.action_space(agent.name)
            mask = np.zeros(self.n_actions, dtype=np.int8)
            mask[actions[self._agent_name_to_index[agent.name]]] = True
            action = action_space.sample(mask)
            self._env.step(action)
            obs, _, termination, truncation, info = self._env.last()
            print(f"{termination = }, {truncation = }")
            rewards = np.array([self.reward(i) for i in range(self.n_nodes)])
        self.n_step += 1
        print("Reached the end")
        return None
        return obs, rewards, termination, truncation, info


if __name__ == "__main__":
    env = CooperativeNavigationEnvironment(n_nodes=10)
    # print(env.landmark_positions, env.agents, env.targets)
    print(env.next_step([4] * 10))
