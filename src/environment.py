"""Supply chain simulation environment for RL inventory optimization."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SupplyChainEnv(gym.Env):
    """Multi-echelon supply chain environment.

    State: [inventory_level, demand_forecast, lead_time, delay_probability]
    Actions: discrete order quantities (0, 10, 20, ..., 100)
    Reward: negative of (holding_cost + shortage_cost)
    """

    def __init__(self, holding_cost=1.0, shortage_cost=5.0, max_steps=100):
        super().__init__()
        self.holding_cost = holding_cost
        self.shortage_cost = shortage_cost
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(11)  # 0 to 100 in steps of 10
        self.observation_space = spaces.Box(low=0, high=500, shape=(4,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.inventory = 50.0
        self.step_count = 0
        self.demand_mean = np.random.uniform(20, 40)
        return self._get_state(), {}

    def _get_state(self):
        demand_forecast = self.demand_mean + np.random.normal(0, 5)
        lead_time = np.random.uniform(1, 5)
        delay_prob = np.random.uniform(0, 0.3)
        return np.array([self.inventory, demand_forecast, lead_time, delay_prob], dtype=np.float32)

    def step(self, action):
        order_qty = action * 10

        # Simulate demand
        demand = max(0, np.random.normal(self.demand_mean, 8))

        # Simulate delivery with possible delay
        delay_prob = np.random.uniform(0, 0.3)
        delivered = order_qty if np.random.random() > delay_prob else order_qty * 0.5

        # Update inventory
        self.inventory += delivered - demand

        # Calculate costs
        holding = max(0, self.inventory) * self.holding_cost
        shortage = max(0, -self.inventory) * self.shortage_cost
        reward = -(holding + shortage)

        self.inventory = max(0, self.inventory)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_state(), reward, done, False, {}
