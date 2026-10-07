import unittest
import numpy as np
import pybullet as p
from gymnasium.utils.env_checker import check_env
from physics.environment import BalanceEnv


class EnvironmentTests(unittest.TestCase):
    def make(self, **kwargs):
        env = BalanceEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_gymnasium_contract(self):
        check_env(self.make(), skip_render_check=True)

    def test_seed_and_action_observations(self):
        env = self.make()
        a, _ = env.reset(seed=42)
        b, _ = env.reset(seed=42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (55,))
        c, _ = env.reset(seed=43)
        self.assertFalse(np.array_equal(a, c))
        obs, reward, _, _, info = env.step([2, -2, .25]*4)
        self.assertEqual(env.sim.actions, [1, -1, .25]*4)
        np.testing.assert_array_equal(obs[-13:-1], [1, -1, .25]*4)
        self.assertAlmostEqual(sum(info['reward_components'].values()), reward)

    def test_time_limit_and_reset(self):
        env = self.make(max_seconds=.1, perturbation=0)
        env.reset(seed=1)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(np.zeros(12))
            done = terminated or truncated
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info['reason'], 'time_limit')
        self.assertAlmostEqual(info['episode']['seconds'], .1)
        self.assertEqual(obs[-1], 0)
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(12))
        env.reset(seed=1)
        self.assertEqual(env.sim.time, 0)
        self.assertEqual(env.total_reward, 0)

    def test_fall_and_boundary(self):
        env = self.make()
        for position, reason in [([0, .2, 0], 'fall'), ([11, 1, 0], 'out_of_bounds')]:
            env.reset(seed=1)
            p.resetBasePositionAndOrientation(env.sim.torso, position, [0, 0, 0, 1], physicsClientId=env.sim.client)
            _, _, terminated, truncated, info = env.step(np.zeros(12))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info['reason'], reason)

    def test_balance_baseline_and_invalid_actions(self):
        env = self.make(max_seconds=2, perturbation=0)
        env.reset(seed=1)
        for action in [[0]*4, [float('nan')]*12, [True]*12]:
            with self.assertRaises(ValueError):
                env.step(action)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(np.zeros(12))
            done = terminated or truncated
        self.assertEqual(info['reason'], 'time_limit')
        self.assertGreater(info['episode']['upright_seconds'], 1.8)
        self.assertGreater(info['episode']['reward'], 1.5)

    def test_walk_task_rewards_forward_motion_without_scripted_actions(self):
        balance, walk = self.make(perturbation=0), self.make(perturbation=0, task='walk')
        for env in (balance, walk):
            env.reset(seed=1)
            p.resetBaseVelocity(env.sim.torso, [0,0,1], [0,0,0], physicsClientId=env.sim.client)
        _, balance_reward, _, _, balance_info = balance.step(np.zeros(12))
        _, walk_reward, _, _, walk_info = walk.step(np.zeros(12))
        self.assertEqual(balance_info['reward_components']['forward'], 0)
        self.assertGreater(walk_info['reward_components']['forward'], 0)
        self.assertGreater(walk_reward, balance_reward)
        self.assertEqual(walk.sim.actions, [0]*12)
