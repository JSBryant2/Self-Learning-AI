"""Gymnasium balance task. Actions always bypass demonstration controllers."""
import math
import gymnasium as gym
import numpy as np
import pybullet as p
from .simulation import Simulation


class BalanceEnv(gym.Env):
    metadata = {'render_modes': ['state'], 'render_fps': 30}
    observation_names = (
        [f'body_up_{axis}' for axis in 'xyz'] +
        [f'body_velocity_{axis}' for axis in 'xyz'] +
        [f'body_angular_velocity_{axis}' for axis in 'xyz'] + ['height'] +
        [f'joint_angle_{i}' for i in range(12)] +
        [f'joint_velocity_{i}' for i in range(12)] +
        [f'foot_contact_{leg}' for leg in Simulation.leg_names] +
        [f'foot_force_{leg}' for leg in Simulation.leg_names] +
        [f'previous_action_{i}' for i in range(12)] + ['time_remaining'])

    def __init__(self, max_seconds=20.0, action_repeat=2, perturbation=.08, render_mode=None):
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('max_seconds must be positive and finite')
        if not isinstance(action_repeat, int) or action_repeat < 1:
            raise ValueError('action_repeat must be a positive integer')
        if not math.isfinite(perturbation) or not 0 <= perturbation <= .2:
            raise ValueError('perturbation must be in [0, .2]')
        if render_mode not in (None, 'state'):
            raise ValueError('render_mode must be None or state')
        self.render_mode = render_mode
        self.max_seconds, self.action_repeat, self.perturbation = max_seconds, action_repeat, perturbation
        self.action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (55,), dtype=np.float32)
        self.sim = Simulation()
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('No reset options are currently supported')
        self.world_seed = int(self.np_random.integers(1, 1000000))
        self.sim.reset(self.world_seed)
        amplitude = self.perturbation
        tilt = self.np_random.uniform(-amplitude, amplitude, 3)
        p.resetBasePositionAndOrientation(self.sim.torso, [0, 1, 0], p.getQuaternionFromEuler(tilt),
                                         physicsClientId=self.sim.client)
        for index, neutral, (low, high) in zip(self.sim.joint_ids, self.sim.neutral, self.sim.limits):
            angle = float(np.clip(neutral+self.np_random.uniform(-amplitude, amplitude), low, high))
            p.resetJointState(self.sim.torso, index, angle, physicsClientId=self.sim.client)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.total_reward = self.upright_seconds = self.motor_work = 0.0
        self.steps = 0
        self.start_position = np.array(self.sim.observe()['position'])
        self.done = False
        return self._vector(self.sim.observe()), {'world_seed': self.world_seed}

    def _vector(self, obs):
        rotation = np.array(p.getMatrixFromQuaternion(obs['orientation'])).reshape(3, 3)
        angle = [2*(a-low)/(high-low)-1 for a, (low, high) in zip(obs['jointAngles'], self.sim.limits)]
        remaining = 0.0 if self.sim.time >= self.max_seconds-1e-9 else 1-self.sim.time/self.max_seconds
        return np.asarray([*(rotation.T @ [0, 1, 0]),
                           *(rotation.T @ obs['linearVelocity']/3),
                           *(rotation.T @ obs['angularVelocity']/4), obs['height'],
                           *angle, *(np.array(obs['jointVelocities'])/4),
                           *obs['footContacts'], *(np.array(obs['footForces'])/100),
                           *self.previous_action, remaining], dtype=np.float32)

    def _reason(self, obs):
        if obs['height'] < .45 or obs['upright'] < .45:
            return 'fall'
        if max(abs(obs['position'][0]), abs(obs['position'][2])) > 10:
            return 'out_of_bounds'
        return None

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset before stepping a new episode')
        raw = np.asarray(action)
        if raw.shape != (12,) or raw.dtype.kind not in 'fiu' or not np.all(np.isfinite(raw)):
            raise ValueError('Action must contain 12 finite numeric values')
        action = np.clip(raw, -1, 1).astype(np.float32)
        smoothness = float(np.mean((action-self.previous_action)**2))
        components = dict(upright=0.0, height=0.0, effort=0.0, smoothness=0.0, fall=0.0)
        reason = None
        for _ in range(self.action_repeat):
            obs = self.sim.step(actions=action.tolist())
            dt = self.sim.dt
            power = sum(abs(t*v) for t, v in zip(obs['jointTorques'], obs['jointVelocities']))
            components['upright'] += .75*max(0, obs['upright'])*dt
            components['height'] += .25*max(0, 1-abs(obs['height']-.98)/.53)*dt
            components['effort'] -= .02*(power/48)*dt
            components['smoothness'] -= .02*smoothness*dt
            self.motor_work += power*dt
            if obs['upright'] > .85 and obs['height'] > .65:
                self.upright_seconds += dt
            reason = self._reason(obs)
            if reason or self.sim.time >= self.max_seconds-1e-9:
                break
        terminated = reason is not None
        truncated = not terminated and self.sim.time >= self.max_seconds-1e-9
        if reason == 'fall':
            components['fall'] = -1.0
        reward = sum(components.values())
        self.total_reward += reward
        self.steps += 1
        self.previous_action = action.copy()
        self.done = terminated or truncated
        info = {'reward_components': components, 'reason': reason or ('time_limit' if truncated else None)}
        if self.done:
            displacement = np.array(obs['position'])-self.start_position
            info['episode'] = dict(reward=self.total_reward, steps=self.steps, seconds=self.sim.time,
                                   upright_seconds=self.upright_seconds, distance=float(np.linalg.norm(displacement[[0, 2]])),
                                   forward_distance=float(displacement[2]), motor_work_joules=self.motor_work,
                                   reason=info['reason'], world_seed=self.world_seed)
        return self._vector(obs), float(reward), terminated, truncated, info

    def render(self):
        return self.sim.snapshot() if self.render_mode == 'state' else None

    def close(self):
        self.sim.close()
