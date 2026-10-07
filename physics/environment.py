"""Gymnasium balance task. Actions always bypass demonstration controllers."""
import math
import gymnasium as gym
import numpy as np
import pybullet as p
from .simulation import Simulation
from .walking import STAGES


class BalanceEnv(gym.Env):
    metadata = {'render_modes': ['state'], 'render_fps': 30}
    observation_names = (
        [f'body_up_{axis}' for axis in 'xyz'] +
        [f'body_velocity_{axis}' for axis in 'xyz'] +
        [f'body_angular_velocity_{axis}' for axis in 'xyz'] + ['height'] +
        [f'{leg}_{joint}_angle' for leg in Simulation.leg_names for joint in ('hip_roll', 'hip_pitch', 'knee')] +
        [f'{leg}_{joint}_velocity' for leg in Simulation.leg_names for joint in ('hip_roll', 'hip_pitch', 'knee')] +
        [f'foot_contact_{leg}' for leg in Simulation.leg_names] +
        [f'foot_force_{leg}' for leg in Simulation.leg_names] +
        [f'{leg}_{joint}_previous_action' for leg in Simulation.leg_names for joint in ('hip_roll', 'hip_pitch', 'knee')] + ['time_remaining'])

    def __init__(self, max_seconds=20.0, action_repeat=2, perturbation=.08, render_mode=None, task="balance", walking_stage=None):
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('max_seconds must be positive and finite')
        if not isinstance(action_repeat, int) or action_repeat < 1:
            raise ValueError('action_repeat must be a positive integer')
        if not math.isfinite(perturbation) or not 0 <= perturbation <= .2:
            raise ValueError('perturbation must be in [0, .2]')
        if render_mode not in (None, 'state'):
            raise ValueError('render_mode must be None or state')
        if task not in ('balance', 'walk'):
            raise ValueError('task must be balance or walk')
        if walking_stage is not None and (task != 'walk' or isinstance(walking_stage, bool)
                                          or not isinstance(walking_stage, int) or not 0 <= walking_stage < len(STAGES)):
            raise ValueError('walking_stage requires walk task and a stage from 0 to 3')
        self.walking_stage = self.pending_stage = walking_stage
        self.observation_names = list(type(self).observation_names)
        if walking_stage is not None:
            self.observation_names.extend(['target_forward_speed','body_target_direction_x','body_target_direction_y','body_target_direction_z'])
        self.task = task
        self.render_mode = render_mode
        self.max_seconds, self.action_repeat, self.perturbation = max_seconds, action_repeat, perturbation
        self.action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len(self.observation_names),), dtype=np.float32)
        self.sim = Simulation()
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('No reset options are currently supported')
        self.world_seed = int(self.np_random.integers(1, 1000000))
        self.sim.reset(self.world_seed)
        self.walking_stage = self.pending_stage
        amplitude = STAGES[self.walking_stage]['perturbation'] if self.walking_stage is not None else self.perturbation
        if self.walking_stage is not None:
            # Leave a clear straight corridor for early locomotion; no obstacle curriculum yet.
            for i, body in enumerate(item for item in self.sim.bodies if item['name'] == 'object'):
                p.resetBasePositionAndOrientation(body['id'], [(-1 if i%2 else 1)*6, .5, i-3],
                                                 [0,0,0,1], physicsClientId=self.sim.client)
        tilt = self.np_random.uniform(-amplitude, amplitude, 3)
        p.resetBasePositionAndOrientation(self.sim.torso, [0, 1, 0], p.getQuaternionFromEuler(tilt),
                                         physicsClientId=self.sim.client)
        for index, neutral, (low, high) in zip(self.sim.joint_ids, self.sim.neutral, self.sim.limits):
            angle = float(np.clip(neutral+self.np_random.uniform(-amplitude, amplitude), low, high))
            p.resetJointState(self.sim.torso, index, angle, physicsClientId=self.sim.client)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.total_reward = self.upright_seconds = self.motor_work = 0.0
        self.slip_distance = self.body_contact_seconds = self.speed_error_integral = 0.0
        self.foot_cycles = [0]*4
        self.foot_was_grounded = [False]*4
        self.swing_time = [0.0]*4
        self.swing_peak = [0.0]*4
        self.steps = 0
        self.start_position = np.array(self.sim.observe()['position'])
        self.done = False
        return self._vector(self.sim.observe()), {'world_seed': self.world_seed}

    def _vector(self, obs):
        rotation = np.array(p.getMatrixFromQuaternion(obs['orientation'])).reshape(3, 3)
        angle = [2*(a-low)/(high-low)-1 for a, (low, high) in zip(obs['jointAngles'], self.sim.limits)]
        remaining = 0.0 if self.sim.time >= self.max_seconds-1e-9 else 1-self.sim.time/self.max_seconds
        values = [*(rotation.T @ [0, 1, 0]),
                           *(rotation.T @ obs['linearVelocity']/3),
                           *(rotation.T @ obs['angularVelocity']/4), obs['height'],
                           *angle, *(np.array(obs['jointVelocities'])/4),
                           *obs['footContacts'], *(np.array(obs['footForces'])/100),
                           *self.previous_action, remaining]
        if self.walking_stage is not None:
            values.extend([STAGES[self.walking_stage]['target_speed'], *(rotation.T @ [0,0,1])])
        return np.asarray(values, dtype=np.float32)

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
        components = dict(upright=0.0, height=0.0, effort=0.0, smoothness=0.0, fall=0.0, forward=0.0, tracking=0.0, slip=0.0, dragging=0.0, lateral=0.0, footsteps=0.0)
        reason = None
        for _ in range(self.action_repeat):
            obs = self.sim.step(actions=action.tolist())
            dt = self.sim.dt
            power = sum(abs(t*v) for t, v in zip(obs['jointTorques'], obs['jointVelocities']))
            components['upright'] += .75*max(0, obs['upright'])*dt
            components['height'] += .25*max(0, 1-abs(obs['height']-.98)/.53)*dt
            components['effort'] -= .02*(power/48)*dt
            components['smoothness'] -= .02*smoothness*dt
            contacts = np.asarray(obs['footContacts'])
            foot_speed = np.linalg.norm(np.asarray(obs['footVelocities'])[:, [0,2]], axis=1)
            slip_speed = float(np.mean(foot_speed[contacts])) if contacts.any() else 0.0
            self.slip_distance += slip_speed*dt
            self.body_contact_seconds += float(obs['nonFootContact'])*dt
            touches = 0
            for i, contact in enumerate(contacts):
                if contact:
                    if self.foot_was_grounded[i] and self.swing_time[i] >= .05 and self.swing_peak[i] >= .09:
                        self.foot_cycles[i] += 1
                        touches += 1
                    self.foot_was_grounded[i] = True
                    self.swing_time[i] = self.swing_peak[i] = 0.0
                elif self.foot_was_grounded[i]:
                    self.swing_time[i] += dt
                    self.swing_peak[i] = max(self.swing_peak[i], obs['footPositions'][i][1])
            if self.walking_stage is not None:
                target = STAGES[self.walking_stage]['target_speed']
                components.update({key: components[key]+value for key,value in
                                   self.walking_terms(obs, target, slip_speed, touches, dt).items()})
                self.speed_error_integral += abs(obs['linearVelocity'][2]-target)*dt
            elif self.task == 'walk':
                components['forward'] += .5*float(np.clip(obs['linearVelocity'][2], -2, 2))*max(0, obs['upright'])*dt
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
            info['episode'] = dict(r=self.total_reward, l=self.steps, t=self.sim.time,
                                   reward=self.total_reward, steps=self.steps, seconds=self.sim.time,
                                   upright_seconds=self.upright_seconds, distance=float(np.linalg.norm(displacement[[0, 2]])),
                                   forward_distance=float(displacement[2]), motor_work_joules=self.motor_work,
                                   reason=info['reason'], world_seed=self.world_seed,
                                   stage=self.walking_stage,
                                   target_speed=STAGES[self.walking_stage]['target_speed'] if self.walking_stage is not None else None,
                                   upright_ratio=self.upright_seconds/self.sim.time,
                                   forward_speed=float(displacement[2])/self.sim.time,
                                   mean_slip_speed=self.slip_distance/self.sim.time,
                                   body_contact_ratio=self.body_contact_seconds/self.sim.time,
                                   speed_error=self.speed_error_integral/self.sim.time,
                                   foot_cycles=sum(self.foot_cycles), moving_legs=sum(c > 0 for c in self.foot_cycles),
                                   foot_cycles_by_leg=self.foot_cycles.copy())
        return self._vector(obs), float(reward), terminated, truncated, info

    @staticmethod
    def walking_terms(obs, target, slip_speed, touches, dt):
        posture = float(np.clip((obs['upright']-.65)/.35, 0, 1)*np.clip((obs['height']-.6)/.35, 0, 1))
        posture *= float(not obs['nonFootContact'] and any(obs.get('footContacts',[True])))
        posture *= float(np.exp(-3*(slip_speed/max(.05,target*.5))**2))
        velocity = obs['linearVelocity'][2]
        tracking = np.exp(-4*((velocity-target)/max(.1,target))**2)-np.exp(-4)
        return dict(forward=2*float(np.clip(velocity/target, -1, 1))*posture*dt,
                    tracking=.75*float(tracking)*posture*dt,
                    slip=-.4*min(2,slip_speed)*dt,
                    dragging=-.75*float(obs['nonFootContact'])*dt,
                    lateral=-.25*abs(obs['linearVelocity'][0])*dt,
                    footsteps=.02*touches*posture if velocity > .02 and slip_speed < .12 else 0.0)

    def set_walking_stage(self, stage):
        if self.walking_stage is None or isinstance(stage, bool) or not isinstance(stage,int) or not 0 <= stage < len(STAGES):
            raise ValueError('Invalid walking stage')
        self.pending_stage = stage  # Apply on reset, never change a goal mid-episode.

    def training_snapshot(self):
        return dict(scene=self.sim.snapshot(), observation=self._vector(self.sim.observe()).tolist(),
                    episode_reward=self.total_reward, episode_steps=self.steps,
                    upright_seconds=self.upright_seconds, stage=self.walking_stage,
                    target_speed=STAGES[self.walking_stage]['target_speed'] if self.walking_stage is not None else None,
                    foot_cycles=sum(self.foot_cycles), mean_slip_speed=self.slip_distance/max(self.sim.time, self.sim.dt))

    def render(self):
        return self.sim.snapshot() if self.render_mode == 'state' else None

    def close(self):
        self.sim.close()
