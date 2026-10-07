"""Read-only deterministic policy playback in a fresh environment."""
import json
import math
from pathlib import Path
import numpy as np
from .environment import BalanceEnv


def saved_policies(root=Path('runs')):
    root = Path(root).resolve()
    entries = []
    if not root.is_dir():
        return entries
    for directory in root.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            config = json.loads((directory/'config.json').read_text())
            if config['task'] not in ('balance', 'walk'):
                continue
            for filename, label in [('initial.zip', 'Initial policy'), ('policy.zip', 'Latest saved policy')]:
                path = directory/filename
                if path.is_file() and not path.is_symlink():
                    entries.append(dict(id=f'{directory.name}/{filename}', run_id=directory.name,
                                        label=label, task=config['task'], training_steps=config.get('steps'),
                                        modified=path.stat().st_mtime))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(entries, key=lambda entry: (-entry['modified'], entry['id']))


def validate_load(message, root=Path('runs')):
    if not isinstance(message, dict):
        raise ValueError('Playback command must be an object')
    entries = {entry['id']: entry for entry in saved_policies(root)}
    identifier = message.get('policy')
    if not isinstance(identifier, str) or identifier not in entries:
        raise ValueError('Select an available saved policy')
    seed, seconds = message.get('seed', 10001), message.get('seconds', 20)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 999999:
        raise ValueError('Seed must be an integer from 0 to 999999')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 1 <= seconds <= 120:
        raise ValueError('Episode duration must be from 1 to 120 seconds')
    return entries[identifier], seed, float(seconds)


class PlaybackSession:
    def __init__(self, root=Path('runs')):
        self.root = Path(root)
        self.env = self.model = None
        self.status = 'idle'
        self.results = []
        self.current = None
        self.revision = 0

    def load(self, message):
        entry, seed, seconds = validate_load(message, self.root)
        from .learner import PPO, torch
        torch.set_num_threads(1)
        model = PPO.load(self.root/entry['id'], device='cpu')
        env = BalanceEnv(task=entry['task'], max_seconds=seconds)
        try:
            observation, _ = env.reset(seed=seed)
        except Exception:
            env.close()
            raise
        if self.env is not None:
            self.env.close()
        self.env, self.model, self.observation = env, model, observation
        self.entry, self.seed, self.seconds = entry, seed, seconds
        self.current = None
        self.status = 'playing'
        self.revision += 1

    def command(self, message):
        if not isinstance(message, dict):
            raise ValueError('Playback command must be an object')
        kind = message.get('type')
        if kind == 'load':
            self.load(message)
        elif kind == 'pause' and self.env is not None:
            self.status = 'paused' if not self.env.done else 'finished'
        elif kind == 'resume' and self.env is not None and not self.env.done:
            self.status = 'playing'
        elif kind == 'restart' and self.env is not None:
            self.observation, _ = self.env.reset(seed=self.seed)
            self.current = None
            self.status = 'playing'
            self.revision += 1
        else:
            raise ValueError('Load a policy first, or restart a finished attempt')

    def step(self):
        if self.status == 'playing':
            action = self.model.predict(self.observation, deterministic=True)[0]
            self.observation, _, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                self.status = 'finished'
                self.current = dict(policy=self.entry['id'], label=self.entry['label'], task=self.entry['task'],
                                    seed=self.seed, duration=self.seconds, **info['episode'])
                self.results.append(self.current)
                self.results = self.results[-20:]
        return self.snapshot()

    def snapshot(self):
        result = dict(status=self.status, revision=self.revision, results=self.results)
        if self.env is not None:
            result.update(policy=self.entry, seed=self.seed, seconds=self.seconds,
                          scene=self.env.sim.snapshot(), metrics=dict(reward=self.env.total_reward,
                          upright_seconds=self.env.upright_seconds, motor_work_joules=self.env.motor_work),
                          result=self.current, saved_steps=int(self.model.num_timesteps))
        return result

    def close(self):
        if self.env is not None:
            self.env.close()
