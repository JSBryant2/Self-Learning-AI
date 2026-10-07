"""PPO training and truthful telemetry from the actual actor network."""
import json
import os
import tempfile
import queue
import time
from functools import partial
from pathlib import Path
import numpy as np
import torch
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'origin-matplotlib'))
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .environment import BalanceEnv


def make_env(task):
    return BalanceEnv(task=task)


def actor_layers(model):
    return [layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear)] + [model.policy.action_net]


def brain_snapshot(model, observation, initial_weights):
    layers = actor_layers(model)
    weights = [layer.weight.detach().cpu().numpy() for layer in layers]
    with torch.no_grad():
        value = torch.as_tensor(np.asarray(observation, dtype=np.float32)[None], device=model.device)
        activations = [value[0].cpu().tolist()]
        for layer in model.policy.mlp_extractor.policy_net:
            value = layer(value)
            if isinstance(layer, torch.nn.Tanh):
                activations.append(value[0].cpu().tolist())
        mean = model.policy.action_net(value)[0]
        activations.append(mean.cpu().tolist())
        sensitivity_input = torch.as_tensor(np.asarray(observation, dtype=np.float32)[None], device=model.device).requires_grad_(True)
    # Local gradient describes sensitivity at THIS state, not general causal importance.
    with torch.enable_grad():
        output_mean = model.policy.action_net(model.policy.mlp_extractor.policy_net(sensitivity_input))
        sensitivity = []
        for index in range(12):
            grad = torch.autograd.grad(output_mean[0, index], sensitivity_input, retain_graph=index < 11)[0]
            sensitivity.append(grad.detach().abs()[0].cpu().numpy())
    return dict(sizes=[55, *[layer.out_features for layer in layers]],
                weights=[weight.tolist() for weight in weights],
                changes=[(weight-initial).tolist() for weight, initial in zip(weights, initial_weights)],
                activations=activations,
                meanActions=mean.cpu().tolist(),
                explorationStd=model.policy.log_std.detach().exp().cpu().tolist(),
                inputSensitivity=np.mean(sensitivity, axis=0).tolist(),
                inputNames=BalanceEnv.observation_names,
                weightChange=float(np.sqrt(sum(np.sum((weight-initial)**2) for weight, initial in zip(weights, initial_weights)))))


def evaluate(model, task, seeds=(10001, 10002), seconds=5):
    env = BalanceEnv(task=task, max_seconds=seconds)
    results = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            while True:
                action = np.zeros(12, np.float32) if model is None else model.predict(observation, deterministic=True)[0]
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    results.append(info['episode'])
                    break
    finally:
        env.close()
    return dict(reward=float(np.mean([row['reward'] for row in results])),
                upright_seconds=float(np.mean([row['upright_seconds'] for row in results])),
                forward_distance=float(np.mean([row['forward_distance'] for row in results])), seconds=seconds, seeds=list(seeds))


def run_training(config, directory, checkpoint, output, paused, stop, fast):
    torch.set_num_threads(1)
    directory = Path(directory)
    (directory/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    factories = [partial(make_env, config['task']) for _ in range(config['worlds'])]
    envs = DummyVecEnv(factories) if config['worlds'] == 1 else SubprocVecEnv(factories, start_method='spawn')
    envs.seed(config['seed'])
    try:
        model = PPO.load(checkpoint, env=envs, device='cpu') if checkpoint else PPO(
            'MlpPolicy', envs, device='cpu', seed=config['seed'], n_steps=128, batch_size=64, n_epochs=4,
            learning_rate=3e-4, ent_coef=.005,
            policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]), log_std_init=-.7), verbose=0)
        initial = [layer.weight.detach().cpu().numpy().copy() for layer in actor_layers(model)]
        model.save(directory/'initial.zip')
        baseline = evaluate(None, config['task'])
        evaluations = [dict(steps=0, **evaluate(model, config['task']))]
        history = []
        checkpoint_path = directory/'policy.zip'

        class LiveCallback(BaseCallback):
            last_publish = 0
            run_start_steps = model.num_timesteps
            last_evaluation = 0
            latest_scene = None

            def publish(self, status='running', force=False):
                now = time.monotonic()
                if not force and now-self.last_publish < .15:
                    return
                self.last_publish = now
                world = envs.env_method('training_snapshot', indices=0)[0]
                self.latest_scene = world['scene']
                steps = self.num_timesteps-self.run_start_steps
                metrics = dict(steps=steps, total_steps=self.num_timesteps, updates=model._n_updates,
                               episodes=len(history), current_reward=world['episode_reward'],
                               current_upright_seconds=world['upright_seconds'], worlds=config['worlds'],
                               progress=min(1, steps/config['steps']), speed='fast' if fast.value else 'realtime',
                               losses={key: float(value) for key, value in model.logger.name_to_value.items()
                                       if key in ('train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss')})
                payload = dict(status=status, run_id=directory.name, config=config, scene=world['scene'], metrics=metrics,
                               history=history[-200:], evaluations=evaluations[-100:], baseline=baseline,
                               brain=brain_snapshot(model, world['observation'], initial),
                               checkpoint=str(checkpoint_path) if checkpoint_path.is_file() else None)
                # Drop stale telemetry, never let a slow viewer stall physics.
                try:
                    output.put_nowait(payload)
                except queue.Full:
                    try:
                        output.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        output.put(payload, timeout=.1)
                    except queue.Full:
                        pass

            def _on_step(self):
                began = time.monotonic()
                for world, info in enumerate(self.locals['infos']):
                    if 'episode' in info:
                        record = dict(index=len(history)+1, world=world,
                                      total_steps=self.num_timesteps, **info['episode'])
                        history.append(record)
                        with (directory/'episodes.jsonl').open('a') as log:
                            log.write(json.dumps(record, allow_nan=False)+'\n')
                while paused.is_set() and not stop.is_set():
                    self.publish('paused')
                    time.sleep(.05)
                if stop.is_set():
                    return False
                self.publish()
                if not fast.value:
                    time.sleep(max(0, 1/30-(time.monotonic()-began)))
                return True

            def _on_rollout_start(self):
                if self.num_timesteps-self.run_start_steps-self.last_evaluation >= 1024:
                    self.record_evaluation()

            def record_evaluation(self):
                self.publish('evaluating', force=True)
                model.save(checkpoint_path)
                steps = self.num_timesteps-self.run_start_steps
                result = dict(steps=steps, **evaluate(model, config['task']))
                evaluations.append(result)
                self.last_evaluation = steps
                with (directory/'evaluation.jsonl').open('a') as log:
                    log.write(json.dumps(result, allow_nan=False)+'\n')
                self.publish(force=True)

            def _on_training_end(self):
                self.record_evaluation()
                self.publish('stopped' if stop.is_set() else 'finished', force=True)

        callback = LiveCallback()
        model.learn(total_timesteps=config['steps'], callback=callback, reset_num_timesteps=not bool(checkpoint))
    finally:
        envs.close()
