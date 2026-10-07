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
from .walking import STAGES, WalkingCurriculum


def make_env(task, walking_stage=None):
    return BalanceEnv(task=task, walking_stage=walking_stage)


def actor_layers(model):
    return [layer for layer in model.policy.mlp_extractor.policy_net if isinstance(layer, torch.nn.Linear)] + [model.policy.action_net]


def brain_snapshot(model, observation, initial_weights, input_names=None):
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
    return dict(sizes=[model.observation_space.shape[0], *[layer.out_features for layer in layers]],
                weights=[weight.tolist() for weight in weights],
                changes=[(weight-initial).tolist() for weight, initial in zip(weights, initial_weights)],
                activations=activations,
                meanActions=mean.cpu().tolist(),
                explorationStd=model.policy.log_std.detach().exp().cpu().tolist(),
                inputSensitivity=np.mean(sensitivity, axis=0).tolist(),
                inputNames=input_names or (BalanceEnv.observation_names+(['target_forward_speed','body_target_direction_x','body_target_direction_y','body_target_direction_z'] if model.observation_space.shape[0] == 59 else [])),
                weightChange=float(np.sqrt(sum(np.sum((weight-initial)**2) for weight, initial in zip(weights, initial_weights)))))


def evaluate(model, task, seeds=(10001, 10002), seconds=5, walking_stage=None):
    env = BalanceEnv(task=task, max_seconds=seconds, walking_stage=walking_stage)
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
                forward_distance=float(np.mean([row['forward_distance'] for row in results])), seconds=seconds, seeds=list(seeds), stage=walking_stage,
                target_speed=STAGES[walking_stage]['target_speed'] if walking_stage is not None else None,
                upright_ratio=float(np.mean([row['upright_ratio'] for row in results])),
                forward_speed=float(np.mean([row['forward_speed'] for row in results])),
                mean_slip_speed=float(np.mean([row['mean_slip_speed'] for row in results])),
                body_contact_ratio=float(np.mean([row['body_contact_ratio'] for row in results])),
                foot_cycles=float(np.mean([row['foot_cycles'] for row in results])),
                moving_legs=float(np.mean([row['moving_legs'] for row in results])),
                fall_rate=float(np.mean([row['reason']=='fall' for row in results])), trials=results)


def create_model(envs, config, checkpoint=None):
    """Transfer preserves actor parameters; new task gets fresh value head/optimizer.

    Appended speed/direction channels start at zero weight, so migrating a
    55-input policy retains its action means exactly for the original inputs.
    """
    old = PPO.load(checkpoint, device='cpu') if checkpoint else None
    transfer = config.get('initialization') == 'transfer'
    changed_inputs = old is not None and old.observation_space.shape != envs.observation_space.shape
    if old is not None and not transfer and not changed_inputs:
        return PPO.load(checkpoint, env=envs, device='cpu')
    walking = config['task'] == 'walk'
    model = PPO('MlpPolicy', envs, device='cpu', seed=config['seed'],
                n_steps=512 if walking else 128, batch_size=128 if walking else 64,
                n_epochs=6 if walking else 4, learning_rate=1e-4 if old is not None else 3e-4,
                ent_coef=.002 if walking else .005,
                policy_kwargs=dict(net_arch=dict(pi=[32,32], vf=[32,32]), log_std_init=-1.2 if walking else -.7), verbose=0)
    if old is not None:
        source, destination = old.policy.state_dict(), model.policy.state_dict()
        for name, tensor in destination.items():
            if name.startswith('value_net.') and (transfer or changed_inputs):
                continue  # Different reward scales; relearn value estimates.
            original = source[name]
            if tensor.shape == original.shape:
                tensor.copy_(original)
            elif name in ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.value_net.0.weight') and tensor.shape[1] == original.shape[1]+4:
                tensor.zero_()
                tensor[:, :original.shape[1]].copy_(original)
            else:
                raise ValueError('Source checkpoint architecture is incompatible with walking transfer')
        model.policy.load_state_dict(destination)
        model.num_timesteps = old.num_timesteps
    return model


def save_policy(model, path):
    """Publish complete archives so playback never observes a partial write."""
    path = Path(path)
    temporary = path.with_name('.'+path.stem+'-pending.zip')
    model.save(temporary)
    try:
        for attempt in range(20):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.025)
    finally:
        temporary.unlink(missing_ok=True)


def run_training(config, directory, checkpoint, output, paused, stop, fast):
    torch.set_num_threads(1)
    directory = Path(directory)
    (directory/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    walking_stage = config.get('stage',0) if config['task'] == 'walk' and config.get('walk_version') == 2 else None
    curriculum = WalkingCurriculum(walking_stage or 0, config.get('curriculum',True))
    factories = [partial(make_env, config['task'], walking_stage) for _ in range(config['worlds'])]
    envs = DummyVecEnv(factories) if config['worlds'] == 1 else SubprocVecEnv(factories, start_method='spawn')
    envs.seed(config['seed'])
    try:
        model = create_model(envs, config, checkpoint)
        initial = [layer.weight.detach().cpu().numpy().copy() for layer in actor_layers(model)]
        save_policy(model, directory/'initial.zip')
        baseline = evaluate(None, config['task'], seconds=10 if walking_stage is not None else 5, walking_stage=walking_stage)
        evaluations = [dict(steps=0, **evaluate(model, config['task'], seconds=10 if walking_stage is not None else 5, walking_stage=walking_stage))]
        best_scores = {}
        if walking_stage is not None:
            first = evaluations[0]
            best_scores[walking_stage] = first['forward_speed']-first['mean_slip_speed']-.5*first['fall_rate']-.5*first['body_contact_ratio']
            save_policy(model, directory/f'best-stage-{walking_stage}.zip')
        history = []
        checkpoint_path = directory/'policy.zip'

        class LiveCallback(BaseCallback):
            last_publish = 0
            run_start_steps = model.num_timesteps
            last_evaluation = 0
            latest_scene = None
            generalization = None

            def publish(self, status='running', force=False):
                now = time.monotonic()
                if not force and now-self.last_publish < .15:
                    return
                self.last_publish = now
                world = envs.env_method('training_snapshot', indices=0)[0]
                self.latest_scene = world['scene']
                steps = self.num_timesteps-self.run_start_steps
                metrics = dict(stage=world.get('stage'), target_speed=world.get('target_speed'),
                               foot_cycles=world.get('foot_cycles',0), mean_slip_speed=world.get('mean_slip_speed',0), steps=steps, total_steps=self.num_timesteps, updates=model._n_updates,
                               episodes=len(history), current_reward=world['episode_reward'],
                               current_upright_seconds=world['upright_seconds'], worlds=config['worlds'],
                               progress=min(1, steps/config['steps']), speed='fast' if fast.value else 'realtime',
                               losses={key: float(value) for key, value in model.logger.name_to_value.items()
                                       if key in ('train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss')})
                payload = dict(generalization=self.generalization, status=status, run_id=directory.name, config=config, scene=world['scene'], metrics=metrics,
                               history=history[-200:], evaluations=evaluations[-100:], baseline=baseline,
                               brain=brain_snapshot(model, world['observation'], initial),
                               curriculum=dict(stage=curriculum.stage, name=STAGES[curriculum.stage]['name'],
                                               streak=curriculum.streak, transitions=curriculum.transitions) if walking_stage is not None else None,
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
                nonlocal baseline
                self.publish('evaluating', force=True)
                stage = curriculum.stage if walking_stage is not None else None
                if baseline.get('stage') != stage:
                    baseline = evaluate(None, config['task'], seconds=10, walking_stage=stage)
                config['checkpoint_stage'] = stage
                (directory/'config.json').write_text(json.dumps(config, indent=2)+'\n')
                save_policy(model, checkpoint_path)
                steps = self.num_timesteps-self.run_start_steps
                result = dict(steps=steps, **evaluate(model, config['task'], seconds=10 if stage is not None else 5, walking_stage=stage))
                evaluations.append(result)
                self.last_evaluation = steps
                with (directory/'evaluation.jsonl').open('a') as log:
                    log.write(json.dumps(result, allow_nan=False)+'\n')
                if stage is not None:
                    score = result['forward_speed']-result['mean_slip_speed']-.5*result['fall_rate']-.5*result['body_contact_ratio']
                    if score > best_scores.get(stage, -float('inf')):
                        best_scores[stage] = score
                        save_policy(model, directory/f'best-stage-{stage}.zip')
                    if curriculum.consider(result, steps):
                        save_policy(model, directory/f'stage-{stage}.zip')
                        envs.env_method('set_walking_stage', curriculum.stage)
                self.publish(force=True)

            def _on_training_end(self):
                self.record_evaluation()
                if walking_stage is not None and not stop.is_set():
                    self.generalization = evaluate(model, config['task'], seeds=(20001,20002,20003), seconds=10,
                                                   walking_stage=config['checkpoint_stage'])
                    (directory/'generalization.json').write_text(json.dumps(self.generalization, indent=2)+'\n')
                self.publish('stopped' if stop.is_set() else 'finished', force=True)

        callback = LiveCallback()
        model.learn(total_timesteps=config['steps'], callback=callback, reset_num_timesteps=not bool(checkpoint))
    finally:
        envs.close()
