import json
import os
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'origin-matplotlib'))
from physics.environment import BalanceEnv
from physics.training import TrainingManager, validate_start
import multiprocessing as mp
import importlib.util
HAS_TRAINING = importlib.util.find_spec('stable_baselines3') is not None
if HAS_TRAINING:
    from stable_baselines3 import PPO
    from physics.learner import actor_layers, brain_snapshot, run_training


class Collector:
    def __init__(self):
        self.rows = []

    def put_nowait(self, payload):
        # Snapshot dictionaries stay unchanged after publication.
        self.rows.append(payload)


@unittest.skipUnless(HAS_TRAINING, 'Install requirements-training.txt to exercise PPO')
class TrainingTests(unittest.TestCase):
    def test_command_validation(self):
        self.assertEqual(validate_start({})['worlds'], 1)
        for command in [{'worlds': 0}, {'worlds': True}, {'steps': 127}, {'task':'bad'}, {'speed':'bad'}]:
            with self.assertRaises(ValueError):
                validate_start(command)

    def test_real_ppo_updates_weights_and_saves_model(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.get_context('spawn')
            output = Collector()
            config = validate_start(dict(steps=128, speed='fast'))
            run_training(config, directory, None, output, context.Event(), context.Event(), context.Value('b', True))
            final = output.rows[-1]
            self.assertEqual(final['status'], 'finished')
            self.assertGreater(final['metrics']['updates'], 0)
            self.assertGreater(final['brain']['weightChange'], 0)
            self.assertEqual(final['brain']['sizes'], [55, 32, 32, 12])
            self.assertEqual(len(final['scene']['actions']), 12)
            self.assertTrue(np.isfinite(final['brain']['inputSensitivity']).all())
            self.assertEqual(final['evaluations'][-1]['seeds'], [10001,10002])
            path = Path(directory)/'policy.zip'
            self.assertTrue(path.is_file())
            self.assertTrue(Path(directory, 'config.json').is_file())
            env = BalanceEnv()
            try:
                obs, _ = env.reset(seed=8)
                model = PPO.load(path, device='cpu')
                initial = PPO.load(Path(directory)/'initial.zip', device='cpu')
                self.assertFalse(np.array_equal(model.predict(obs, deterministic=True)[0], initial.predict(obs, deterministic=True)[0]))
                weights = [layer.weight.detach().numpy().copy() for layer in actor_layers(model)]
                brain = brain_snapshot(model, obs, weights)
                np.testing.assert_allclose(brain['meanActions'], model.predict(obs, deterministic=True)[0], atol=1e-6)
            finally:
                env.close()

    def test_two_actual_worker_worlds_share_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = TrainingManager(Path(directory))
            try:
                manager.command(dict(type='start', worlds=2, steps=256, speed='fast'))
                deadline = time.monotonic()+60
                while time.monotonic() < deadline:
                    state = manager.poll()
                    if state['status'] in ('finished', 'error'):
                        break
                    time.sleep(.1)
                self.assertEqual(state['status'], 'finished', state.get('error'))
                self.assertEqual(state['metrics']['worlds'], 2)
                self.assertEqual(state['metrics']['steps'], 256)
                self.assertGreater(state['brain']['weightChange'], 0)
                self.assertTrue(Path(state['checkpoint']).is_file())
                fresh_manager = TrainingManager(Path(directory))
                self.assertEqual(fresh_manager.last_checkpoint, state['checkpoint'])
                self.assertEqual(fresh_manager.last_task, 'balance')
            finally:
                manager.close()
