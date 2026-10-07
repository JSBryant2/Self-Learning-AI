import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from physics.playback import PlaybackSession, saved_policies, validate_load
from physics.environment import BalanceEnv

HAS_TRAINING = importlib.util.find_spec('stable_baselines3') is not None


@unittest.skipUnless(HAS_TRAINING, 'Install requirements-training.txt for policy playback')
class PlaybackTests(unittest.TestCase):
    def setUp(self):
        from physics.learner import PPO
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.run = self.root/'test-run'
        self.run.mkdir()
        (self.run/'config.json').write_text(json.dumps({'task':'balance','steps':128}))
        env = BalanceEnv()
        try:
            model = PPO('MlpPolicy', env, n_steps=128, batch_size=64, seed=1,
                        policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32])), device='cpu')
            model.save(self.run/'initial.zip')
            model.num_timesteps = 128
            model.save(self.run/'policy.zip')
        finally:
            env.close()
        self.session = PlaybackSession(self.root)
        self.addCleanup(self.session.close)

    def load(self, **kwargs):
        self.session.command(dict(type='load', policy='test-run/policy.zip', seed=10001, seconds=1, **kwargs))

    def test_catalog_and_validation(self):
        self.assertEqual({entry['id'] for entry in saved_policies(self.root)}, {'test-run/initial.zip','test-run/policy.zip'})
        for message in [{'policy':'../../README.md'}, {'policy':'test-run/policy.zip','seed':True},
                        {'policy':'test-run/policy.zip','seconds':float('nan')}]:
            with self.assertRaises(ValueError):
                validate_load(message, self.root)
        broken = self.root/'bad-run'
        broken.mkdir()
        (broken/'config.json').write_text('[]')
        self.assertEqual(len(saved_policies(self.root)), 2)

    def test_read_only_deterministic_pause_restart_and_completion(self):
        path = self.run/'policy.zip'
        before = hashlib.sha256(path.read_bytes()).digest()
        self.load()
        weights = {name:value.clone() for name,value in self.session.model.policy.state_dict().items()}
        first = [self.session.step()['scene'] for _ in range(8)]
        self.session.command({'type':'pause'})
        paused = self.session.snapshot()
        self.assertEqual(self.session.step(), paused)
        self.session.command({'type':'restart'})
        second = [self.session.step()['scene'] for _ in range(8)]
        self.assertEqual(first, second)
        while self.session.status == 'playing':
            self.session.step()
        result = self.session.snapshot()
        self.assertEqual(result['status'], 'finished')
        self.assertEqual(result['result']['reason'], 'time_limit')
        self.assertEqual(result['saved_steps'], 128)
        self.assertEqual(len(result['results']), 1)
        for name,value in self.session.model.policy.state_dict().items():
            self.assertTrue(np.array_equal(value.numpy(), weights[name].numpy()))
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).digest())
        self.session.command({'type':'load','policy':'test-run/initial.zip','seed':10001,'seconds':1})
        self.assertEqual(self.session.snapshot()['saved_steps'], 0)
        self.assertEqual(len(self.session.results), 1)

    def test_invalid_checkpoint_keeps_current_session(self):
        self.load()
        old = self.session.model
        (self.run/'initial.zip').write_bytes(b'not a checkpoint')
        with self.assertRaises(Exception):
            self.session.command({'type':'load','policy':'test-run/initial.zip'})
        self.assertIs(self.session.model, old)
        self.assertEqual(self.session.status, 'playing')
