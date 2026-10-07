import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from gymnasium.utils.env_checker import check_env
from physics.environment import BalanceEnv
from physics.walking import WalkingCurriculum
from physics.training import validate_start


class WalkingTests(unittest.TestCase):
    def test_command_and_new_observation_contract(self):
        with self.assertRaises(ValueError):
            validate_start(dict(task='balance', initialization='transfer', source_policy='x/policy.zip'))
        env = BalanceEnv(task='walk', walking_stage=0, max_seconds=.1)
        try:
            check_env(env, skip_render_check=True)
            observation, _ = env.reset(seed=1)
            self.assertEqual(observation.shape, (59,))
            self.assertAlmostEqual(observation[55], .1)
            env.set_walking_stage(2)
            self.assertEqual(env.walking_stage, 0)
            observation, _ = env.reset(seed=1)
            self.assertEqual(env.walking_stage, 2)
            self.assertAlmostEqual(observation[55], .35)
        finally:
            env.close()

    def test_sliding_and_dragging_do_not_earn_walking_credit(self):
        obs = dict(upright=1, height=.98, nonFootContact=False, linearVelocity=[0,0,.1])
        stable = BalanceEnv.walking_terms(obs,.1,0,1,.1)
        slip = BalanceEnv.walking_terms(obs,.1,1,1,.1)
        self.assertLess(sum(slip.values()),sum(stable.values()))
        dragging = BalanceEnv.walking_terms({**obs,'nonFootContact':True},.1,0,1,.1)
        self.assertEqual(dragging['forward'],0)
        self.assertEqual(dragging['tracking'],0)
        self.assertEqual(dragging['footsteps'],0)
        self.assertLess(dragging['dragging'],0)
        falling = BalanceEnv.walking_terms({**obs,'upright':.4},.1,0,1,.1)
        self.assertEqual(falling['forward'],0)

    def test_curriculum_requires_repeated_genuine_progress(self):
        curriculum = WalkingCurriculum()
        successful = dict(upright_ratio=1,forward_speed=.1,mean_slip_speed=.01,
                          body_contact_ratio=0,moving_legs=4,foot_cycles=4,fall_rate=0)
        self.assertFalse(curriculum.consider({**successful,'foot_cycles':0},1))
        self.assertFalse(curriculum.consider(successful,2))
        self.assertFalse(curriculum.consider({**successful,'mean_slip_speed':.5},3))
        for step in (4,5):
            self.assertFalse(curriculum.consider(successful,step))
        self.assertTrue(curriculum.consider(successful,6))
        self.assertEqual(curriculum.stage,1)
        self.assertEqual(curriculum.transitions[0]['steps'],6)
        self.assertFalse(WalkingCurriculum(enabled=False).consider(successful,10))

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Training dependencies required')
    def test_transfer_preserves_actor_commands_and_source_archive(self):
        from physics.learner import PPO, create_model
        from stable_baselines3.common.vec_env import DummyVecEnv
        old_env = BalanceEnv()
        new_envs = DummyVecEnv([lambda:BalanceEnv(task='walk',walking_stage=0)])
        try:
            original = PPO('MlpPolicy', old_env, n_steps=128, batch_size=64,
                           policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32])),seed=2)
            original.num_timesteps = 30000
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'balance.zip'
                original.save(path)
                before = hashlib.sha256(path.read_bytes()).digest()
                config = validate_start(dict(task='walk', initialization='transfer',source_policy='run/policy.zip'))
                moved = create_model(new_envs, config, path)
                observation,_ = old_env.reset(seed=42)
                extended = np.concatenate([observation,[.1,0,0,1]]).astype(np.float32)
                np.testing.assert_allclose(original.predict(observation,deterministic=True)[0],moved.predict(extended,deterministic=True)[0],atol=1e-6)
                self.assertEqual(moved.num_timesteps,30000)
                self.assertEqual(moved.observation_space.shape,(59,))
                self.assertTrue(np.all(moved.policy.mlp_extractor.policy_net[0].weight.detach().numpy()[:,-1]==0))
                self.assertEqual(before,hashlib.sha256(path.read_bytes()).digest())
        finally:
            old_env.close()
            new_envs.close()

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Training dependencies required')
    def test_parallel_transfer_training_and_new_policy_playback(self):
        import json
        import time
        from physics.learner import PPO
        from physics.training import TrainingManager
        from physics.playback import PlaybackSession, saved_policies
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'standing'
            source.mkdir()
            (source/'config.json').write_text(json.dumps({'task':'balance','steps':30000}))
            env = BalanceEnv()
            try:
                model = PPO('MlpPolicy',env,n_steps=128,batch_size=64,
                            policy_kwargs=dict(net_arch=dict(pi=[32,32],vf=[32,32])),seed=1)
                model.num_timesteps=30000
                model.save(source/'policy.zip')
            finally:
                env.close()
            original_hash=hashlib.sha256((source/'policy.zip').read_bytes()).digest()
            manager=TrainingManager(root)
            try:
                manager.command(dict(type='start',task='walk',initialization='transfer',source_policy='standing/policy.zip',
                                     worlds=2,steps=1024,speed='fast'))
                deadline=time.monotonic()+90
                while time.monotonic()<deadline:
                    result=manager.poll()
                    if result['status'] in ('finished','error'):
                        break
                    time.sleep(.1)
                self.assertEqual(result['status'],'finished',result.get('error'))
                self.assertEqual(result['brain']['sizes'][0],59)
                self.assertEqual(result['metrics']['steps'],1024)
                self.assertGreater(result['brain']['weightChange'],0)
                self.assertEqual(result['metrics']['total_steps'],31024)
                self.assertTrue(Path(result['checkpoint']).with_name('best-stage-0.zip').is_file())
                self.assertEqual(original_hash,hashlib.sha256((source/'policy.zip').read_bytes()).digest())
                identifier=f"{result['run_id']}/policy.zip"
                self.assertIn(identifier,{entry['id'] for entry in saved_policies(root)})
                player=PlaybackSession(root)
                try:
                    player.load(dict(policy=identifier,seed=10001,seconds=1))
                    self.assertEqual(player.env.observation_space.shape,(59,))
                    while player.status=='playing':
                        player.step()
                    self.assertEqual(player.snapshot()['status'],'finished')
                finally:
                    player.close()
            finally:
                manager.close()
