import math
import unittest
from physics.simulation import Simulation
from physics.server import validate


class PhysicsTests(unittest.TestCase):
    def make(self, seed=1):
        sim = Simulation(seed)
        self.addCleanup(sim.close)
        return sim

    def test_seed_reset_and_independent_clients(self):
        a, b = self.make(42), self.make(42)
        for _ in range(300):
            a.step('explore')
            b.step('explore')
        self.assertEqual(a.snapshot(), b.snapshot())
        a.reset(43)
        self.assertNotEqual(a.snapshot()['bodies'][-1]['position'], b.snapshot()['bodies'][-1]['position'])
        a.reset(42)
        self.assertEqual(a.snapshot(), self.make(42).snapshot())

    def test_gravity_contact_and_stability(self):
        sim = self.make()
        for _ in range(600):
            sim.step()
        self.assertGreater(sim.observe()['height'], .1)
        self.assertLess(sim.observe()['height'], 1.25)
        for body in sim.snapshot()['bodies']:
            self.assertTrue(all(math.isfinite(v) for v in body['position']+body['quaternion']))
            if body['name'] == 'object':
                self.assertGreater(body['position'][1], .25)

    def test_actuation_and_direct_training_api(self):
        sim, idle = self.make(), self.make()
        for _ in range(1800):
            sim.step('explore')
            idle.step()
        self.assertNotEqual(sim.observe()['position'], idle.observe()['position'])
        self.assertTrue(all(math.isfinite(v) for v in sim.observe()['jointAngles']))
        sim.step('manual', 1, .5)
        self.assertEqual(len(sim.actions), 12)
        sim.step(actions=[9, -9, 1]*4)
        self.assertEqual(sim.actions, [1, -1, 1]*4)
        with self.assertRaises(ValueError):
            sim.step(actions=[float('nan')]*12)

    def test_protocol_validation(self):
        for message in [[], {'type':'reset','seed':0}, {'type':'control','drive':float('nan')},
                        {'type':'control','mode':'bad'}, {'type':'control','paused':'yes'}]:
            with self.assertRaises(ValueError):
                validate(message)
        self.assertEqual(validate({'type':'reset','seed':42}), ('reset',42))

    def test_articulated_stance_and_senses(self):
        sim = self.make()
        self.assertEqual(len(sim.joint_ids), 12)
        self.assertEqual(len(sim.foot_ids), 4)
        self.assertEqual(len(sim.snapshot()['bodies']), 25)
        self.assertEqual(sim.observe()['footContacts'], [False]*4)
        for _ in range(600):
            sim.step()
        observation = sim.observe()
        self.assertGreater(observation['upright'], .98)
        self.assertEqual(observation['footContacts'], [True]*4)
        self.assertGreater(sum(observation['footForces']), 50)
        self.assertEqual(len(observation['jointTorques']), 12)
        for angle, neutral in zip(observation['jointAngles'], sim.neutral):
            self.assertAlmostEqual(angle, neutral, delta=.03)
        sim.reset()
        self.assertEqual(sim.observe()['footContacts'], [False]*4)

    def test_joint_limits_and_motor_torque(self):
        sim = self.make()
        for direction in (-1, 1):
            for _ in range(240):
                sim.step(actions=[direction]*12)
                for angle, (low, high) in zip(sim.observe()['jointAngles'], sim.limits):
                    self.assertGreaterEqual(angle, low-.06)
                    self.assertLessEqual(angle, high+.06)
                for torque, maximum in zip(sim.observe()['jointTorques'], sim.forces):
                    self.assertLessEqual(abs(torque), maximum+.001)
        for bad in ([0]*4, [True]*12, [float('inf')]*12):
            with self.assertRaises(ValueError):
                sim.step(actions=bad)

    def test_scripted_mobility_baseline(self):
        sim = self.make()
        for _ in range(600):
            sim.step('explore')
        observation = sim.observe()
        self.assertGreater(observation['position'][2], .5)
        self.assertGreater(observation['upright'], .9)
