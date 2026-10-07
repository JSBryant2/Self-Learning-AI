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
        self.assertNotEqual(a.snapshot()['bodies'][6]['position'], b.snapshot()['bodies'][6]['position'])
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
        self.assertEqual(sim.actions, [1.5, 1.5, 2.5, 2.5])
        sim.step(actions=[9, -9, 1, 0])
        self.assertEqual(sim.actions, [4, -4, 1, 0])
        with self.assertRaises(ValueError):
            sim.step(actions=[float('nan')]*4)

    def test_protocol_validation(self):
        for message in [[], {'type':'reset','seed':0}, {'type':'control','drive':float('nan')},
                        {'type':'control','mode':'bad'}, {'type':'control','paused':'yes'}]:
            with self.assertRaises(ValueError):
                validate(message)
        self.assertEqual(validate({'type':'reset','seed':42}), ('reset',42))
