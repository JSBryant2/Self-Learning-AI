"""Deterministic fixed-step simulation; no browser or server required."""
import math
import random
from pathlib import Path
import pybullet as p


class Simulation:
    dt = 1 / 60
    leg_names = ('FL', 'FR', 'RL', 'RR')
    action_size = 12
    neutral = [0.0, .35, -.7] * 4

    def __init__(self, seed=1):
        self.client = p.connect(p.DIRECT)
        self.reset(seed)

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def reset(self, seed=1):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, -9.81, 0, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=50, deterministicOverlappingPairs=1,
                                   physicsClientId=self.client)
        self.time = 0
        self.actions = [0.0] * self.action_size
        self.bodies = []

        def shape(size):
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size],
                                          physicsClientId=self.client)

        def box(name, size, position, mass, color):
            body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape(size),
                                     basePosition=position, physicsClientId=self.client)
            self.bodies.append(dict(name=name, size=size, color=color, id=body, link=-1))
            return body

        box('ground', [24, .5, 24], [0, -.25, 0], 0, 0x172d32)
        self.torso = p.loadURDF(str(Path(__file__).with_name('quadruped.urdf')),
                                basePosition=[0, 1.0, 0],
                                flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                                physicsClientId=self.client)
        self.bodies.append(dict(name='torso', size=[.85, .3, 1.2], color=0x55ddbd, id=self.torso, link=-1))
        self.joint_ids, self.joint_names, self.limits, self.forces, self.foot_ids = [], [], [], [], []
        sizes = {'hip': [.14]*3, 'upper': [.12,.4,.12], 'lower': [.1,.4,.1], 'foot': [.22,.08,.3]}
        for index in range(p.getNumJoints(self.torso, physicsClientId=self.client)):
            info = p.getJointInfo(self.torso, index, physicsClientId=self.client)
            name = info[12].decode()
            segment = name.split('_')[-1]
            self.bodies.append(dict(name=name, size=sizes[segment], color=0xf0ba72 if segment=='foot' else 0xa8f4de,
                                    id=self.torso, link=index))
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_ids.append(index)
                self.joint_names.append(info[1].decode())
                self.limits.append([info[8], info[9]])
                self.forces.append(info[10])
            if segment == 'foot':
                self.foot_ids.append(index)
        for index, angle in zip(self.joint_ids, self.neutral):
            p.resetJointState(self.torso, index, angle, physicsClientId=self.client)
        rng = random.Random(seed)
        for _ in range(7):
            box('object', [.6]*3, [(rng.random()-.5)*12, .5, 2+rng.random()*6], .8, 0xf0ba72)
        for body in self.bodies:
            p.changeDynamics(body['id'], body['link'], lateralFriction=.65,
                             linearDamping=.15, angularDamping=.3, physicsClientId=self.client)

    def step(self, mode='idle', drive=0, turn=0, actions=None):
        """12 normalized position commands, leg-major: roll, pitch, knee.

        Zero holds a bent-knee neutral stance. ±1 maps to each joint limit.
        Controllers are scripted demonstrations, not learned policies.
        """
        if mode not in ('idle', 'explore', 'manual'):
            raise ValueError('Unknown controller')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   and abs(v) <= 1 for v in (drive, turn)):
            raise ValueError('Drive and turn must be finite numbers in [-1, 1]')
        if actions is None:
            actions = []
            for leg in range(4):
                phase = self.time*4 + (math.pi if leg in (1, 2) else 0)
                strength = .45 if mode == 'explore' else drive*.45 if mode == 'manual' else 0
                actions.extend([turn*.2 if mode == 'manual' else 0,
                                strength*math.sin(phase), -abs(strength)*max(0, math.cos(phase))])
        if len(actions) != self.action_size or not all(isinstance(a, (int, float)) and not isinstance(a, bool)
                                                       and math.isfinite(a) for a in actions):
            raise ValueError('Expected 12 finite normalized position commands')
        self.actions = [max(-1, min(1, float(a))) for a in actions]
        targets = [neutral+a*((high-neutral) if a >= 0 else (neutral-low))
                   for a, neutral, (low, high) in zip(self.actions, self.neutral, self.limits)]
        # Substeps improve joint-limit/contact stability without changing the public 60 Hz step.
        p.setTimeStep(self.dt/4, physicsClientId=self.client)
        for _ in range(4):
            for index, target, force in zip(self.joint_ids, targets, self.forces):
                p.setJointMotorControl2(self.torso, index, p.POSITION_CONTROL,
                                       targetPosition=target, force=force, maxVelocity=4,
                                       positionGain=.15, velocityGain=1, physicsClientId=self.client)
            p.stepSimulation(physicsClientId=self.client)
        self.time += self.dt
        return self.observe()

    def observe(self):
        position, rotation = p.getBasePositionAndOrientation(self.torso, physicsClientId=self.client)
        velocity, angular_velocity = p.getBaseVelocity(self.torso, physicsClientId=self.client)
        joints = p.getJointStates(self.torso, self.joint_ids, physicsClientId=self.client)
        contacts = p.getContactPoints(bodyA=self.torso, physicsClientId=self.client)
        foot_forces = [sum(c[9] for c in contacts if c[3] == index and c[2] != self.torso)
                       for index in self.foot_ids]
        foot_states = [p.getLinkState(self.torso, index, computeLinkVelocity=True,
                                      physicsClientId=self.client) for index in self.foot_ids]
        non_foot_contact = any(c[3] not in self.foot_ids and c[2] != self.torso and c[9] > .01
                               for c in contacts)
        return dict(footPositions=[list(state[0]) for state in foot_states],
                    footVelocities=[list(state[6]) for state in foot_states], nonFootContact=non_foot_contact,
                    orientation=list(rotation), linearVelocity=list(velocity), angularVelocity=list(angular_velocity),
                    footContacts=[force > .01 for force in foot_forces], footForces=foot_forces,
                    jointTorques=[j[3] for j in joints], height=position[1], upright=p.getMatrixFromQuaternion(rotation)[4],
                    speed=math.sqrt(sum(v*v for v in velocity)),
                    distance=math.hypot(position[0], position[2]-7), position=list(position),
                    jointAngles=[j[0] for j in joints], jointVelocities=[j[1] for j in joints])

    def snapshot(self):
        bodies = []
        for item in self.bodies:
            if item['link'] == -1:
                position, rotation = p.getBasePositionAndOrientation(item['id'], physicsClientId=self.client)
            else:
                # Link COM is at the centre of the rendered leg box.
                state = p.getLinkState(item['id'], item['link'], computeForwardKinematics=True,
                                       physicsClientId=self.client)
                position, rotation = state[0], state[1]
            bodies.append({k: item[k] for k in ('name', 'size', 'color')} |
                          dict(position=list(position), quaternion=list(rotation)))
        return dict(time=self.time, actions=self.actions, observations=self.observe(), bodies=bodies,
                    joints=[dict(name=name, limits=limits, neutral=neutral, maxTorque=force)
                            for name, limits, neutral, force in zip(self.joint_names, self.limits, self.neutral, self.forces)],
                    legs=list(self.leg_names), actionType='normalized_joint_position')
