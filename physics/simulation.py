"""Deterministic fixed-step simulation; no browser or server required."""
import math
import random
import pybullet as p


class Simulation:
    dt = 1 / 60

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
        self.actions = [0.0] * 4
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
        leg_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.09, .35, .09],
                                          collisionFramePosition=[0, -.35, 0], physicsClientId=self.client)
        self.torso = p.createMultiBody(
            baseMass=3, baseCollisionShapeIndex=shape([.8, .45, 1.1]), basePosition=[0, 1.25, 0],
            linkMasses=[.6]*4, linkCollisionShapeIndices=[leg_shape]*4,
            linkVisualShapeIndices=[-1]*4,
            linkPositions=[[x, -.22, z] for x in [-.32, .32] for z in [-.4, .4]],
            linkOrientations=[[0, 0, 0, 1]]*4, linkInertialFramePositions=[[0, -.35, 0]]*4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]]*4, linkParentIndices=[0]*4,
            linkJointTypes=[p.JOINT_REVOLUTE]*4, linkJointAxis=[[1, 0, 0]]*4,
            physicsClientId=self.client)
        self.bodies.append(dict(name='torso', size=[.8, .45, 1.1], color=0x55ddbd, id=self.torso, link=-1))
        for i in range(4):
            self.bodies.append(dict(name='leg', size=[.18, .7, .18], color=0xa8f4de, id=self.torso, link=i))
        rng = random.Random(seed)
        for _ in range(7):
            box('object', [.6]*3, [(rng.random()-.5)*12, .5, 2+rng.random()*6], .8, 0xf0ba72)
        for body in self.bodies:
            p.changeDynamics(body['id'], body['link'], lateralFriction=.65,
                             linearDamping=.15, angularDamping=.3, physicsClientId=self.client)

    def step(self, mode='idle', drive=0, turn=0, actions=None):
        if actions is None:
            actions = [math.sin(self.time*2.8+i*1.7)*2 if mode == 'explore' else
                       drive*2+(-turn if i < 2 else turn) if mode == 'manual' else 0 for i in range(4)]
        if len(actions) != 4 or not all(math.isfinite(a) for a in actions):
            raise ValueError('Expected four finite joint velocity commands')
        self.actions = [max(-4, min(4, float(a))) for a in actions]
        p.setJointMotorControlArray(self.torso, range(4), p.VELOCITY_CONTROL,
                                   targetVelocities=self.actions, forces=[4]*4, physicsClientId=self.client)
        p.stepSimulation(physicsClientId=self.client)
        self.time += self.dt
        return self.observe()

    def observe(self):
        position, rotation = p.getBasePositionAndOrientation(self.torso, physicsClientId=self.client)
        velocity, _ = p.getBaseVelocity(self.torso, physicsClientId=self.client)
        joints = p.getJointStates(self.torso, range(4), physicsClientId=self.client)
        return dict(height=position[1], upright=p.getMatrixFromQuaternion(rotation)[4],
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
        return dict(time=self.time, actions=self.actions, observations=self.observe(), bodies=bodies)
