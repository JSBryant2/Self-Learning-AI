# Origin — Self-Learning AI

An expandable 3D playground. Python owns all physics using PyBullet; Three.js renders snapshots in the browser. No learning algorithm is connected yet. Scripted exploration is a fixed motor pattern, not learned walking.

## Setup

Use Python **3.11** (Linux x86_64 has a PyBullet wheel; Windows needs compilation) and Node.js 22.12+.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --only-binary :all: -r requirements.txt
npm ci
```

On Windows install Microsoft C++ Build Tools with **Desktop development with C++** and a Windows SDK, then use PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci
```

Install Git, Python 3.11 and Node.js first. Get the project with:

```sh
git clone https://github.com/JSBryant2/Self-Learning-AI.git
cd Self-Learning-AI
```

On Apple Silicon Macs, PyBullet wheel availability may require an x86_64 Python 3.11 under Rosetta. The binary-only install fails rather than silently compiling if no compatible wheel is available.

## Run

From the repository root, run these in two terminals:

```sh
.venv/bin/python -m physics.server
```

```sh
npm run dev
```

On Windows, the Python startup command is:

```powershell
.\.venv\Scripts\python.exe -m physics.server
```

Run `npm run dev` in the second terminal (`npm.cmd run dev` on Windows if PowerShell blocks npm.ps1). Open the Vite address (usually http://localhost:5173) in a browser. Vite proxies `/physics` WebSocket connections to Python on localhost port 8765. Each viewer owns an independent world; closing the viewer releases it. The Python server is local-only and is a development service, not a public deployment. A production static build needs an equivalent WebSocket reverse proxy.

Drag to orbit, scroll to zoom. Select neutral standing, scripted exploration, or manual joint controls (WASD / arrows). Pause freezes Python stepping; reset rebuilds the world with the entered seed while preserving pause/controller selection. A disconnected viewer displays a status message; restart the backend and reload to reconnect.

## Physics and future learning

`physics/simulation.py` supplies `Simulation.reset(seed)`, `step(actions=[...])`, `observe()`, `snapshot()` and `close()`. It runs headlessly without a browser or server for training. The body in `physics/quadruped.urdf` has four articulated legs with two hip joints (roll and pitch), one knee and a fixed foot per leg. No actuated ankles yet. Adjacent links are excluded from self-collision; other body links can collide.

**Action interface v2:** pass 12 normalized position commands in leg-major order: FL, FR, RL, RR (front/rear, left/right), with hip roll, hip pitch, knee within each leg. +Z is forward and +Y is up. Actions are clamped to [-1, 1]; zero holds the neutral bent-knee posture, ±1 maps to the corresponding URDF joint limit. This replaces the earlier four velocity commands and is an intentional breaking change.

Hip roll limits: ±0.45 rad; hip pitch: ±0.9 rad; knee: -1.8 to -0.08 rad. Neutral angles are [0, 0.35, -0.7] rad per leg. Hip torque is capped at 12 N·m, knee torque at 10 N·m and target speed at 4 rad/s. Four internal physics substeps per public step improve contact stability.

Observations include body position/quaternion, linear/angular velocity, height, uprightness, target distance, 12 joint angles/velocities/applied motor torques, and four foot-contact flags/normal-force sums against external objects (N). Snapshots include joint names, limits, neutral angles, torque caps and leg ordering. Stand holds neutral posture; exploration demonstrates a diagonal-pair gait; manual forward/backward controls gait direction and turn requests hip roll. Turning is a demonstration control, not a validated navigation controller.

Physics steps at 60 Hz using gravity, rigid-body contact and 12 revolute joints. Python streams snapshots over WebSocket; the browser only renders them and sends control/reset commands. Rendering rate does not advance physics. Slow connections can reduce real-time throughput; training should call the simulation directly.

PyBullet replaces cannon-es, so trajectories and seeded layouts differ from the earlier prototype. Reproducibility is checked within the same installed engine/platform, not promised across engines or platforms. No rewards, policy, camera perception or experience storage yet.

## Validate

```sh
npm test
npm run build
```

On Windows run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` instead of `npm test`.

Tests cover seeded resets, independent simulations, contact/gravity, 12-joint stance with four-foot support, physical joint limits and motor torque caps, direct action commands, a 10-second scripted mobility baseline and protocol validation. This checks the body can support movement; it does not establish learned walking or general terrain ability. The build may report a non-blocking Three.js bundle-size warning.

## Balance training interface

`physics/environment.py` provides a Gymnasium `BalanceEnv` for future learning algorithms. No trained policy or PPO implementation is included yet. The browser still runs the demonstration controls; baseline rollouts run headlessly and do not appear in that viewer.

- `reset(seed=...)` starts a reproducible world with small randomized body tilt and joint angles. Repeating the same seed repeats the initial state. `perturbation=0` disables pose variation for calibration.
- `step(action)` applies **12 independent normalized position commands**, never the scripted walking controller. Default decision frequency is 30 Hz (two public physics steps per action).
- Attempts end on a fall (height <0.45 m or uprightness <0.45), leaving the central 20×20 m area, or reaching the default 20-second limit. Falls/boundaries return `terminated`; time limits return `truncated`. Call reset before stepping again. The rollout runner resets automatically between attempts.
- The 55-value float32 observation contains body-frame world-up, body-frame linear velocity /3, angular velocity /4, height in metres, 12 joint angles mapped relative to limits, 12 joint velocities /4, four foot contacts, four normal forces /100, 12 previous actions and remaining-time fraction. Exact order is in `BalanceEnv.observation_names`. Values are scaled but not clipped, and the observation space intentionally has unbounded numeric limits.
- Reward is accumulated per second: `0.75 * uprightness + 0.25 * height_quality - 0.02 * (motor_power / 48) - 0.02 * mean(action_change²)`. Uprightness is floored at zero. Height quality peaks at 0.98 m and reaches zero at a deviation of 0.53 m. Falling subtracts 1. There is **no forward-progress reward** in this initial balance task.
- Every step reports reward components. Completed episodes report reward, time upright, horizontal/forward displacement, motor work, length, reason and world seed.

Example policy integration:

```python
from physics.environment import BalanceEnv

env = BalanceEnv()
try:
    observation, info = env.reset(seed=1)
    while True:
        action = env.action_space.sample()  # replace with your learning policy
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(info['episode'])
            observation, info = env.reset()
finally:
    env.close()
```

Run baseline experiments on Windows after pulling the update:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m physics.rollout --policy zero --episodes 5 --output runs/zero.jsonl
.\.venv\Scripts\python.exe -m physics.rollout --policy random --episodes 5 --output runs/random.jsonl
```

On macOS/Linux substitute `.venv/bin/python`. Outputs are JSON Lines (one summary per episode), excluded from Git. Choose a new output filename for each run: existing logs are never overwritten. These are **baselines, not training**. Zero holds the existing neutral stance and is intentionally an easy calibration reference; later experiments will need disturbances, more difficult starts, forward movement objectives and held-out evaluation seeds. Learning gains must be compared against this baseline, not simply against random flailing.
