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
