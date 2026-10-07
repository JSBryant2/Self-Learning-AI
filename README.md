# Origin — Self-Learning AI

An expandable 3D playground. Python owns all physics using PyBullet; Three.js renders snapshots in the browser. No learning algorithm is connected yet. Scripted exploration is a fixed motor pattern, not learned walking.

## Setup

Use Python **3.11** (PyBullet has a binary wheel for this version) and Node.js 22.12+.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --only-binary :all: -r requirements.txt
npm ci
```

On Windows (PowerShell), use:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary :all: -r requirements.txt
npm ci
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

Run `npm run dev` in the second terminal on all platforms. Open the Vite address (usually http://localhost:5173) in a browser. Vite proxies `/physics` WebSocket connections to Python on localhost port 8765. Each viewer owns an independent world; closing the viewer releases it. The Python server is local-only and is a development service, not a public deployment. A production static build needs an equivalent WebSocket reverse proxy.

Drag to orbit, scroll to zoom. Select passive, scripted exploration, or manual joint controls (WASD / arrows). Pause freezes Python stepping; reset rebuilds the world with the entered seed while preserving pause/controller selection. A disconnected viewer displays a status message; restart the backend and reload to reconnect.

## Physics and future learning

`physics/simulation.py` supplies `Simulation.reset(seed)`, `step(actions=[...])`, `observe()`, `snapshot()` and `close()`. It runs headlessly without a browser or server for training. Four velocity commands are clamped to ±4 rad/s with motor torque limited to 4 N·m. Observations include body position, height, uprightness, speed, target distance, joint angles and velocities.

Physics steps at 60 Hz using gravity, rigid-body contact and four revolute joints. Python streams snapshots over WebSocket; the browser only renders them and sends control/reset commands. Rendering rate does not advance physics. Slow connections can reduce real-time throughput; training should call the simulation directly.

PyBullet replaces cannon-es, so trajectories and seeded layouts differ from the earlier prototype. Reproducibility is checked within the same installed engine/platform, not promised across engines or platforms. No rewards, policy, camera perception or experience storage yet.

## Validate

```sh
npm test
npm run build
```

On Windows run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` instead of `npm test`.

Tests cover seeded resets, independent simulations, contact/gravity, actuator stability, direct action commands and protocol validation. The build may report a non-blocking Three.js bundle-size warning.
