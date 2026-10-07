# Origin — Self-Learning AI

A 3D embodied-learning playground: Python/PyBullet owns physics, PyTorch and Stable-Baselines3 train a randomly initialized PPO policy, and Three.js displays the **actual training world**, progress and actor-network measurements. No pretrained model or scripted walking examples are used by training.

## Install

Requires Python **3.11**, Node.js 22.12+ (Node 24 recommended) and Git. Clone the repository and open it in VS Code:

```sh
git clone https://github.com/JSBryant2/Self-Learning-AI.git
cd Self-Learning-AI
```

### Windows / PowerShell

PyBullet has no Windows wheel for the pinned version. Install Microsoft C++ Build Tools with **Desktop development with C++** and a Windows SDK before its first installation. If your existing project environment already works, keep it and skip the venv creation command.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
npm.cmd ci
```

The CPU PyTorch download is sizeable. No GPU is required. `npm.cmd` works when PowerShell blocks `npm.ps1`.

### Linux x86_64 / macOS

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Linux: install CPU PyTorch without downloading CUDA dependencies.
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-training.txt
npm ci
```

On macOS install PyTorch from the default index (`pip install torch==2.8.0`, omitting the CPU index option). PyBullet may require native build tools or, on Apple Silicon, an Intel Python under Rosetta. Windows and macOS instructions have not been executed on those platforms; runtime validation was on Linux x86_64.

For only the old demonstration viewer or baseline runner, `requirements.txt` is sufficient. Live learning requires `requirements-training.txt`.

## Start the viewer and training

From the project root, run these in **two terminals**.

Windows terminal 1:

```powershell
.\.venv\Scripts\python.exe -m physics.server
```

Windows terminal 2:

```powershell
npm.cmd run dev
```

On Linux/macOS use `.venv/bin/python -m physics.server` and `npm run dev`.

Open the Vite address, normally **http://localhost:5173**. In the viewer choose **Training · live world 1**, then use **Open learning lab** or scroll down:

1. Leave **Worlds = 1** to train one creature at a time.
2. Select **Balance first**, or **Balance + move forward** for the first locomotion objective.
3. Click **Start learning**. Initialization and fixed-seed evaluations can take a few seconds before movement begins.
4. **Real time** paces policy decisions at about 30 Hz; **Fast training** removes that delay and displays sampled frames. Rendering doesn't control physics.
5. **Pause learning** freezes collection; **Stop & save** ends the run and saves the policy. Evaluation/optimization may finish before a command takes effect.

To edit Python code: stop the server with Ctrl+C, restart it, and reload the browser. JavaScript/CSS changes normally update through Vite automatically.

### What the dashboard means

- **3D world:** the very same creature generating training experience, not a separate demonstration. Episode resets are automatic. World 1 is visible when training multiple worlds. During evaluation the last training frame remains visible and the status says Evaluating.
- **Episode results:** reward, time upright, forward displacement or motor work. Dots are completed episodes; the line averages the last 10. Until the first episode finishes, the chart is empty and current reward updates separately.
- **Evaluation:** deterministic policy trials on seeds 10001 and 10002, separate from gradient-training seeds, compared against neutral standing. Trials last 5 seconds and run initially, every ~1,024 experience steps, and at the end. This is a small fixed evaluation, not evidence of general intelligence or reliable walking on new terrain.
- **Actor graph:** real activations at the displayed observation and the strongest 80 weight connections per layer. Positive weights are teal, negative amber. It is a fixed 55→32→32→12 network; weights change, new neurons are not added. Training actions also contain exploration noise.
- **Weight heatmap:** every connection, with hover inspection; switch between current weights and changes since this run started. Color intensity is scaled within the selected matrix, so use numeric hover values for comparisons.
- **Input sensitivity:** mean absolute action-mean gradient at the displayed state. It shows local sensitivity to scaled inputs, not causal importance, concepts or thoughts. Biases are not drawn in the graph/heatmap. PPO trains a separate value network which is not depicted.

**Changing weights does not establish getting smarter.** Judge behavior and evaluation performance. Neutral standing already performs well at the simple balance task; improvements require comparing against that baseline. Walking may require more experience, reward tuning and later curriculum changes. No successful learned gait is promised by this initial implementation.

### Parallel worlds and persistence

Set **Worlds** from 1 to 8 before starting. One world uses a local vector environment; more worlds use genuinely separate spawned simulation processes (`SubprocVecEnv`, compatible with Windows). Each has its own physics state and contributes experience to **one shared PPO policy**. They do not communicate or live in a shared arena. More worlds require more CPU resources; PPO optimization is CPU-based.

Experience steps are aggregated across worlds. PPO finishes full batches, so it can exceed the requested budget by up to one rollout (128 steps × worlds).

Each run creates `runs/<run-id>/` containing:

- `config.json`: task, seed, budget and settings.
- `initial.zip`: policy at the beginning of this run.
- `policy.zip`: latest evaluated policy, saved every evaluation and at run end.
- `episodes.jsonl`: completed episode measurements.
- `evaluation.jsonl`: policy evaluations after initialization.

**Initialisation → Continue same task** resumes a selected checkpoint (or the latest checkpoint when no source is selected) from the same task. The server rediscovers the latest saved checkpoint after restart; if its task differs, select that task or start fresh. Continuation starts fresh environments and episode history, keeping network parameters and optimizer state; it is not an exact replay of interrupted simulation/RNG state. Checkpoints and logs are ignored by Git. Closing the browser does not stop a server-owned run. Stop/save before closing the Python server to preserve the latest work.

Use **Demonstration · separate world** to inspect scripted/manual controls independently. WASD does not override the training policy.

## Watch saved policies without training

Your laptop's existing `runs` folder is preserved by `git pull`; saved policies are not stored on GitHub. The app automatically lists `initial.zip` and `policy.zip` for each local run that has a valid `config.json`.

1. Start the same Python server and Vite viewer.
2. Select **Saved policies · no learning**, then **Open saved policies**, or scroll to the Saved policies panel.
3. Pick **Latest saved policy** for your run, leave the seed at 10001 and duration at 20 seconds, and click **Watch policy**. The 3D view switches to that checkpoint.
4. Use **Pause playback**, **Resume playback**, or **Restart same trial**. A trial freezes at its ending; it does not automatically start another attempt.
5. To compare, choose **Initial policy** from the same run and keep the seed/duration unchanged. Completed results appear in a table with reward, time upright, forward displacement, motor work and ending reason.

Playback uses the checkpoint's original task and deterministic action means, with no exploration noise or optimization. It creates its own environment and does not change checkpoint files, weights or training counters. The checkpoint is copied into memory at load time; refreshing the list and loading again picks up a newer save. The saved-experience count comes from the model, not the requested training budget. `initial.zip` can already contain learned weights when its run continued an earlier checkpoint.

Playback comparisons remain in the current connection (last 20 completed trials); reloading the page clears them. Different browsers have independent playback sessions. Closing the browser releases the playback world. The training server and its run can continue independently. The network charts in the Learning lab remain labelled **training world 1**; they do not depict the playback world.

Use **Refresh list** after new checkpoints are saved. No entries means the server cannot find compatible checkpoints under the repository's local `runs` directory. Existing archives must stay alongside their run's `config.json`; importing arbitrary external models is not supported.

## Body and training interface

`physics/quadruped.urdf` describes a torso and four legs with hip roll, hip pitch, knee and fixed feet. Twelve joints have physical angle limits, target speed 4 rad/s, hip torque caps 12 N·m and knee caps 10 N·m. Foot contacts and normal forces are sensed. +Y is up and +Z is forward.

`physics/simulation.py` exposes `reset`, `step(actions=...)`, `observe`, `snapshot`, `close`. Actions are leg-major **FL, FR, RL, RR**, with roll/pitch/knee per leg. Twelve values in [-1,1] select joint positions: zero holds neutral [0,0.35,-0.7] rad, ±1 selects each joint's low/high limit. Physics runs at 60 Hz with four internal substeps.

`physics/environment.py` supplies the Gymnasium `BalanceEnv`, used by training. Default decisions repeat each action for two physics steps (30 Hz). Its 55 float32 observations are ordered by `BalanceEnv.observation_names`: body-frame world-up, body-frame linear velocity /3, angular velocity /4, height, 12 joint angles scaled by limits, 12 joint velocities /4, four contacts, four normal forces /100, 12 previous actions, and remaining-time fraction. Scaled values are not clipped; observation bounds are intentionally unbounded.

Episodes start with reproducible small pose perturbations and end on a fall (height <0.45 m or uprightness <0.45), leaving the central 20×20 m area, or reaching 20 seconds. Falls/boundaries terminate; time limits truncate. The learner handles bootstrap values and resets through Stable-Baselines3.

Balance reward per second:

`0.75 × uprightness + 0.25 × height_quality − 0.02 × motor_power/48 − 0.02 × mean(action_change²)`

Uprightness is floored at zero. Height quality peaks at 0.98 m, decreasing to zero at a deviation of 0.53 m. A fall subtracts 1. Legacy walking checkpoints use an extra `0.5 × clipped_world_forward_velocity × uprightness` per second (velocity clipped to ±2 m/s). New walking runs use the curriculum described below. Body posture, reward definitions and motor servos are human-designed priors; a fresh network starts without learned knowledge.

## Validate and baseline experiments

```sh
npm test
npm run build
```

On Windows run `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` and `npm.cmd run build`. With training dependencies installed, 24 tests cover the Gymnasium contract, episodes, physics/joint limits, walking reward, real PPO weight updates, saved policy reload, two-process training, and deterministic/read-only playback. The PPO and saved-policy playback tests are skipped when only base dependencies are installed. The build has a non-blocking Three.js bundle-size warning.

The original headless baseline runner is still available (not learning):

```powershell
.\.venv\Scripts\python.exe -m physics.rollout --policy zero --episodes 5 --output runs/zero.jsonl
.\.venv\Scripts\python.exe -m physics.rollout --policy random --episodes 5 --output runs/random.jsonl
```

Choose a new output filename each time; baseline logs are never overwritten.

The server binds locally on 8765. Vite proxies `/physics`, `/training` and `/playback` WebSockets. This is a local development app; a production static build requires equivalent proxying. No external services, accounts or pretrained model downloads are needed at runtime.

## Transfer standing into a walking curriculum

New walking runs use an updated task (version 2). Your existing balance checkpoints and legacy walking checkpoints remain loadable and unchanged.

1. Pull this update and restart both servers. No new dependencies are needed.
2. In the Learning lab select **Initialisation → Transfer to walking**. This selects the walking task and suggests **150,000 additional experience steps** in fast mode.
3. Choose your successful **balance checkpoint** in **Source checkpoint**. If a new save is missing, press **Refresh list** in Saved policies to refresh both selectors.
4. Leave **Starting stage = first steps**, **Auto progression enabled**, and **Worlds = 1**, then start learning. You can watch sampled frames in fast mode or switch to real time.
5. Use **Continue same task** with a selected walking checkpoint to continue it. Its saved stage is inherited. Fresh starts or transfers from balance use the chosen starting stage.

Transfer copies the actor, including exploration parameters, to a new run. The original archive is never overwritten. A new reward objective gets a fresh value output head and optimizer; this is fine-tuning, not an exact optimizer continuation across tasks. Existing 55-input checkpoints gain four zero-weight input columns, so their initial action means are preserved. These channels supply target speed and the requested world-forward direction expressed in body coordinates. Balance and legacy playback retain the original 55-value interface; new walking policies use **59 values**. Their actor graph updates its input count accordingly.

### Targets and promotion

The stages request **0.10, 0.20, 0.35 and 0.50 m/s** along world +Z. Later stages also increase initial pose variation. Early walking worlds have a clear flat corridor; objects are moved to the sides. This is not yet a terrain or turning curriculum.

No prescribed gait, leg order, or scripted W-key movement is supplied. The policy controls all 12 joints independently. Walking rewards add speed tracking and progress to the balance reward, penalize motor work, sideways drift, body dragging and foot slipping, and give a small bonus for completed foot swing/contact cycles. Progress credit is gated by upright posture, body height, foot support and low contact-foot slip speed; belly dragging or airborne translation does not receive walking progress credit.

A foot cycle requires a previously grounded foot to leave contact for at least 0.05 seconds, reach centre height of at least 0.09 m, and return to contact. It is a useful measurement, not a proof of a natural walking gait.

A stage promotes only after **three consecutive evaluations** where **every trial**:

- remains upright at least 90% of the time and finishes without falling/leaving bounds;
- averages at least 60% of the requested forward speed;
- keeps mean contact-foot slip speed below `min(0.12 m/s, half the target speed)`;
- has non-foot body contact for no more than 2% of the attempt;
- completes at least two foot cycles involving at least two legs.

These evaluations last 10 seconds on seeds 10001/10002. They guide promotion and are therefore validation trials, not a final independent test. Targets change at the **next episode reset**, never halfway through an attempt. Disable Auto progression to work on a fixed stage. Natural run completion also produces a separate 10-second test on seeds **20001/20002/20003**, not used for promotions, saved as `generalization.json` and shown in the dashboard. Stopped runs skip this final test to save time.

The interface shows target speed, measured forward speed, foot cycles, slip speed and falls. Evaluation reward curves are filtered to one stage, since reward scores across different objectives should not be compared directly.

### Preserve and compare useful walking policies

In addition to initial/latest policies, runs save **Best stage N policy** (best evaluation score observed, including the initial policy) and **Completed stage N policy** when promoted. Best score is forward speed minus slip speed, half fall rate and half body-contact fraction. A "best" checkpoint need not have passed a stage or learned to walk. Checkpoints are published as complete archives, avoiding partial files during playback.

Saved-policy playback understands both old and new observation formats. For fair comparisons between initial/latest checkpoints from a walking run, select the **same explicit Test stage**, seed and duration; their original checkpoint stages may differ. The table now reports speed/target, slipping and foot cycles alongside balance, reward and effort. Playback never trains or modifies the selected policy.

Long training and reward tuning may still be necessary. The implementation tests transfer, updates, curriculum rules and playback; it does not establish that a short run learns successful walking. Keep your standing checkpoint as the reference and judge movement through deterministic playback and the measured evaluations.
