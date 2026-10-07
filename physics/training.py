"""One shared PPO run, streamed to viewers. Heavy learning runs off the server loop."""
import importlib.util
import multiprocessing as mp
import queue
import json
import time
import uuid
from pathlib import Path


def validate_start(message):
    if not isinstance(message, dict):
        raise ValueError('Training command must be an object')
    worlds, steps, seed = (message.get(k, default) for k, default in [('worlds', 1), ('steps', 30000), ('seed', 1)])
    for value, low, high, name in [(worlds, 1, 8, 'worlds'), (steps, 128, 1000000, 'steps'), (seed, 0, 999999, 'seed')]:
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer between {low} and {high}')
    task, speed, resume = message.get('task', 'balance'), message.get('speed', 'realtime'), message.get('resume', False)
    if task not in ('balance', 'walk') or speed not in ('realtime', 'fast') or not isinstance(resume, bool):
        raise ValueError('Invalid task, speed or resume option')
    return dict(worlds=worlds, steps=steps, seed=seed, task=task, speed=speed, resume=resume)


class TrainingManager:
    def __init__(self, root=Path('runs')):
        self.root = Path(root)
        self.context = mp.get_context('spawn')
        self.process = None
        self.latest = {'status': 'idle'}
        self.last_checkpoint = None
        self.last_task = None
        candidates = sorted(self.root.glob('*/policy.zip'), key=lambda path: path.stat().st_mtime, reverse=True)
        for candidate in candidates:
            try:
                task = json.loads(candidate.with_name('config.json').read_text())['task']
                if task in ('balance', 'walk'):
                    self.last_checkpoint, self.last_task = str(candidate), task
                    break
            except (OSError, ValueError, KeyError):
                continue

    def poll(self):
        if self.process is not None:
            while True:
                try:
                    self.latest = self.output.get_nowait()
                    if self.latest.get('checkpoint'):
                        self.last_checkpoint = self.latest['checkpoint']
                except queue.Empty:
                    break
            if not self.process.is_alive() and self.latest['status'] not in ('finished', 'stopped', 'error'):
                # A final queue message can take one polling interval to arrive.
                if getattr(self, '_dead_seen', False):
                    self.latest = {**self.latest, 'status': 'error', 'error': f'Training process exited ({self.process.exitcode})'}
                self._dead_seen = True
        return self.latest

    def command(self, message):
        if not isinstance(message, dict):
            raise ValueError('Training command must be an object')
        kind = message.get('type')
        if kind == 'start':
            config = validate_start(message)
            self.poll()
            if self.process is not None and self.latest['status'] in ('finished', 'stopped', 'error'):
                self.process.join(timeout=2)
            if self.process is not None and self.process.is_alive():
                raise ValueError('A shared training run is already active')
            if importlib.util.find_spec('stable_baselines3') is None:
                raise ValueError('Install requirements-training.txt before starting training')
            checkpoint = self.last_checkpoint if config['resume'] else None
            if config['resume'] and (not checkpoint or not Path(checkpoint).is_file() or self.last_task != config['task']):
                raise ValueError('Resume requires a saved checkpoint from the same task')
            if self.process is not None:
                self.process.join(timeout=1)
                self.output.close()
            run_dir = self.root / uuid.uuid4().hex[:12]
            run_dir.mkdir(parents=True, exist_ok=False)
            self.output = self.context.Queue(maxsize=2)
            self.paused, self.stop = self.context.Event(), self.context.Event()
            self.fast = self.context.Value('b', config['speed'] == 'fast')
            self.last_task = config['task']
            self._dead_seen = False
            self.latest = dict(status='starting', config=config, run_id=run_dir.name)
            self.process = self.context.Process(target=training_worker,
                args=(config, str(run_dir), checkpoint, self.output, self.paused, self.stop, self.fast))
            self.process.start()
        elif kind in ('pause', 'resume', 'stop', 'speed'):
            if self.process is None or not self.process.is_alive():
                raise ValueError('No active training run')
            if kind == 'pause':
                self.paused.set()
            elif kind == 'resume':
                self.paused.clear()
            elif kind == 'stop':
                self.stop.set()
                self.paused.clear()
            elif message.get('speed') in ('realtime', 'fast'):
                self.fast.value = message['speed'] == 'fast'
            else:
                raise ValueError('Invalid training speed')
        else:
            raise ValueError('Unknown training command')

    def close(self):
        if self.process is not None:
            self.stop.set()
            self.paused.clear()
            deadline = time.monotonic()+10
            while self.process.is_alive() and time.monotonic() < deadline:
                self.poll()  # Drain telemetry so the worker can flush its queue and exit.
                self.process.join(timeout=.1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
            self.output.close()


def training_worker(config, directory, checkpoint, output, paused, stop, fast):
    # Imported only in the worker so demonstration mode remains lightweight.
    try:
        from .learner import run_training
        run_training(config, directory, checkpoint, output, paused, stop, fast)
    except Exception as error:
        import traceback
        traceback.print_exc()
        payload = {'status': 'error', 'error': f'{type(error).__name__}: {error}', 'run_id': Path(directory).name}
        try:
            output.put(payload, timeout=2)
        except queue.Full:
            try:
                output.get_nowait()
            except queue.Empty:
                pass
            output.put(payload, timeout=2)
