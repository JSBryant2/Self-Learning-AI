"""Run baseline episodes and log measurements; this does not train a policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from .environment import BalanceEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['zero', 'random'], default='random')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('runs/balance.jsonl'))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    env = BalanceEnv(max_seconds=args.seconds)
    rng = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation preserves prior measurements rather than overwriting them.
        with args.output.open('x') as log:
            for episode in range(args.episodes):
                env.reset(seed=args.seed+episode)
                done = False
                while not done:
                    action = np.zeros(12, np.float32) if args.policy == 'zero' else rng.uniform(-1, 1, 12).astype(np.float32)
                    _, _, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                result = dict(policy=args.policy, seed=args.seed+episode, **info['episode'])
                log.write(json.dumps(result, allow_nan=False)+'\n')
                log.flush()
                print(json.dumps(result, allow_nan=False))
    finally:
        env.close()


if __name__ == '__main__':
    main()
