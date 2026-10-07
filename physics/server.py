"""Each viewer connection owns an independent simulation; localhost only."""
import asyncio
import json
import math
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from .simulation import Simulation


def validate(message):
    if not isinstance(message, dict):
        raise ValueError('Command must be an object')
    kind = message.get('type')
    if kind == 'reset':
        seed = message.get('seed', 1)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= seed <= 999999:
            raise ValueError('Seed must be an integer from 1 to 999999')
        return kind, seed
    if kind == 'control':
        mode = message.get('mode', 'idle')
        if mode not in ('idle', 'explore', 'manual') or not isinstance(message.get('paused', False), bool):
            raise ValueError('Invalid controller or pause flag')
        values = [message.get(k, 0) for k in ('drive', 'turn')]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) <= 1 for v in values):
            raise ValueError('Drive and turn must be finite numbers in [-1, 1]')
        return kind, dict(mode=mode, drive=values[0], turn=values[1], paused=message.get('paused', False))
    raise ValueError('Unknown command')


async def handle(socket):
    sim = Simulation()
    control = dict(mode='idle', drive=0, turn=0, paused=False)

    async def receive():
        nonlocal control
        async for raw in socket:
            try:
                kind, value = validate(json.loads(raw))
                if kind == 'reset':
                    sim.reset(value)
                else:
                    control = value
            except (ValueError, TypeError):
                await socket.send(json.dumps({'error': 'Invalid simulation command'}))

    reader = asyncio.create_task(receive())
    try:
        while not reader.done():
            start = asyncio.get_running_loop().time()
            if not control['paused']:
                sim.step(control['mode'], control['drive'], control['turn'])
            await socket.send(json.dumps(sim.snapshot(), allow_nan=False))
            await asyncio.sleep(max(0, sim.dt-(asyncio.get_running_loop().time()-start)))
    except ConnectionClosed:
        pass
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        sim.close()


async def main():
    async with serve(handle, '127.0.0.1', 8765, max_size=4096):
        print('Python physics listening on 127.0.0.1:8765', flush=True)
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())
