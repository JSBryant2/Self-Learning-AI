"""Local demo worlds and a shared, independently running training session."""
import asyncio
import json
import math
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from .simulation import Simulation
from .training import TrainingManager
from .playback import PlaybackSession, saved_policies


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


async def handle_training(socket, manager):
    async def receive():
        async for raw in socket:
            try:
                manager.command(json.loads(raw))
            except (ValueError, TypeError, OSError) as error:
                await socket.send(json.dumps({'commandError': str(error)}))
    reader = asyncio.create_task(receive())
    try:
        while not reader.done():
            await socket.send(json.dumps(manager.poll(), allow_nan=False))
            await asyncio.sleep(.1)
    except ConnectionClosed:
        pass
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def handle_playback(socket):
    session = PlaybackSession()
    lock = asyncio.Lock()

    async def receive():
        async for raw in socket:
            try:
                message = json.loads(raw)
                if isinstance(message, dict) and message.get('type') == 'list':
                    await socket.send(json.dumps({'policies': saved_policies()}))
                    continue
                async with lock:
                    # Loading PyTorch/checkpoints must not block the training socket.
                    if isinstance(message, dict) and message.get('type') == 'load':
                        await socket.send(json.dumps({'status': 'loading'}))
                        await asyncio.to_thread(session.command, message)
                    else:
                        session.command(message)
            except Exception as error:
                await socket.send(json.dumps({'commandError': str(error)}))
    reader = asyncio.create_task(receive())
    try:
        await socket.send(json.dumps({'policies': saved_policies()}))
        while not reader.done():
            start = asyncio.get_running_loop().time()
            async with lock:
                state = session.step()
            await socket.send(json.dumps(state, allow_nan=False))
            await asyncio.sleep(max(0, 1/30-(asyncio.get_running_loop().time()-start)))
    except ConnectionClosed:
        pass
    finally:
        # Let a pending checkpoint load finish before freeing its simulation.
        if not reader.done():
            try:
                await asyncio.wait_for(asyncio.shield(reader), timeout=.05)
            except asyncio.TimeoutError:
                async with lock:
                    reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        session.close()


async def main():
    manager = TrainingManager()

    async def route(socket):
        if socket.request.path == '/playback':
            await handle_playback(socket)
        elif socket.request.path == '/training':
            await handle_training(socket, manager)
        else:
            await handle(socket)

    try:
        async with serve(route, '127.0.0.1', 8765, max_size=4096):
            print('Python physics and training listening on 127.0.0.1:8765', flush=True)
            await asyncio.Future()
    finally:
        manager.close()


if __name__ == '__main__':
    asyncio.run(main())
