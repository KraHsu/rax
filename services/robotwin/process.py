"""Host-side subprocess/RPC owner; no native simulator imports."""
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

from services.robotwin.catalog import configured_manifest, source_root
from services.simulator.io import exit_report, simulator_request


class RoboTwinProcess:
    def __init__(self, *, output_root):
        self.output_root = Path(output_root)
        self.process = self.rpc = self.episode_id = self.output_dir = self.log_path = self.run_id = None
        self.step_id, self.log_mark = 0, 0
        self.finished = True
        self.metadata = None

    @property
    def active(self):
        return self.process is not None and self.rpc is not None and not self.finished

    def start(self, *, task, collection, index, run_id, sim_gpu, startup_timeout=300):
        if self.process is not None:
            raise RuntimeError('Stop the previous simulator process first')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id):
            raise ValueError('Invalid artifact ID')
        python = os.environ.get('ROBOTWIN_PYTHON')
        if not python or not Path(python).is_file():
            raise ValueError('Set ROBOTWIN_PYTHON to an existing simulator environment interpreter')
        manifest = configured_manifest(task)
        manifest.entry(collection, index)
        self.output_dir = (self.output_root / run_id).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.run_id, self.log_path = run_id, self.output_dir / 'sim.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(sim_gpu), PYTHONUNBUFFERED='1',
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]))
        command = [python, '-m', 'services.robotwin.server', '--task', task, '--collection', str(collection),
                   '--index', str(index), '--manifest', str(manifest.path), '--output', str(self.output_dir / 'sim')]
        ready, port = threading.Event(), []
        def read_log(stream):
            with self.log_path.open('w') as log:
                for line in stream:
                    log.write(line); log.flush()
                    try:
                        packet = json.loads(line)
                        if packet.get('event') == 'ready':
                            port.append(packet['port']); ready.set()
                    except (ValueError, AttributeError):
                        pass
        self.process = subprocess.Popen(command, cwd=source_root(), env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=read_log, args=(self.process.stdout,), daemon=True).start()
        started = time.monotonic()
        try:
            while not ready.wait(0.2):
                if self.process.poll() is not None or time.monotonic()-started >= startup_timeout:
                    raise RuntimeError('RoboTwin startup failed: '+exit_report(self.process, self.log_path))
            from services.simulator.protocol import RPCClient
            self.rpc = RPCClient('127.0.0.1', port[0], timeout=startup_timeout)
            self.metadata = simulator_request(self, 'metadata')
            result = simulator_request(self, 'reset')
            self.episode_id, self.step_id, self.finished = result['episode_id'], result['step_id'], False
            self.startup_seconds = time.monotonic()-started
            return result
        except BaseException:
            self.stop()
            raise

    def request(self, op, **kwargs):
        if not self.active:
            raise RuntimeError('No active RoboTwin episode')
        result = simulator_request(self, op, episode_id=self.episode_id, step_id=self.step_id, **kwargs)
        if isinstance(result, dict) and result.get('episode_id') == self.episode_id:
            self.step_id = int(result['step_id'])
        return result

    def finish(self, reason):
        result = self.request('finish_pilot', reason=reason)
        self.finished = True
        return result

    def stop(self):
        if self.rpc is not None:
            with suppress(Exception):
                self.rpc.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
        self.rpc = self.process = None
        self.finished = True
