"""Model artifacts and process lifecycle, independent of Slurm and DexJoCo."""
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .probes import MessagesClient


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare_image(config, root):
    """The image can be prepared and inspected independently of the large checkpoint."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    image = root/'vllm.sif'
    with (root/'image.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not image.is_file():
            subprocess.run(['singularity', 'pull', '--disable-cache', str(image), 'docker://'+config.image], check=True)
        subprocess.run(['singularity', 'inspect', str(image)], check=True, stdout=subprocess.DEVNULL)
    return image


def prepare(config, root):
    """Download the pinned checkpoint and convert its pinned container to a portable SIF."""
    from huggingface_hub import snapshot_download
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    weights = root/'weights'
    snapshot_download(repo_id=config.model, revision=config.revision, local_dir=weights,
                      allow_patterns=['*.json', '*.safetensors', '*.py', '*.txt', '*.model', '*.tiktoken', '*.jinja'],
                      max_workers=8)
    image = prepare_image(config, root)
    # A completed manifest is the only signal that both preparations succeeded.
    write_json(root/'prepared.json', {'config': asdict(config), 'weights': str(weights),
                                    'image': str(image), 'finished_at': time.time()})


def serve(config, root, key_file, *, ready_timeout=7200):
    from harness.claude_cli.login import private_text
    root = Path(root).resolve()
    prepared = json.loads((root/'prepared.json').read_text())
    # Runtime scheduling/graph settings do not change the downloaded artifacts.
    # Keep immutable model and container identity checks when reusing them.
    if any(prepared['config'].get(name) != getattr(config, name) for name in ('model', 'revision', 'image')):
        raise RuntimeError('Prepared artifacts do not match the requested model or container')
    env = dict(os.environ)
    env['SINGULARITYENV_VLLM_API_KEY'] = private_text(key_file).strip()
    env['SINGULARITYENV_VLLM_ENGINE_READY_TIMEOUT_S'] = str(ready_timeout)
    env['SINGULARITYENV_HF_HUB_OFFLINE'] = '1'
    cache = (root/'cache').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    env['SINGULARITYENV_HF_HOME'] = str(cache/'huggingface')
    env['SINGULARITYENV_XDG_CACHE_HOME'] = str(cache)
    # Host preprocessing may use several threads, while vLLM's token-by-token
    # dispatch benefits from its usual single-thread CPU execution policy.
    env.setdefault('SINGULARITYENV_OMP_NUM_THREADS', str(config.cpu_threads))
    env.setdefault('SINGULARITYENV_OPENBLAS_NUM_THREADS', str(config.cpu_threads))
    if 'CUDA_VISIBLE_DEVICES' in env:
        env['SINGULARITYENV_CUDA_VISIBLE_DEVICES'] = env['CUDA_VISIBLE_DEVICES']
    container = ['singularity', 'exec', '--nv', '--bind', f'{root}:{root}',
                 '--bind', f"{prepared['weights']}:{prepared['weights']}:ro",
                 '--bind', f'{cache}:{cache}',
                 '--bind', f'{Path(__file__).resolve().parent}:/rax-serving:ro', prepared['image']]
    preflight = [*container, 'python3', '-m', 'torch.distributed.run', '--standalone',
                 f'--nproc-per-node={config.tensor_parallel_size}', '/rax-serving/gpu_probe.py']
    command = [*container, 'python3', '-m', 'vllm.entrypoints.openai.api_server',
               *config.arguments(prepared['weights'])]
    endpoint = root/'endpoint.json'
    record = {'status': 'starting', 'base_url': f'http://{socket.getfqdn()}:{config.port}',
              'model': config.served_model, 'model_id': config.model, 'revision': config.revision,
              'reasoning_effort': config.reasoning_effort, 'started_at': time.time(),
              'runtime_env': {name: env['SINGULARITYENV_'+name] for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    write_json(endpoint, record)
    print(json.dumps({'command': command, 'endpoint': record}), flush=True)
    child = None
    def stop(number, _):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, number)
    previous = {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, stop)
    try:
        # Validate the complete tensor-parallel allocation before loading hundreds
        # of GB. Reuse this allocation rather than queueing for the same GPUs twice.
        child = subprocess.Popen(preflight, env=env, start_new_session=True)
        code = child.wait(timeout=600)
        if code:
            raise RuntimeError(f'GPU preflight failed: {code}')
        child = subprocess.Popen(command, env=env, start_new_session=True)
        client = MessagesClient(f'http://127.0.0.1:{config.port}', key_file, timeout=5)
        deadline = time.monotonic()+ready_timeout
        while child.poll() is None:
            try:
                models = client.json('/v1/models')
                if config.served_model not in [m['id'] for m in models.get('data', [])]:
                    raise RuntimeError('Unexpected served model')
                break
            except (OSError, RuntimeError):
                if time.monotonic() > deadline:
                    raise TimeoutError('Model service readiness timed out')
                time.sleep(10)
        if child.poll() is not None:
            raise RuntimeError(f'Model server exited before becoming ready: {child.returncode}')
        record.update(status='ready', ready_at=time.time())
        write_json(endpoint, record)
        code = child.wait()
        if code:
            raise RuntimeError(f'Model server exited: {code}')
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        record.update(status='stopped', stopped_at=time.time(),
                      returncode=child.returncode if child is not None else None)
        write_json(endpoint, record)
        for number, handler in previous.items():
            signal.signal(number, handler)
