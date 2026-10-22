import json
import io
from pathlib import Path
import shlex
import socket
import threading
from types import SimpleNamespace

import pytest

from services.robotwin import remote
from services.simulator.protocol import RPCClient, VERSION, receive_packet, send_packet


def config():
    return dict(host='operator@gpu.example', port=20379, rax_root='/work/rax', python='/env/bin/python',
                robotwin_path='/work/RoboTwin', manifest='/work/manifest.json',
                dataset_root='/work/dataset', output_root='/work/runs',
                task_step_limits={'adjust_bottle':1000})


def test_remote_configuration_keeps_catalog_and_gpu_discovery_off_local_host(tmp_path, monkeypatch):
    path = tmp_path/'remote.json'
    path.write_text(json.dumps(config()))
    monkeypatch.setenv('ROBOTWIN_REMOTE_CONFIG', str(path))
    monkeypatch.delenv('ROBOTWIN_PATH', raising=False)
    from services.robotwin.catalog import task_names, step_limit
    assert task_names() == ['adjust_bottle'] and step_limit('adjust_bottle') == 1000
    def check(args, **kwargs):
        assert args[0] == 'ssh' and args[-2] == config()['host']
        assert shlex.split(args[-1])[0] == 'nvidia-smi'
        return '0, GPU-abc\n'
    monkeypatch.setattr(remote.subprocess, 'check_output', check)
    assert remote.gpu_inventory() == '0, GPU-abc\n'


def test_remote_launch_quotes_paths_and_pins_manifest():
    cfg = config()
    cfg['rax_root'] = '/work/a b;$(unexpected)'
    args = remote.launch_command(cfg, task='adjust_bottle', collection=0, index=2,
                                 run_id='abc', sim_gpu='GPU-xyz', manifest_sha256='f'*64)
    decoded = shlex.split(args[-1])
    assert 'PYTHONPATH='+cfg['rax_root'] in decoded
    assert decoded[decoded.index('--manifest-sha256')+1] == 'f'*64
    assert decoded[-1] == '--watch-stdin'
    assert not any('CODEX' in item or 'auth.json' in item for item in args)


def test_remote_host_cannot_be_an_ssh_option(tmp_path, monkeypatch):
    cfg = config(); cfg['host'] = '-oProxyCommand=bad'
    path = tmp_path/'remote.json'; path.write_text(json.dumps(cfg))
    monkeypatch.setenv('ROBOTWIN_REMOTE_CONFIG', str(path))
    with pytest.raises(ValueError, match='host'):
        remote.configuration()


def test_local_codex_launcher_keeps_nvm_node_directory(tmp_path):
    from harness.codex_cli.auto_research import codex_binary
    from harness.agent_host import environment
    install = tmp_path/'nvm/bin'; install.mkdir(parents=True)
    script = tmp_path/'package/codex.js'; script.parent.mkdir(); script.write_text('#!/usr/bin/env node\n')
    binary = install/'codex'; binary.symlink_to(script)
    selected = codex_binary(binary)
    assert selected == binary
    env = environment(home=tmp_path, workspace=tmp_path, socket=tmp_path/'mcp.sock', cpus=1, binary=selected)
    assert str(install) in env['PATH'].split(':')


@pytest.mark.parametrize('timeout', [0, 1801, True, '900'])
def test_startup_timeout_is_bounded_operator_configuration(tmp_path, monkeypatch, timeout):
    path=tmp_path/'remote.json'; path.write_text(json.dumps(dict(config(), startup_timeout_seconds=timeout)))
    monkeypatch.setenv('ROBOTWIN_REMOTE_CONFIG',str(path))
    with pytest.raises(ValueError, match='startup_timeout_seconds'):
        remote.configuration()


def test_operator_startup_limit_reaches_supervisor_and_task_metadata(tmp_path, monkeypatch):
    from services.benchmarks import BENCHMARKS
    path = tmp_path/'remote.json'
    path.write_text(json.dumps(dict(config(), startup_timeout_seconds=600)))
    monkeypatch.setenv('ROBOTWIN_REMOTE_CONFIG', str(path))
    bench = BENCHMARKS['robotwin']
    assert bench.startup_limit('adjust_bottle') == 600
    assert bench.task_timeouts('adjust_bottle')['startup_seconds'] == 600
    for name in ('robodojo', 'behavior'):
        assert BENCHMARKS[name].startup_limit('unused') == BENCHMARKS[name].startup_seconds
    monkeypatch.delenv('ROBOTWIN_REMOTE_CONFIG')
    assert bench.startup_limit('adjust_bottle') == 300


def test_rpc_over_forwarded_unix_socket_preserves_arrays_and_identity(tmp_path):
    import numpy as np
    path = tmp_path/'sim.sock'
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path)); listener.listen(1)
    failures = []
    def serve():
        try:
            with listener.accept()[0] as connection:
                packet = receive_packet(connection)
                assert packet['op'] == 'observe'
                send_packet(connection, dict(version=VERSION, request_id=packet['request_id'], ok=True,
                    result={'image':np.full((3,4,3), 19, dtype=np.uint8)}))
        except BaseException as exc:
            failures.append(exc)
    thread = threading.Thread(target=serve); thread.start()
    client = RPCClient(None, None, unix_path=path)
    try:
        assert (client.request('observe')['image'] == 19).all()
    finally:
        client.close(); thread.join(3); listener.close()
    assert not thread.is_alive() and not failures


def test_ssh_process_keeps_raw_rpc_private_and_mirrors_operator_verdict(tmp_path, monkeypatch):
    from harness import agent_host
    from services.robotwin import process
    from services.simulator import protocol
    monkeypatch.setattr(agent_host, 'SOCKETS', tmp_path/'private-sockets')
    monkeypatch.setattr(remote, 'configuration', lambda:dict(config(), startup_timeout_seconds=900))
    monkeypatch.setattr(process, 'configured_manifest', lambda task:SimpleNamespace(
        path=tmp_path/'manifest.json', sha256='a'*64, entry=lambda *_:{}))
    children = []
    class Child:
        def __init__(self, args, **kwargs):
            self.stdin, self.stdout = io.StringIO(), io.StringIO('{"event":"ready","port":12345}\n')
            self.pid = 999999
            self.args = args
            if '-L' in args:
                Path(args[args.index('-L')+1].split(':')[0]).touch()
            children.append(self)
        def poll(self): return None
        def wait(self, **kwargs): return 0
        def terminate(self): pass
    monkeypatch.setattr(process.subprocess, 'Popen', Child)
    class Client:
        def __init__(self, *_, unix_path, **kwargs):
            assert kwargs['timeout'] == 900
            assert unix_path.parent.parent == agent_host.SOCKETS
            exposed = agent_host.socket_path(tmp_path/'sessions/run').parent
            assert not unix_path.is_relative_to(exposed)
            assert any(unix_path.is_relative_to(p) for p in agent_host.hidden_paths(tmp_path/'sessions/run'))
        def request(self, op, **kwargs):
            if op == 'metadata': return {'native_step_limit':400}
            result = {'episode_id':'episode', 'step_id':0}
            if op == 'operator_state': result['reward'] = {'native_success':False}
            return result
        def close(self): pass
        def set_timeout(self, value): assert value == 300
    monkeypatch.setattr(protocol, 'RPCClient', Client)
    sim = process.RoboTwinProcess(output_root=tmp_path/'results')
    try:
        assert sim.start(task='adjust_bottle',collection=0,index=0,run_id='smoke',sim_gpu='GPU-abc')['step_id'] == 0
        assert sim.active
        sim.finish('smoke')
        assert not sim.active
        state = json.loads((sim.output_dir/'sim/operator_state.json').read_text())
        assert state['reward']['native_success'] is False
        assert len(children) == 2
    finally:
        sim.stop()
    assert sim.tunnel is None and sim.process is None
