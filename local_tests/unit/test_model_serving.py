"""Serving configuration and a real HTTP protocol round trip without a model/GPU."""
import errno
import importlib.util
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from harness.claude_cli.login import Credential
from harness.model_serving.configuration import ServeConfig
from harness.model_serving.probes import MessagesClient, smoke
from harness.model_serving.slurm import submit_pipeline
from services.mcp_contract import Contract

PROJECT = Path(__file__).resolve().parents[2]


def test_local_only_contract_omits_external_perception_in_every_phase():
    from types import SimpleNamespace
    config = SimpleNamespace(external_perception=False)
    for isolated in (False, True):
        contract = Contract.from_config(config, isolated=isolated)
        assert 'gemini_generate' not in contract.tool_names
        assert contract.manifest()['gemini'] is False
        with pytest.raises(PermissionError):
            contract.require_robot('gemini_generate', {'messages': []})


def test_external_perception_cannot_change_when_a_session_resumes(tmp_path):
    from test_controller_backend import Backend, Runner
    from test_controller_backend import configuration as session_configuration

    from services.controller.supervisor import Supervisor
    config = replace(session_configuration(tmp_path), external_perception=False)
    Supervisor(config, lambda: Backend(), runner_factory=Runner).close()
    with pytest.raises(ValueError, match='saved configuration cannot change'):
        Supervisor(replace(config, external_perception=True), lambda: Backend(), runner_factory=Runner)


def test_shared_filesystem_retry_is_bounded_and_does_not_hide_permission_errors(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('services.workspace.build')
    path = tmp_path/'skill'
    path.mkdir()
    original = module.shutil.rmtree
    attempts = []
    def transient(target):
        attempts.append(target)
        if len(attempts) == 1:
            raise OSError(errno.ENOTEMPTY, 'NFS metadata lag')
        original(target)
    monkeypatch.setattr(module.shutil, 'rmtree', transient)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    module.remove_skill_tree(path)
    assert not path.exists() and len(attempts) == 2
    def denied(_):
        raise PermissionError(errno.EACCES, 'permission denied')
    monkeypatch.setattr(module.shutil, 'rmtree', denied)
    with pytest.raises(PermissionError):
        module.remove_skill_tree(path)


def configuration():
    return ServeConfig.load(PROJECT/'configs/serving/deepseek-v41-flash-1017.json')


def test_model_and_container_are_immutable_and_reasoning_is_maximal():
    config = configuration()
    command = config.arguments('/weights')
    assert command[:2] == ['--model', '/weights']
    assert command.count('--model') == 1
    assert json.loads(command[command.index('--default-chat-template-kwargs')+1]) == {
        'thinking': True, 'reasoning_effort': 100}
    assert '--language-model-only' not in command
    with pytest.raises(ValueError, match='commit'):
        replace(config, revision='main')
    with pytest.raises(ValueError, match='digest'):
        replace(config, image='vllm/vllm-openai:latest')


def test_custom_endpoint_authentication_and_model_roles(tmp_path):
    key = tmp_path/'token'
    key.write_text('local-service-token')
    key.chmod(0o600)
    credential = Credential('custom', key_file=key, base_url='http://compute:8000/')
    env = credential.environment(model='deepseek-flash')
    assert env['ANTHROPIC_BASE_URL'] == 'http://compute:8000'
    assert env['ANTHROPIC_AUTH_TOKEN'] == 'local-service-token'
    assert env['ANTHROPIC_DEFAULT_HAIKU_MODEL'] == 'deepseek-flash'
    with pytest.raises(ValueError, match='base-url'):
        Credential('custom', key_file=key)
    with pytest.raises(ValueError, match='embedded'):
        Credential('custom', key_file=key, base_url='http://user:pass@compute')


def test_tuning_options_keep_real_verification_and_maximum_reasoning():
    config = replace(configuration(), enforce_eager=False,
                     compilation_config={'cudagraph_mode':'FULL_AND_PIECEWISE'},
                     speculative_config={'method':'dspark','num_speculative_tokens':5,
                                         'rejection_sample_method':'block'})
    command = config.arguments('/weights')
    assert '--enforce-eager' not in command
    assert json.loads(command[command.index('--speculative-config')+1])['method'] == 'dspark'
    assert json.loads(command[command.index('--default-chat-template-kwargs')+1])['reasoning_effort'] == 100
    with pytest.raises(ValueError, match='Synthetic'):
        replace(config,speculative_config={'rejection_sample_method':'synthetic'})
    with pytest.raises(ValueError, match='enforce_eager'):
        replace(config,enforce_eager=True)


def test_stream_benchmark_uses_token_usage_with_batched_speculative_chunks():
    import io
    from harness.model_serving.benchmark import measure
    class Client:
        def request(self, path, payload):
            assert payload['thinking']['type'] == 'adaptive'
            assert payload['output_config']['effort'] == 'max'
            events = [
                {'type':'message_start','message':{'usage':{'input_tokens':25,'output_tokens':0}}},
                {'type':'content_block_start','content_block':{'type':'thinking'}},
                {'type':'content_block_delta','delta':{'thinking':'several tokens in one chunk'}},
                {'type':'message_delta','usage':{'output_tokens':7},'delta':{'stop_reason':'end_turn'}},
                {'type':'message_stop'}]
            return io.BytesIO(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    result = measure(Client(),{'thinking':{'type':'adaptive'},'output_config':{'effort':'max'}},concurrency=2)
    assert result['total_output_tokens'] == 14
    assert all(row['output_tokens']==7 for row in result['requests'])


def tuning_results():
    from dataclasses import asdict
    def variant(name, single, aggregate):
        cases = []
        for workload in ('short_code','long_code'):
            for repeat in (0,1):
                cases.append({'workload':workload,'repeat':repeat,'concurrency':1,
                    'payload_sha256':workload,'aggregate_end_to_end_tps':single,
                    'requests':[{'ttft_seconds':0.1,'post_first_token_tps_approx':single}]})
        cases.append({'workload':'batch_code','concurrency':8,'payload_sha256':'batch_code',
            'aggregate_end_to_end_tps':aggregate,
            'requests':[{'ttft_seconds':0.1,'post_first_token_tps_approx':aggregate/8}]*8})
        return {'name':name,'status':'passed','config':asdict(configuration()),'cases':cases,
                'smoke':dict.fromkeys(('tool_call','tool_result','streaming','thinking_blocks','vision'),True)}
    return {'variants':[variant('baseline',14,100),variant('fast-single',100,500),
                        variant('large-batch',13,2000)]}


def test_tuning_selection_prioritizes_single_request_speed():
    from harness.model_serving.benchmark import summarize
    report = summarize(tuning_results())
    assert report['winner'] == 'fast-single'
    assert report['single_request_speedup'] == pytest.approx(100/14)


def test_tuning_comparison_rejects_different_payloads_or_models():
    from harness.model_serving.benchmark import summarize
    results = tuning_results()
    results['variants'][1]['cases'][0]['payload_sha256'] = 'changed'
    with pytest.raises(ValueError,match='payloads differ'):
        summarize(results)
    results = tuning_results()
    results['variants'][1]['config']['revision'] = 'f'*40
    with pytest.raises(ValueError,match='changed model'):
        summarize(results)


def test_smoke_verifies_tools_roundtrip_vision_and_complete_thinking_stream(tmp_path):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            assert self.headers['Authorization'] == 'Bearer local-service-token'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            if body.get('stream'):
                events = [
                    {'type': 'content_block_start', 'content_block': {'type': 'thinking', 'thinking': ''}},
                    {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': '323'}},
                    {'type': 'message_stop'}]
                data = (''.join('data: '+json.dumps(event)+'\n\n' for event in events)
                        + 'data: [DONE]\n\n').encode()
            elif len(seen) == 1:
                data = json.dumps({'content': [{'type': 'tool_use', 'id': 't1', 'name': 'report_number', 'input': {'value': 7}}]}).encode()
            else:
                data = json.dumps({'content': [{'type': 'text', 'text': 'DONE: a robot and a mouse.'}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    key = tmp_path/'key'
    key.write_text('local-service-token')
    key.chmod(0o600)
    image = tmp_path/'robot.jpg'
    image.write_bytes(b'fixture-image')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = smoke(MessagesClient(f'http://127.0.0.1:{server.server_port}', key), 'deepseek-flash', image=image)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert all(result[name] for name in ('tool_call', 'tool_result', 'vision', 'streaming', 'thinking_blocks'))
    assert all(body['output_config']['effort'] == 'max' for body in seen)
    assert all(body['thinking']['type'] == 'adaptive' for body in seen)


def test_render_pipeline_separates_cpu_preparation_from_gpu_serving(tmp_path):
    profile = json.loads((PROJECT/'configs/serving/slurm.example.json').read_text())
    profile['prepare']['account'] = 'cpu_account'
    site = tmp_path/'site.json'
    site.write_text(json.dumps(profile))
    root = tmp_path/'model service'
    result = submit_pipeline(project=PROJECT, root=root, config=PROJECT/'configs/serving/deepseek-v41-flash-1017.json',
        profile=site, key_file=tmp_path/'private key', openrouter_key_file=tmp_path/'api key', image=tmp_path/'robot.jpg')
    assert result['job_ids'] == {}
    prepare = (root/'jobs/prepare.sbatch').read_text()
    serve = (root/'jobs/serve.sbatch').read_text()
    assert '#SBATCH --account=cpu_account' in prepare and '--gres=' not in prepare
    assert '#SBATCH --gres=gpu:nvidia_h200:4' in serve
    assert 'secrets' not in serve  # Only caller-supplied paths, no secret values.


@pytest.mark.parametrize('tuned', [False, True])
def test_failed_gpu_preflight_stops_before_loading_the_model(tmp_path, monkeypatch, tuned):
    from dataclasses import asdict

    from harness.model_serving.lifecycle import serve, write_json
    config = configuration()
    executable = tmp_path/'singularity'
    executable.write_text('#!/bin/sh\nexit 17\n')
    executable.chmod(0o700)
    monkeypatch.setenv('PATH', str(tmp_path))
    key = tmp_path/'key'
    key.write_text('local-service-token')
    key.chmod(0o600)
    write_json(tmp_path/'prepared.json', {'config': asdict(config), 'weights': '/weights', 'image': '/image'})
    if tuned:
        config = replace(config, enforce_eager=False, max_num_seqs=config.max_num_seqs*2)
    with pytest.raises(RuntimeError, match='GPU preflight failed: 17'):
        serve(config, tmp_path, key)
    endpoint = json.loads((tmp_path/'endpoint.json').read_text())
    assert endpoint['status'] == 'stopped' and endpoint['returncode'] == 17


def test_runtime_tuning_does_not_allow_reusing_different_model_artifacts(tmp_path):
    from dataclasses import asdict
    from harness.model_serving.lifecycle import serve, write_json
    config = configuration()
    write_json(tmp_path/'prepared.json', {'config':asdict(config),'weights':'/weights','image':'/image'})
    with pytest.raises(RuntimeError, match='requested model or container'):
        serve(replace(config,revision='0'*40),tmp_path,tmp_path/'unused-key')


def test_single_visible_gpu_overrides_slurm_physical_ordinal(monkeypatch):
    path = PROJECT/'scripts/run_deepseek_experiment.py'
    spec = importlib.util.spec_from_file_location('deepseek_experiment', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('SLURM_JOB_GPUS', '3')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: '0, GPU-allocated\n')
    assert module.allocated_gpu() == 'GPU-allocated'
