"""Only validated model/transport pairs may reach GPU submission."""
from pathlib import Path
import json

import pytest

from harness.backends import PROVIDERS
from scripts.submit_native_cli_experiment import SETTINGS, configure_command, validate_report
from services.native_cli_usage import UsageReader

BACKEND = next(iter(PROVIDERS))


def report():
    return {'status': 'passed', 'backend': BACKEND, 'provider': 'openrouter', 'model': 'pinned-model',
        'effort': 'max', 'vision_wire': True, 'wire_max_effort': True, 'resume': True,
        'native_file': True, 'native_shell': True, 'tool_call': True, 'tool_result': True,
        'boundary': {'offline_tools': True, 'hidden_paths': True}}


def test_matching_validation_is_required():
    validate_report(report(), BACKEND, 'pinned-model')
    for update in ({'model': 'different'}, {'provider': 'custom'}, {'status': 'failed'},
                   {'vision_wire': False}, {'resume': False}, {'effort': 'low'},
                   {'boundary': {'offline_tools': False}}):
        with pytest.raises(ValueError, match='matching native CLI/model'):
            validate_report({**report(), **update}, BACKEND, 'pinned-model')


def test_comparison_configuration_matches_deepseek():
    record = {**SETTINGS, 'agent_cli': BACKEND}
    command = configure_command(record, 'GPU-aaaa')
    def option(name):
        return command[command.index('--'+name)+1]
    assert option('episodes') == '16' and option('formal-episodes') == '150'
    assert option('formal-workers') == '4' and option('agent-effort') == 'max'
    assert option('exploration-envs') == '1' and option('subagents') == '0'
    assert option('external-perception') == 'off' and option('episode-reviews') == 'off'


def test_usage_does_not_double_count_cached_tokens_or_hide_missing_responses(tmp_path):
    (tmp_path/'request-audit.jsonl').write_text(''.join(json.dumps({'request_id': x})+'\n' for x in ('a', 'b')))
    usage = {'prompt_tokens': 100, 'completion_tokens': 30, 'total_tokens': 130,
             'prompt_tokens_details': {'cached_tokens': 50}, 'completion_tokens_details': {'reasoning_tokens': 20}}
    response = {'request_id': 'a', 'status': 200, 'usage': usage}
    (tmp_path/'response-usage.jsonl').write_text(json.dumps(response)+'\n'+json.dumps(response)+'\n')
    result = UsageReader().read(tmp_path)
    assert result['model_calls'] == 2 and result['measured_model_calls'] == 1
    assert result['token_usage']['input_tokens'] == 100
    assert result['token_usage']['cached_input_tokens'] == 50
    assert result['token_usage']['reasoning_tokens'] == 20
    assert result['token_usage']['total_tokens'] == 130
    assert result['token_usage_complete'] is False
    with (tmp_path/'response-usage.jsonl').open('a') as stream:
        stream.write(json.dumps({**response, 'request_id': 'b'})+'\n')
        stream.write('{"truncated":')
    assert UsageReader().read(tmp_path)['token_usage_complete'] is True
    assert UsageReader().read(tmp_path)['token_usage']['total_tokens'] == 260


def test_missing_token_counts_stay_unknown(tmp_path):
    assert UsageReader().read(tmp_path)['token_usage'] is None
    assert UsageReader().read(tmp_path)['token_usage_complete'] is False
