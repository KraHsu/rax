"""Provider-reported usage from the native CLI transport; missing usage stays unknown."""
import json
from pathlib import Path


def rows(path):
    if not path.is_file():
        return []
    result = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith('\n'):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


class UsageReader:
    def read(self, home):
        home = Path(home)
        requests = rows(home/'request-audit.jsonl')
        responses = {r['request_id']: r for r in rows(home/'response-usage.jsonl') if r.get('request_id')}
        complete = bool(requests)
        totals = dict.fromkeys(('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens',
                               'output_tokens', 'reasoning_tokens', 'total_tokens'), 0)
        measured = 0
        for request in requests:
            response = responses.get(request.get('request_id'), {})
            usage = response.get('usage') or {}
            if not all(isinstance(usage.get(k), int) for k in ('prompt_tokens', 'completion_tokens')):
                complete = False
                continue
            measured += 1
            inputs, outputs = usage['prompt_tokens'], usage['completion_tokens']
            details = usage.get('prompt_tokens_details') or {}
            totals['input_tokens'] += inputs  # OpenAI prompt_tokens already includes cache hits.
            totals['cached_input_tokens'] += details.get('cached_tokens', 0)
            totals['cache_write_input_tokens'] += details.get('cache_write_tokens', 0)
            totals['output_tokens'] += outputs
            totals['reasoning_tokens'] += (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
            totals['total_tokens'] += usage.get('total_tokens', inputs+outputs)
        return {'token_usage': totals if measured else None, 'total_session_usage': totals if measured else None,
                'model_calls': len(requests), 'measured_model_calls': measured,
                'agent_steps': measured, 'token_usage_complete': complete}
