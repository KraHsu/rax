"""Usage projected by native DSH steps, including cached prompt tokens and retries."""
import json
from pathlib import Path

KEYS = ('input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens', 'total_tokens')


class UsageReader:
    def read(self, home):
        home = Path(home)
        steps, complete = {}, True
        path = home/'usage.jsonl'
        if path.exists():
            for line in path.read_text().splitlines(keepends=True):
                if not line.endswith('\n'):
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = (row.get('session_id'), row.get('turn'), row.get('step'))
                steps[key] = row.get('usage')
        totals = dict.fromkeys(KEYS, 0)
        for usage in steps.values():
            if not isinstance(usage, dict):
                complete = False
                continue
            cached, written = usage.get('cacheReadTokens', 0), usage.get('cacheWriteTokens', 0)
            # pi-ai's inputTokens excludes cache hits/writes; its totalTokens includes them.
            inputs = usage['inputTokens'] + cached + written
            outputs = usage['outputTokens']
            for key, value in {'input_tokens': inputs, 'cached_input_tokens': cached,
                'cache_write_input_tokens': written, 'output_tokens': outputs,
                'total_tokens': usage.get('totalTokens', inputs+outputs)}.items():
                totals[key] += value
        audit = home/'request-audit.jsonl'
        calls = None
        if audit.exists():
            with audit.open() as stream:
                calls = sum(line.endswith('\n') for line in stream)
        return {'token_usage': totals if steps else None, 'total_session_usage': totals if steps else None,
                'model_calls': calls, 'agent_steps': len(steps), 'token_usage_complete': complete and bool(steps)}
