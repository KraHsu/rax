"""Live Anthropic-protocol checks, including tools, vision, reasoning and streaming."""
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class MessagesClient:
    def __init__(self, base_url, key_file, *, timeout=300):
        self.base_url = base_url.rstrip('/')
        self.key_file = Path(key_file)
        self.timeout = timeout

    def request(self, path, payload=None):
        from harness.claude_cli.login import private_text
        key = private_text(self.key_file).strip()
        request = urllib.request.Request(self.base_url+path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json',
                     'anthropic-version': '2023-06-01'})
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read(4096).decode(errors='replace').replace(key, '[REDACTED]')
            raise RuntimeError(f'{path}: HTTP {error.code}: {detail}') from None

    def json(self, path, payload=None):
        with self.request(path, payload) as response:
            return json.load(response)


def smoke(client, model, *, image=None):
    started = time.time()
    base = {'model': model, 'max_tokens': 4096,
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': 'max'}}
    report = {'model': model, 'effort': 'max', 'started_at': started}
    tools = [{'name': 'report_number', 'description': 'Report the requested integer.',
              'input_schema': {'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']}}]
    first = client.json('/v1/messages', {**base, 'messages': [{'role': 'user',
        'content': 'Use the report_number tool to report the integer 7. Do not answer with plain text.'}],
        'tools': tools, 'tool_choice': {'type': 'auto'}})
    calls = [b for b in first.get('content', []) if b['type'] == 'tool_use']
    if not calls or calls[0]['name'] != 'report_number' or calls[0]['input'].get('value') != 7:
        raise RuntimeError(f'Tool-call smoke failed: {first}')
    report['tool_call'] = True
    second = client.json('/v1/messages', {**base, 'tools': tools, 'messages': [
        {'role': 'user', 'content': 'Use report_number to report 7.'},
        {'role': 'assistant', 'content': first['content']},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': calls[0]['id'],
                                     'content': 'The number was accepted. Reply with DONE.'}]}]})
    text = ''.join(b.get('text', '') for b in second.get('content', []) if b['type'] == 'text')
    if 'DONE' not in text.upper():
        raise RuntimeError(f'Tool-result round trip failed: {second}')
    report['tool_result'] = True
    events = []
    with client.request('/v1/messages', {**base, 'stream': True, 'messages': [
            {'role': 'user', 'content': 'What is 17 times 19? Give the result.'}]}) as stream:
        for line in stream:
            if line.startswith(b'data: '):
                payload = line[6:].strip()
                if payload == b'[DONE]':  # OpenRouter adds this after the Anthropic message_stop.
                    break
                if not payload:
                    continue
                value = json.loads(payload)
                if value.get('type') == 'error':
                    raise RuntimeError(f'Stream error: {value}')
                events.append(value)
    if not any(e.get('type') == 'message_stop' for e in events):
        raise RuntimeError('Stream ended without message_stop')
    text = ''.join(e.get('delta', {}).get('text', '') for e in events)
    if '323' not in text:
        raise RuntimeError('Streaming response did not contain the correct answer')
    report['streaming'] = True
    report['thinking_blocks'] = any(e.get('content_block', {}).get('type') == 'thinking' for e in events)
    if not report['thinking_blocks']:
        raise RuntimeError('Maximum-effort smoke returned no thinking content')
    if image:
        data = base64.b64encode(Path(image).read_bytes()).decode()
        answer = client.json('/v1/messages', {**base, 'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': data}},
            {'type': 'text', 'text': 'Briefly describe the visible robot and objects.'}]}]})
        if not any(b.get('text', '').strip() for b in answer.get('content', []) if b['type'] == 'text'):
            raise RuntimeError('Vision smoke returned no description')
        report['vision'] = True
    report['finished_at'] = time.time()
    return report
