// Route the vendor SDK's fetch through one Unix socket. No secrets enter Node.
import http from 'node:http';
import { Readable } from 'node:stream';

globalThis.fetch = async function(input, init = {}) {
  const request = new Request(input, init);
  const url = new URL(request.url);
  const paths = process.env.RAX_MODEL_PROTOCOL === 'messages'
    ? ['/v1/messages', '/v1/messages/count_tokens'] : ['/v1/chat/completions'];
  if (url.origin !== 'http://rax-model.invalid' || !paths.includes(url.pathname))
    throw new Error('RAX: network endpoint is outside this experiment');
  const body = Buffer.from(await request.arrayBuffer());
  return new Promise((resolve, reject) => {
    const outgoing = http.request({socketPath: process.env.RAX_MODEL_SOCKET,
      path: url.pathname, method: request.method,
      headers: {'content-type': 'application/json', 'content-length': body.length}}, response => {
      resolve(new Response(Readable.toWeb(response), {status: response.statusCode,
        headers: {'content-type': response.headers['content-type'] || 'application/json'}}));
    });
    const abort = () => outgoing.destroy(new Error('Model request aborted'));
    request.signal.addEventListener('abort', abort, {once: true});
    outgoing.on('close', () => request.signal.removeEventListener('abort', abort));
    outgoing.on('error', reject);
    if (request.signal.aborted) abort();
    else outgoing.end(body);
  });
};
