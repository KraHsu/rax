// Record wire capabilities without headers, credentials, prompts or image bytes.
import { appendFileSync } from 'node:fs';

const original = globalThis.fetch;
function imageCount(value) {
  if (!value || typeof value !== 'object') return 0;
  if (Array.isArray(value)) return value.reduce((sum, item) => sum + imageCount(item), 0);
  if (['image_url', 'image', 'input_image'].includes(value.type)) return 1;
  // Anthropic images are nested inside tool_result.content; OpenAI uses message.content.
  return Object.values(value).reduce((sum, item) => sum + imageCount(item), 0);
}
globalThis.fetch = async function(input, init) {
  const url = String(input?.url ?? input);
  if (process.env.RAX_DSH_AUDIT && /\/(chat\/completions|responses|messages)(\?|$)/.test(url)) {
    try {
      const body = JSON.parse(init?.body ?? '{}');
      const messages = body.messages ?? body.input ?? [];
      const imageBlocks = imageCount(messages);
      appendFileSync(process.env.RAX_DSH_AUDIT, JSON.stringify({ at: Date.now()/1000,
        endpoint: url.split('?')[0], model: body.model, reasoning: body.reasoning,
        reasoning_effort: body.reasoning_effort, thinking: body.thinking, output_config: body.output_config,
        max_tokens: body.max_tokens ?? body.max_completion_tokens ?? body.max_output_tokens,
        tools: (body.tools ?? []).map(tool => tool.function?.name ?? tool.name), image_blocks: imageBlocks })+'\n', { mode: 0o600 });
    } catch { /* Observability must never change a provider request. */ }
  }
  return Reflect.apply(original, this, [input, init]);
};
