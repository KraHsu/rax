// Host-only adapter to the pinned vendor's credential parser and token resolver.
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const root = process.argv[2], filename = process.argv[3];
try {
  const {parseCredentialsDocument} = await import(pathToFileURL(`${root}/node_modules/@deepseek-ai/dsh-credentials-local/lib/index.js`));
  const {PlatformAccount, Config} = await import(pathToFileURL(`${root}/node_modules/@deepseek-ai/dsh-deepseek-account-platform/lib/index.js`));
  const records = parseCredentialsDocument(readFileSync(filename, 'utf8'), filename).records;
  const config = Config({});
  const token = await PlatformAccount.prototype.resolveToken.call({
    closed: false, origin: config.platformOrigin, inferenceOrigin: config.inferenceOrigin,
    ctx: {credentials: {readRecord: async key => records.get(key)}}
  }, 'https://api.deepseek.com/anthropic');
  if (!token) throw new Error('Sign-in required');
  process.stdout.write(JSON.stringify({headers: {'x-dsh-auth-token': token}}));
} catch {
  process.stderr.write('DeepSeek account authentication failed; run the dedicated account login.\n');
  process.exitCode = 1;
}
