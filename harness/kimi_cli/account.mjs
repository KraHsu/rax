// Host-only helper. Stdout is a private pipe to the model relay, never a log.
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const {KimiOAuthToolkit, createKimiDefaultHeaders} = require(process.argv[2]);
const home = process.argv[3];
const oauthRef = JSON.parse(process.argv[4]);
const identity = {productName: 'kimi-code-cli', version: '2.2.0', platform: 'kimi_code_cli'};
const toolkit = new KimiOAuthToolkit({homeDir: home, identity});
try {
  const token = await toolkit.ensureFresh(undefined, {oauthRef});
  process.stdout.write(JSON.stringify({headers: {
    ...createKimiDefaultHeaders({...identity, homeDir: home}), Authorization: `Bearer ${token}`}}));
} catch {
  process.stderr.write('Kimi account authentication failed. Run kimi login in the dedicated auth home.\n');
  process.exitCode = 1;
}
