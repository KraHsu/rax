// Observe the native account service without exporting its credential grant.
import { writeFileSync, renameSync } from 'node:fs';
import { join } from 'node:path';

export const name = 'rax-account-status';
export const inject = ['deepseekAccount', 'timer'];
export function apply(ctx) {
  const path = join(process.env.DSH_HOME, 'account-status.json');
  let started = false;
  const update = async () => {
    try {
      let value = await ctx.deepseekAccount.getState();
      if (!started && value.status === 'signed-out' && process.env.RAX_DSH_START_SIGNIN === '1') {
        started = true;
        value = await ctx.deepseekAccount.startSignIn({ version: '0.2.1-rc.2', locale: 'zh-CN',
          timezoneOffsetSeconds: -new Date().getTimezoneOffset()*60 },
          process.env.RAX_DSH_CALLBACK_ORIGIN, 'web');
      }
      const safe = { status: value.status, signed_in: value.status === 'credential-stored',
        attempt_phase: value.attempt?.phase ?? null, authorize_url: value.attempt?.authorizeUrl,
        expires_at: value.attempt?.expiresAt, attempt_error_code: value.attempt?.errorCode,
        checked_at: Date.now()/1000 };
      writeFileSync(`${path}.tmp`, JSON.stringify(safe, null, 2), { mode: 0o600 });
      renameSync(`${path}.tmp`, path);
    } catch (error) {
      // Error classes are sufficient here; native diagnostics stay in the private log.
      writeFileSync(path, JSON.stringify({ status: 'error', error_type: error.name }), { mode: 0o600 });
    }
  };
  ctx.setInterval(update, 5000);
  void update();
}
