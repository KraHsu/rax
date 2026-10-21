"""Freeze a finished exploration, then run its formal evaluations in isolation."""
import json
from pathlib import Path
import time
import uuid

from services.controller.errors import describe
from services.controller.storage import persist
from services.transfer import PARTS, pack
from .runner import harness


def baseline_usage(cli, home, session_id):
    """Codex's cumulative thread usage at freeze time; formal forks subtract it."""
    if cli != 'codex':
        return None  # Claude usage is counted per episode from its own relay log.
    from services.codex_usage import CodexUsageReader
    from harness.codex_cli.auto_research import session_files
    return CodexUsageReader().read(home/session_files(home, session_id)[0])['total_session_usage']


def _register(supervisor, ident, manifest):
    manifest['directory'] = ident
    supervisor.state['bundles'][ident] = manifest


def freeze_exploration(config, supervisor, deployed):
    """Freeze the exploring agent as it stands when exploration ends: every transfer folder
    and its whole conversation, before any formal episode runs. Returns the bundle id.

    Formal (bundle and agent) and adaptation trials start from this package, never from the
    live workspace, so nothing the agent learns afterwards can reach them.
    """
    existing = supervisor.state.get('exploration_checkpoint')
    if existing:
        return existing['bundle']
    cli = harness(config.agent_cli)
    home, workspace = deployed[cli.HOME_KEY], deployed['workspace']
    session_id = cli.latest_session(home)
    ident = uuid.uuid4().hex
    parent = config.root/'bundles'
    parent.mkdir(mode=0o700, exist_ok=True)
    manifest, record = pack(
        parent/ident, workspace=workspace, parts=tuple(PARTS), max_mb=config.transfer_mb,
        home=home, session_files=cli.session_files(home, session_id) if session_id else (),
        home_files=cli.HOME_FILES,
        metadata={'cli': config.agent_cli, 'session_id': session_id, 'frozen_at': time.time(),
                  'stage': 'exploration_end'})
    _register(supervisor, ident, manifest)
    supervisor.state['exploration_checkpoint'] = {'bundle': ident, 'parts': record['parts'],
                                                  'excluded': record['excluded'], 'session_id': session_id}
    supervisor._save()
    return ident


def package_path(root, state):
    """The frozen exploration package of a session, if it has one."""
    checkpoint = state.get('exploration_checkpoint')
    if not checkpoint:
        return None
    return Path(root)/'bundles'/state['bundles'][checkpoint['bundle']]['directory']


def freeze(config, supervisor, deployed, *, compaction=None):
    """The agent formal checkpoint: the configured transfer folders from the exploration
    freeze plus its conversation (compacted, when configured). Returns its bundle id."""
    base = package_path(config.root, supervisor.state)
    record = json.loads((base/'package.json').read_text())
    session_id = record.get('session_id')
    cli = harness(config.agent_cli)
    if compaction is not None and compaction.get('status') == 'compacted':
        # Compaction rewrote the frozen conversation in place in the live home, after the
        # freeze; the folders still come from the freeze.
        home = deployed[cli.HOME_KEY]
        session_files = cli.session_files(home, session_id)
    else:
        home, session_files = base/'session', record['session_files']
    ident = uuid.uuid4().hex
    manifest, packed = pack(
        config.root/'bundles'/ident, workspace=base/'transfer', parts=config.transfer_parts,
        max_mb=config.transfer_mb, home=home, session_files=session_files,
        home_files=record['home_files'] if home == base/'session' else cli.HOME_FILES,
        metadata={'cli': config.agent_cli, 'session_id': session_id, 'frozen_at': record['frozen_at'],
                  'baseline_usage': baseline_usage(config.agent_cli, home, session_id) if session_id else None,
                  'formal_context': config.formal_context, 'compaction': compaction})
    _register(supervisor, ident, manifest)
    supervisor.state['agent_checkpoint'] = {'bundle': ident, 'parts': packed['parts'],
                                            'excluded': packed['excluded'], 'skipped': len(packed['skipped']),
                                            'compaction': compaction}
    supervisor._save()
    return ident


def exploration_closed(config):
    """The agent submitted (code mode), or the frontend saw the exploration budget spent with
    nothing live."""
    if (config.root/'exploration-complete.json').is_file():
        return True
    try:
        state = json.loads((config.root/'state.json').read_text())
    except (OSError, ValueError):
        return False
    return bool(state.get('submission'))


def after_exploration(config, deployed, *, gemini_key_file=None, live_copies=False):
    """Called by the launcher once the exploring agent has exited (or on a resume after it
    closed exploration). Freezes the agent, then runs the session's evaluation in isolation:
    the submitted bundle's formal batch (code mode) or agent formal (live-copy mode).
    live_copies also runs agent formal after a code-mode session's bundle formal, on the same
    layouts from the same freeze. Each step runs at most once; an exploration left open runs
    nothing."""
    summary = {'status': 'exploration_incomplete'}
    if not exploration_closed(config):
        persist(config.root/'formal.json', summary)
        return summary
    from services.benchmarks import for_config
    from services.controller.backend import make_backend
    from services.controller.gemini import GeminiRouter, load_key
    from services.controller.supervisor import Supervisor
    key = load_key(gemini_key_file) if config.external_perception else None
    supervisor = Supervisor(config, lambda: make_backend(config), GeminiRouter(key) if key else None)
    try:
        summary = {'status': 'frozen', 'checkpoint': freeze_exploration(config, supervisor, deployed)}
        persist(config.root/'formal.json', summary)
        if supervisor.state.get('submission') and not config.agent_formal:
            if not supervisor.submission_done():
                summary['bundle_formal'] = 'running'
                persist(config.root/'formal.json', summary)
                supervisor.run_submission()
            summary['bundle_formal'] = _outcome(supervisor.state.get('formal_batch'), supervisor.state,
                                                supervisor.state['submission']['bundle'])
        if config.agent_formal or live_copies:
            summary['agent_formal'] = agent_formal(config, supervisor, deployed)
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='failed', error_type=type(exc).__name__, error=describe(exc))
        raise
    finally:
        persist(config.root/'formal.json', summary)
        supervisor.close()
    return summary


def _outcome(batch, state, bundle):
    if batch:
        return {k: batch.get(k) for k in ('status', 'success_count', 'episode_count', 'success_rate')}
    result = next((r for r in reversed(state['results'])
                   if r.get('mode') == 'formal' and r.get('bundle') == bundle), None)
    return {'status': 'completed', 'task_complete': result.get('task_complete')} if result else None


def agent_formal(config, supervisor, deployed):
    batch = supervisor.state.get('agent_formal_batch')
    if batch and batch.get('status') == 'interrupted':
        # Stopped before every layout had an outcome: play the rest with fresh copies.
        bundle = batch['bundle']
        persist(config.root/'agent-formal.json', {'status': 'resumed', 'bundle': bundle})
        result = _outcome(supervisor.resume_agent_formal(), supervisor.state, None)
        persist(config.root/'agent-formal.json', {'status': supervisor.state['agent_formal_batch']['status'],
                                                  'bundle': bundle, **(result or {})})
        return result
    if batch and batch.get('status') != 'running':
        return _outcome(batch, supervisor.state, None)
    compaction = None
    if not supervisor.state.get('agent_checkpoint'):
        if config.formal_context == 'compact' and supervisor.state['exploration_checkpoint'].get('session_id'):
            from harness.session_context import compact
            compaction = compact(config, deployed, supervisor.state['exploration_checkpoint']['session_id'],
                                 log=config.root/'compaction.jsonl')
        bundle = freeze(config, supervisor, deployed, compaction=compaction)
    else:
        bundle = supervisor.state['agent_checkpoint']['bundle']
    persist(config.root/'agent-formal.json', {'status': 'started', 'bundle': bundle, 'compaction': compaction})
    batch = supervisor.run_agent_formal(bundle)
    result = _outcome(batch, supervisor.state, None)
    persist(config.root/'agent-formal.json', {'status': batch['status'], 'bundle': bundle,
                                              'compaction': compaction, **(result or {})})
    return result


def read_summary(root):
    try:
        return json.loads((root/'agent-formal.json').read_text())
    except (OSError, ValueError):
        return None
