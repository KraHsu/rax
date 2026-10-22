#!/usr/bin/env python3
"""Run the local RAX Codex harness as an offline trajectory annotator."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from harness import agent_host
from harness.codex_cli.auto_research import agent_command, agent_environment, chatgpt_login, codex_binary, expose_published, settings
from harness.codex_cli.configuration import write_config
from services.robotwin.constraint_schema import validate_bundle


INSTRUCTIONS = """# RAX offline RoboTwin constraint annotation

Read TASK.md and input/review.json, then inspect input/trajectory.csv and the
relevant preview images. Read .agents/skills/geometric-constraints/SKILL.md for
the seven geometric primitives and their units. This is OFFLINE annotation:
there is no robot MCP, no simulation, and no physics action to call. The skill's
live recording/upsert instructions do not apply. Frames are the native recorded
observation indices in the CSV, not sparse command boundaries.

Write annotations.json with EXACTLY these two keys:
  source_sha256: copy the input hash verbatim
  constraints: an array following the existing constraint schema
Also write REPORT.md explaining each choice and uncertain geometry.

Use composable geometric constraints, with no operation-class taxonomy. Choose
only task-relevant features; do not constrain every coordinate throughout the
trajectory. Select activation from the recorded demonstration, never generated
actions. Event frames must match a genuine gripper threshold crossing. Use the
input recording_id as evidence.report_id, and demonstration_analysis as source.

You may mark measured reference envelopes accepted using ONLY the explicitly
provided annotation_policy margins. Explain that these are operator priors,
not empirically verified physical feasible regions. Accepted means a checked
annotation, not proof of object grasp/contact/support or task robustness. All
accepted constraints must admit the reference values over their active frames.
For a new geometric claim unsupported by measurements/explicit priors, use
proposed or unknown; these are excluded from training. A source expert's success
does not reveal the boundaries of a successful grasp region.

Prefer a small justified set: closure-event pose or gripper opening, terminal
lift displacement, task-sensitive axis/angle if supported. Do not invent a
required orientation from the task name. State whether a cross-chunk displacement
constraint requires its start frame inside a training chunk; also use local
endpoint/interval constraints where justified. Never copy example numbers.

Use Python to calculate quantitative bounds from the CSV and verify them. Inspect
images to understand the demonstrated behavior. Keep inputs read-only. Do not
use external services, spawn helpers, inspect simulator internals, or train a
model. End when both annotations.json and REPORT.md are complete.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--login-home', type=Path, default=Path.home()/'.codex')
    p.add_argument('--timeout', type=int, default=1800, help='Maximum seconds per offline annotation')
    p.add_argument('--task-template', action='store_true', help='Also review a reusable numeric task template')
    args = p.parse_args()
    source = args.review.resolve()
    review = json.loads((source/'review.json').read_text())
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False, mode=0o700)
    workspace = root/'agent-workspace'; workspace.mkdir(mode=0o700)
    home = root/'codex-home'; home.mkdir(mode=0o700)
    shutil.copytree(source, workspace/'input')
    skill = workspace/'.agents/skills/geometric-constraints'
    shutil.copytree(PROJECT/'agent_skills/control/robotwin/geometric-constraints', skill)
    instructions = INSTRUCTIONS
    if args.task_template:
        instructions += '''\nAlso create a reusable task_template.json. Read
.agents/skills/geometric-constraints/references/programmatic-template.md.
Select task-relevant features once; all numeric values will be recomputed for
each episode. Copy the observed event_signature into signatures; do not expand
coverage beyond the observed pattern. Run ./validate_task_template.py
task_template.json and inspect its audit. You must provide both the per-episode
annotations.json/REPORT.md and the reusable template. Explain any omitted
features that the selector language cannot safely express. The three-seed gate
on the data host checks numerical validity, not physical feasibility.\n'''
        from services.robotwin.programmatic_annotation import instantiate_review
        for relative in ('services/__init__.py','services/robotwin/__init__.py'):
            path=workspace/'template_runtime'/relative; path.parent.mkdir(parents=True,exist_ok=True);path.write_text('')
        for name in ('annotation.py','constraint_schema.py','programmatic_annotation.py'):
            shutil.copy2(PROJECT/'services/robotwin'/name,workspace/'template_runtime/services/robotwin'/name)
        helper=workspace/'validate_task_template.py'
        helper.write_text('#!'+sys.executable+'\n'+'''import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'template_runtime'))
from services.robotwin.programmatic_annotation import instantiate_review
c,a=instantiate_review('input',json.loads(Path(sys.argv[1]).read_text()))
print(json.dumps({'constraints':c,'audit':a},indent=2))
''')
        helper.chmod(0o700)
    (workspace/'AGENTS.md').write_text(instructions)
    (workspace/'TASK.md').write_text('# Demonstration\n\n'+review['instruction']+'\n')
    auth = (args.login_home.expanduser()/'auth.json').resolve(); chatgpt_login(auth)
    local = tomllib.loads((args.login_home.expanduser()/'config.toml').read_text())
    model = local['model']
    login = root/'agent-login.json'
    login.write_text(json.dumps({'model': model, 'chatgpt_auth': str(auth)})); login.chmod(0o600)
    (home/'auth.json').symlink_to(auth)
    config = SimpleNamespace(root=root, agent_effort='xhigh', training=SimpleNamespace(cpus=2))
    deployed = {'workspace': workspace, 'codex_home': home, 'codex': codex_binary(),
                'login': login, 'subagents': 0}
    expose_published(root)
    socket = agent_host.socket_path(root)
    value = settings(config, deployed, socket)
    value.pop('mcp_servers')
    value['permissions']['agent']['filesystem'][str(workspace/'input')] = 'read'
    value['permissions']['agent']['network']['dangerously_allow_all_unix_sockets'] = False
    write_config(home/'config.toml', value)
    command = agent_command(config, deployed, prompt='Read AGENTS.md and TASK.md, then annotate this demonstration.')
    metadata = {'run_id': uuid.uuid4().hex, 'model': model, 'effort': 'xhigh',
                'source_sha256': review['source_sha256'], 'role': 'offline_annotation', 'status': 'running'}
    (root/'run.json').write_text(json.dumps(metadata, indent=2))
    process = None
    try:
        process = agent_host.start(command, cwd=workspace, env=agent_environment(config, deployed, socket),
                                   stdin=subprocess.DEVNULL)
        code = process.wait(timeout=args.timeout)
        if code:
            metadata.update(status='failed', returncode=code)
            raise SystemExit(code)
        proposal = json.loads((workspace/'annotations.json').read_text())
        if set(proposal) != {'source_sha256', 'constraints'} or proposal['source_sha256'] != review['source_sha256']:
            raise ValueError('Agent output has invalid source identity or fields')
        bundle = dict(review['bundle_template'], constraints=proposal['constraints'])
        validate_bundle(bundle, review['frame_count'])
        if not (workspace/'REPORT.md').is_file():
            raise ValueError('Agent did not write its annotation report')
        if args.task_template:
            template=json.loads((workspace/'task_template.json').read_text())
            generated,audit=instantiate_review(source,template)
            if template['signatures'] != [review['event_signature']]:
                raise ValueError('New task template must use only its observed signature')
            (root/'template_audit.json').write_text(json.dumps(audit,indent=2))
        metadata.update(status='proposed', constraints=len(proposal['constraints']),
                        note='Physical/reference checks run on the data host before embedding')
    except BaseException as exc:
        if metadata['status'] == 'running':
            metadata.update(status='failed' if isinstance(exc, Exception) else 'interrupted',
                            error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (root/'run.json').write_text(json.dumps(metadata, indent=2))
        if process is not None:
            agent_host.stop(process)
    print(json.dumps({'proposal': str(workspace/'annotations.json'), **metadata}), flush=True)


if __name__ == '__main__':
    main()
