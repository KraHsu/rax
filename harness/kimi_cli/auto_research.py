"""Native Kimi Code backend; shared lifecycle and sandbox, vendor-owned loop."""
from harness.isolated_cli.adapter import Adapter
from . import spec

_adapter = Adapter(spec)
HOME_KEY, HOME_NAME, HOME_FILES = _adapter.HOME_KEY, _adapter.HOME_NAME, _adapter.HOME_FILES
deploy, redeploy = _adapter.deploy, _adapter.redeploy
write_home, agent_command = _adapter.write_home, _adapter.agent_command
agent_environment, launch = _adapter.agent_environment, _adapter.launch
latest_session, session_files = _adapter.latest_session, _adapter.session_files
last_error, capture_event = _adapter.last_error, _adapter.capture_event
record_subagents, relay = _adapter.record_subagents, _adapter.relay
binary, sandbox = _adapter.binary, _adapter.sandbox
