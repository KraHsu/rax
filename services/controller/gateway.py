"""MCP-only capability gateway. A session is bound by its trusted transport."""
from __future__ import annotations

import json
import secrets
import threading
import urllib.error
import uuid

from .billing import reserve_cost
from .errors import provider_record, public_failure
from .timing import Timings

from services.benchmarks import ALL_MUTATING, ALL_ROBOT_TOOLS, OBSERVE_TOOLS, STATUS_TOOLS, planning_only
from services.mcp_contract import Contract
from services.simulator.io import SimulatorExited
# Every benchmark's physics-advancing tools; a session's contract admits only its own.
MUTATING = ALL_MUTATING
PRIVATE = {"seed", "eval_seed", "formal_seed", "layout_id", "sim_gpu", "sim_port", "metadata", "environment_spec", "demonstration_context"}


def sanitized(value):
    if isinstance(value, dict):
        return {k: sanitized(v) for k, v in value.items() if k not in PRIVATE and not any(
            part in k for part in ("path", "directory", "root", "artifact_storage")
        )}
    if isinstance(value, list):
        return [sanitized(v) for v in value]
    return value


def episode_limits(value):
    """Public step budget from fields that sanitized() removes with the private layout data."""
    if not isinstance(value, dict):
        return None
    sources = [value.get("environment_spec"), value.get("metadata")]
    limit = next((s.get("episode_step_limit") for s in sources
                  if isinstance(s, dict) and type(s.get("episode_step_limit")) is int), None)
    if limit is None:
        return None
    step = value.get("step_id")
    return {"episode_step_limit": limit,
            **({"remaining_episode_steps": max(0, limit - step)} if type(step) is int else {})}


def ended(value):
    if isinstance(value, dict):
        return value.get("episode_ended") is True or any(ended(v) for v in value.values())
    return isinstance(value, list) and any(ended(v) for v in value)


def complete(value):
    if isinstance(value, dict):
        return value.get("task_complete") is True or any(complete(v) for v in value.values())
    return isinstance(value, list) and any(complete(v) for v in value)


class Gateway:
    def __init__(self, backend, gemini, *, audit, charge, refund, budget, on_success=lambda: None,
                 publish_sequence=lambda value: None, contract=None, provider_log=lambda record: None):
        self.provider_log = provider_log
        self.contract = contract or Contract.from_config(getattr(backend, "config", None), isolated=True)
        self.backend, self.gemini = backend, gemini
        self.audit, self.charge, self.refund = audit, charge, refund
        self.budget = budget
        self.on_success = on_success
        self.publish_sequence = publish_sequence
        self.call_id = self.request_id = None
        self._lock = threading.RLock()
        self._token = None
        self.terminal = False
        self.simulator_exited = None  # The exit report once the simulator process is gone.
        self.timings = Timings()

    def acquire(self):
        with self._lock:
            if self._token is not None:
                raise RuntimeError("Another controller owns the episode")
            self._token = secrets.token_hex(32)
            self.terminal = False
            self.simulator_exited = None
            return self._token

    def revoke(self):
        with self._lock:
            self._token = None

    def definitions(self):
        return self.contract.select([t for t in self.backend.tools() if t["name"] in self.contract.robot_names] + [{
            "name": "gemini_generate", "description": "OpenRouter-style chat completion via Gemini 3.8 Flash: text/images, JSON schema output and client function calling. Returns the complete assistant message plus usage/cost/API budget; Gemini spending, call and token counts are not limited. Per-request size/output limits still apply. Preserve assistant messages including reasoning_details when sending tool results. Tool calls are data only: your script validates and executes functions; they are never executed for you. Images: {type: image, mimeType: image/png or image/jpeg, data: base64}, or image_url with an inline data URL. No paths, remote URLs or server tools. See gemini skill references/api.md.",
            "inputSchema": {"type": "object", "additionalProperties": False,
                "required": ["messages"], "properties": {
                    "messages": {"type": "array", "minItems": 1, "maxItems": 64,
                        "description": "Chat messages: system/user/assistant/tool. Tool results use tool_call_id and string content; preserve complete assistant messages.",
                        "items": {"type": "object"}},
                    "max_tokens": {"type": "integer", "minimum": 1, "maximum": 16384},
                    "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                    "response_format": {"type": "object", "description": "text, json_object, or json_schema with {name, strict, schema}."},
                    "tools": {"type": "array", "maxItems": 64, "items": {"type": "object"},
                        "description": "Client function declarations: {type: function, function: {name, description, parameters, strict?}}."},
                    "tool_choice": {"oneOf": [{"type": "string", "enum": ["auto", "none", "required"]}, {"type": "object"}],
                        "description": "Or {type: function, function: {name: declared_name}}. Multiple calls may be returned."},
                    "reasoning": {"type": "object", "description": "OpenRouter reasoning options: effort or max_tokens, optionally enabled/exclude. Provider support varies."},
                }},
        }])

    def handle(self, token, request):
        params = request.get('params', {}) if isinstance(request, dict) else {}
        params = params if isinstance(params, dict) else {}  # _handle rejects it with an error reply.
        name = params.get('name')
        if not isinstance(name, str) or (name not in ALL_ROBOT_TOOLS and name != 'gemini_generate'):
            name = 'protocol_or_rejected'
        if planning_only(name, params.get('arguments')):
            name = name + '_preview'
        with self._lock, self.timings.measure(name):
            return self._handle(token, request)

    def _handle(self, token, request):
        """Handle one actual MCP JSON-RPC message, with no client-selected identity."""
        with self._lock:
            ident = request.get("id") if isinstance(request, dict) else None
            self.call_id, self.request_id = uuid.uuid4().hex, ident
            self.operation, self.stage = None, 'request'
            try:
                if not self._token or not secrets.compare_digest(token, self._token):
                    raise PermissionError("Expired control capability")
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0":
                    raise ValueError("Invalid JSON-RPC message")
                method = request.get("method")
                if method == "notifications/initialized":
                    return None
                if method == "initialize":
                    result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "robodojo-controller", "version": "1.0"}}
                elif method == "tools/list":
                    result = {"tools": self.definitions()}
                elif method == "tools/call":
                    params = request["params"]
                    result = self.call(params["name"], params.get("arguments", {}))
                else:
                    raise PermissionError("Unsupported MCP method")
                return {"jsonrpc": "2.0", "id": ident, "result": result}
            except Exception as exc:
                if isinstance(exc, SimulatorExited):
                    self.terminal, self.simulator_exited = True, str(exc)
                failure = {**public_failure(exc, operation=self.operation, stage=self.stage),
                           'episode_ended': self.terminal}
                if self.stage == 'provider_request':
                    try:  # The provider's raw answer stays in the operator's private log.
                        self.provider_log(provider_record(exc, call_id=self.call_id))
                    except Exception:
                        failure['provider_log_failed'] = True
                try:
                    self._audit({"event": "mcp_error", **failure})
                except Exception:
                    failure['audit_recording_failed'] = True
                try:
                    budget = self.budget()
                except Exception:
                    budget = None
                return {"jsonrpc": "2.0", "id": ident, "error": {
                    "code": -32000, "message": failure['reason'],
                    "data": {**failure, "call_id": self.call_id, "budget": budget}}}

    def _audit(self, event):
        self.audit({"call_id": self.call_id, "request_id": self.request_id, **event})

    def call(self, name, arguments):
        self.operation = name
        self.contract.require_robot(name, arguments)
        if self.simulator_exited and name not in {"gemini_generate"} | STATUS_TOOLS:
            raise PermissionError(f'The simulator is gone ({self.simulator_exited}); this episode cannot continue')
        if self.terminal and name not in ({"gemini_generate"} | STATUS_TOOLS | OBSERVE_TOOLS
                                         | self.contract.bench.terminal_tools):
            raise PermissionError('Episode has ended; further motion is blocked')
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        self._audit({"event": "mcp_request", "tool": name, "arguments": arguments})
        if name == "gemini_generate":
            self.stage = 'api_validation'
            if self.gemini is None:
                raise RuntimeError("Gemini is not configured")
            payload, reservation = self.gemini.prepare(arguments)
            cost_reservation = reserve_cost(reservation, payload['max_tokens'])
            self.stage = 'api_budget'
            self.charge(reservation, cost_reservation)  # Durable reservations before billable I/O.
            self.stage = 'provider_request'
            try:
                value, used = self.gemini.execute(payload, reservation)
            except urllib.error.HTTPError as exc:
                # An error status carries no completion, so the provider billed nothing.
                try:
                    self.refund(reservation, {'total_tokens': 0, 'cost': 0}, cost_reservation)
                    exc.reservation_released = True
                except Exception as error:
                    exc.add_note(f'Releasing its reservation failed: {type(error).__name__}: {error}')
                raise
            self.stage = 'api_accounting'
            self.refund(reservation - used, value["usage"], cost_reservation)
            value["budget"] = self.budget()
            images = []
        elif name in ALL_ROBOT_TOOLS:
            self.stage = 'validation'
            validator = getattr(self.backend, 'validate', None)
            if validator:
                validator(name, arguments)
            self.stage = 'native_execution'
            value, images = self.backend.call(name, arguments)
            self.stage = 'artifact_publication'
            sequence = self.publish_sequence(value)
            limits = episode_limits(value)
            value = self.contract.observation.clean(sanitized(value))
            if sequence is not None:
                value["frame_sequence"] = self.contract.observation.clean(sequence)
            self.terminal = self.terminal or ended(value)
            if isinstance(value, dict):
                if limits:
                    value.setdefault("episode_limits", limits)
                # Status and observe replies do not carry the terminal step rows.
                value.setdefault("episode_ended", self.terminal)
            if complete(value):
                self.on_success()
        else:
            raise PermissionError("Tool is outside controller capabilities")
        result = {"content": [{"type": "text", "text": json.dumps(value, allow_nan=False)}, *images]}
        self.stage = 'response_recording'
        self._audit({"event": "mcp_result", "tool": name, "result": result})
        return result
