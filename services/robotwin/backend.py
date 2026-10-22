"""RAX backend: clean exploration, frozen randomized evaluation, constraint recording."""
import base64
import io
import json
from pathlib import Path
import time
import uuid

from PIL import Image

from services.robotwin.process import RoboTwinProcess
from services.robotwin.tools import tool_definitions, validate_request
from services.simulator.io import action_execution_scope, reject_unless


class RoboTwinBackend:
    evaluates_from_record = True

    def __init__(self, config):
        from services.mcp_contract import Contract
        self.config = config
        self.contract = Contract.from_config(config, isolated=True)
        root = Path(config.root) / 'native'
        root.mkdir(parents=True, exist_ok=True)
        self.frame_workspace = root / 'workspace'
        self.sim = RoboTwinProcess(output_root=root / 'results')
        self._formal = False

    def tools(self):
        return self.contract.select(tool_definitions())

    validate = staticmethod(validate_request)

    def _packet(self, observation):
        packet = {k: v for k, v in observation.items() if k != 'images'}
        packet['environment_spec'] = {'episode_step_limit': self.sim.metadata['native_step_limit']}
        images, attachments = [], []
        for camera, pixels in observation.get('images', {}).items():
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format='PNG')
            images.append({'type': 'image', 'data': base64.b64encode(stream.getvalue()).decode(), 'mimeType': 'image/png'})
            attachments.append({'kind': 'rgb', 'camera': camera, 'content_index': len(images)})
        packet['attachments'] = attachments
        return packet, images

    def start(self, *, formal, seed):
        self.close()
        self._formal = formal
        run_id = f'{self.config.task}_{uuid.uuid4().hex}'
        reset = self.sim.start(task=self.config.task, collection=1 if formal else 0, index=int(seed),
                               run_id=run_id, sim_gpu=self.config.sim_gpu)
        packet, images = self._packet(self.sim.request('observe'))
        packet.update(reset=reset, artifact_id=run_id, artifact_storage='service_managed_not_agent_accessible',
                      environment_spec={'benchmark': 'robotwin', 'episode_mode': 'formal' if formal else 'exploration',
                                        'episode_step_limit': reset['native_step_limit'],
                                        'native_step_limit': reset['native_step_limit']})
        return packet, images

    def call(self, name, arguments):
        self.contract.require_robot(name, arguments)
        validate_request(name, arguments)
        with action_execution_scope(self.sim, name):
            if name == 'robotwin_status':
                return {'active': self.sim.active, 'episode_id': self.sim.episode_id, 'step_id': self.sim.step_id,
                        'action_conventions': self.sim.metadata}, []
            if name == 'robotwin_observe':
                return self._packet(self.sim.request('observe'))
            if name == 'robotwin_constraints':
                op = arguments['operation']
                if op == 'list':
                    return self.sim.request('constraints_list'), []
                reject_unless(not self._formal, 'Constraints are frozen during formal evaluation')
                return self.sim.request('constraint_upsert', constraint=arguments['constraint'],
                                        expected_revision=arguments['expected_revision']), []
            result = self.sim.request('chunk_step', actions=arguments['actions'],
                                      action_type='ee' if name.endswith('_eef') else 'qpos')
            packet, images = self._packet(result['observation'])
            packet.update(executed_rows=result['executed_rows'], episode_ended=result['stopped'])
            return packet, images

    def evaluate(self, agent_assessment=None, confidence=None):
        if self.sim.output_dir is None:
            raise RuntimeError('No episode to evaluate')
        state = self.sim.operator_state()
        if self.sim.finished and state.get('episode_id') == self.sim.episode_id:
            self.sim.step_id = max(self.sim.step_id, state['step_id'])
        if (state.get('episode_id'), state.get('step_id')) != (self.sim.episode_id, self.sim.step_id):
            raise RuntimeError('Native evaluation identity mismatch')
        success = state.get('reward', {}).get('native_success')
        if type(success) is not bool:
            raise RuntimeError('Native evaluation unavailable')
        return {'status': 'completed', 'task_complete': success, 'episode_id': self.sim.episode_id,
                'step_id': self.sim.step_id, 'task': self.config.task, 'evaluation_kind': 'binary_task_completion',
                'decided_at_unix_s': time.time()}

    def close(self):
        try:
            if self.sim.active:
                self.sim.finish('close')
        finally:
            self.sim.stop()
