"""Public RoboTwin action contract. No simulator imports."""
import math

import numpy as np

from services.benchmarks import ROBOTWIN_TOOLS
from services.simulator.io import reject_unless

CAMERAS = ('head_camera', 'left_camera', 'right_camera')
MAX_ROWS = 32


def vector(n):
    return {'type': 'array', 'minItems': n, 'maxItems': n, 'items': {'type': 'number'}}


def tool_definitions():
    def tool(name, description, properties=None, required=()):
        return {'name': name, 'description': description, 'inputSchema': {
            'type': 'object', 'properties': properties or {}, 'required': list(required), 'additionalProperties': False}}
    result = [
        tool('robotwin_status', 'Episode identity, remaining native command budget and action conventions.'),
        tool('robotwin_observe', 'Measured EEF poses, joint state, instruction and camera RGB. No physics.'),
        tool('robotwin_step', 'Execute 1..32 native absolute joint-target rows (arm joints then gripper per arm). '
             'Grippers are [0,1], 0 closed. Each row consumes one native command, not one fixed-duration tick.',
             {'actions': {'type': 'array', 'minItems': 1, 'maxItems': MAX_ROWS,
                          'items': {'type': 'array', 'items': {'type': 'number'}}}}, ('actions',)),
        tool('robotwin_step_eef', 'Execute absolute world-frame EEF targets, metres and quaternion WXYZ. '
             'Each 16D row is [L xyz3, L wxyz4, L gripper, R xyz3, R wxyz4, R gripper]. '
             'RoboTwin plans each target internally; observe actual motion, including planning failure. '
             'These are native commands, not normalized WAM predictions or delta actions.',
             {'actions': {'type': 'array', 'minItems': 1, 'maxItems': MAX_ROWS, 'items': vector(16)}}, ('actions',)),
        tool('robotwin_constraints', 'List or upsert a base geometric constraint in the currently recorded HDF5. '
             'No operation categories. Use the geometric-constraints skill template and recorded observation frame indices. '
             'Only reference data activates constraints. Writes are disabled during formal evaluation.',
             {'operation': {'type': 'string', 'enum': ['list', 'upsert']},
              'constraint': {'type': 'object'}, 'expected_revision': {'type': 'integer', 'minimum': 0}}, ('operation',)),
    ]
    assert {t['name'] for t in result} == ROBOTWIN_TOOLS
    return result


def validate_request(name, arguments):
    reject_unless(name in ROBOTWIN_TOOLS and isinstance(arguments, dict), 'Unknown tool or malformed arguments')
    if name in ('robotwin_status', 'robotwin_observe'):
        reject_unless(not arguments, 'This tool takes no arguments')
    elif name in ('robotwin_step', 'robotwin_step_eef'):
        reject_unless(set(arguments) == {'actions'}, 'Only actions is accepted')
        rows = arguments['actions']
        reject_unless(isinstance(rows, list) and 1 <= len(rows) <= MAX_ROWS, 'Expected 1..32 action rows')
        width = 16 if name.endswith('_eef') else None
        for row in rows:
            reject_unless(isinstance(row, list) and 4 <= len(row) <= 64 and len(row) % 2 == 0, 'Invalid row width')
            reject_unless(all(type(x) in (int, float) and math.isfinite(x) for x in row), 'Actions must be finite')
            reject_unless(width is None or len(row) == width, 'EEF rows must be 16D')
            reject_unless(len(row) == len(rows[0]), 'Inconsistent row widths')
            reject_unless(0 <= row[len(row)//2-1] <= 1 and 0 <= row[-1] <= 1, 'Grippers must be in [0,1]')
            if width:
                for offset in (3, 11):
                    reject_unless(abs(np.linalg.norm(row[offset:offset+4])-1) < 1e-4, 'Quaternions must be unit WXYZ')
    else:
        operation = arguments.get('operation')
        required = {'list': set(), 'upsert': {'constraint', 'expected_revision'}}
        reject_unless(operation in required, 'Unknown constraint operation')
        reject_unless(set(arguments) == required[operation] | {'operation'}, 'Unexpected constraint arguments')
