"""Read instruction fields without invoking task methods that execute actions.

The two adapters below are bound to audited RoboTwin task source, commit
0008ae6800df9f75fc8de7098bacb01735fd8fd2. Actor poses and asset IDs already
exist after setup_demo. Unknown tasks return None, never call get_info.
"""
from copy import deepcopy
import hashlib
import inspect
import math
from numbers import Integral
from pathlib import Path


ADAPTER_VERSION = 'initial_scene_instruction_metadata_v1'
SOURCE_SHA256 = {
    'blocks_ranking_size': '0517ae78933baa225aa09e4db3ce2004538c9c6873c584d4044b29b2f509581c',
    'place_bread_basket': '3751aa10c991f638be145f7536c9e1e398b1e116ba81f207dd4a11a6ae6a99ab',
}


def _x(actor):
    value = float(actor.get_pose().p[0])
    if not math.isfinite(value):
        raise ValueError('Instruction metadata needs a finite initial actor x')
    return value


def _asset_id(value, allowed):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in allowed:
        raise ValueError('Unexpected native asset ID; do not invent instruction metadata')
    return int(value)


def audited_native_source(task, native, sources):
    """Shared source identity check for read-only instruction/success adapters."""
    if type(native).__name__ != task:
        raise ValueError('Instruction task and native environment class differ')
    source = Path(inspect.getfile(type(native)))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != sources[task]:
        raise ValueError('Native task source changed; review the read-only metadata adapter')
    return actual


def readonly_instruction_metadata(task, native):
    """Return audited initial-scene metadata/provenance, or None if unsupported.

    Call immediately after setup_demo, before any action. The caller checks its
    scene fingerprint before and after reading. No native info dictionary is
    mutated, and no get_info/play_once/motion function is called.
    """
    if task not in SOURCE_SHA256:
        from services.robotwin.success_metadata import readonly_success_instruction_metadata
        return readonly_success_instruction_metadata(task, native)
    actual = audited_native_source(task, native, SOURCE_SHA256)
    if task == 'blocks_ranking_size':
        # Native actor identities are size identities, independent of execution
        # order (the expert visits block3, block2, block1).
        info = {'{A}': 'large block', '{B}': 'medium block', '{C}': 'small block'}
        for token, name in zip(('{a}', '{b}', '{c}'), ('block1', 'block2', 'block3')):
            info[token] = 'left' if _x(getattr(native, name)) < 0 else 'right'
        fields = ['block1.get_pose().p[0]', 'block2.get_pose().p[0]', 'block3.get_pose().p[0]']
    else:
        breads, ids = native.bread, native.bread_id
        if len(breads) not in (1, 2) or len(ids) != len(breads):
            raise ValueError('Instruction metadata requires one or two matched bread actors/IDs')
        basket = _asset_id(native.basket_id, range(5))
        ids = [_asset_id(value, (0, 1, 3, 5, 6)) for value in ids]
        positions = [_x(actor) for actor in breads]
        if any(value == 0 for value in positions):
            # load_actors excludes the center strip. Native motion and old
            # post-motion metadata disagree at exactly zero; do not pick one.
            raise ValueError('Bread on the initial center line is outside the audited scene generator')
        # Single bread's original get_info reads x AFTER moving it to the
        # basket. Use the initial arm allocation specified by the native task
        # schema instead; two opposite-side breads require the dual-arm branch.
        arm = ('left' if positions[0] < 0 else 'right')
        if len(breads) == 2 and positions[0] * positions[1] <= 0:
            arm = 'dual'
        info = {'{A}': f'076_breadbasket/base{basket}', '{B}': f'075_bread/base{ids[0]}', '{a}': arm}
        if len(breads) == 2:
            info['{C}'] = f'075_bread/base{ids[1]}'
        fields = ['basket_id', 'bread_id', 'bread[*].get_pose().p[0]']
    return {'info': deepcopy(info), 'provenance': {'adapter': ADAPTER_VERSION, 'task': task,
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_source_sha256': actual, 'source_fields': fields, 'expert_actions_for_instruction': False}}
