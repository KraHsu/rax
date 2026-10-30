"""Execute adapters against classes extracted from the audited native source."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest

from services.robotwin.instruction_metadata import SOURCE_SHA256, readonly_instruction_metadata

FIXTURES = Path(__file__).parents[1]/'fixtures/robotwin_instruction_metadata'


def actor(x):
    pose = SimpleNamespace(p=[x, .03, .74])
    return SimpleNamespace(get_pose=lambda: pose)


def environment(task, monkeypatch):
    source = FIXTURES/(task+'.py')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA256[task]
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == task)
    module = ModuleType('audited_'+task); module.__file__ = str(source); module.Base_Task = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), module.__dict__)
    native = getattr(module, task)()
    native.info = {'info': {}, 'unrelated': 'preserve'}; native.last_gripper = 'untouched'
    def forbidden(*a, **kw):
        raise AssertionError('Metadata initialization executed a native action/get_info')
    for name in ('move', 'play_once', 'get_info', 'pick_and_place_block', 'remove', 'remove_bread',
                 'grasp_actor', 'place_actor', 'open_gripper', 'move_by_displacement', 'back_to_origin'):
        setattr(native, name, forbidden)
    native.scene = SimpleNamespace(remove_actor=forbidden, remove_articulation=forbidden)
    return native


def test_block_size_identity_and_arm_mapping_are_read_only(monkeypatch):
    native = environment('blocks_ranking_size', monkeypatch)
    native.block1, native.block2, native.block3 = [actor(x) for x in (-.2, .21, -.18)]
    before = deepcopy(native.info)
    result = readonly_instruction_metadata('blocks_ranking_size', native)
    assert result['info'] == {'{A}':'large block','{B}':'medium block','{C}':'small block',
                              '{a}':'left','{b}':'right','{c}':'left'}
    assert native.info == before and native.last_gripper == 'untouched'
    assert [getattr(native, name).get_pose().p[0] for name in ('block1','block2','block3')] == [-.2,.21,-.18]
    assert result['provenance']['expert_actions_for_instruction'] is False


@pytest.mark.parametrize('positions,expected_arm', [([-.2],'left'),([.2],'right'),
    ([-.2,-.25],'left'),([.2,.25],'right'),([-.2,.2],'dual'),([.2,-.2],'dual')])
def test_bread_variants_use_initial_scene_asset_ids_without_motion(monkeypatch, positions, expected_arm):
    native = environment('place_bread_basket', monkeypatch)
    native.basket_id = 4; native.bread = [actor(x) for x in positions]; native.bread_id = [5,1][:len(positions)]
    before = deepcopy(native.info)
    result = readonly_instruction_metadata('place_bread_basket', native)
    expected = {'{A}':'076_breadbasket/base4','{B}':'075_bread/base5','{a}':expected_arm}
    if len(positions) == 2: expected['{C}'] = '075_bread/base1'
    assert result['info'] == expected and native.info == before
    assert [bread.get_pose().p[0] for bread in native.bread] == positions
    assert native.last_gripper == 'untouched'


def test_actual_native_instruction_placeholders_are_available(monkeypatch):
    for task in SOURCE_SHA256:
        native = environment(task, monkeypatch)
        native.block1, native.block2, native.block3 = actor(-.2), actor(.2), actor(-.2)
        native.basket_id=0;native.bread=[actor(-.2),actor(.2)];native.bread_id=[0,6]
        info = readonly_instruction_metadata(task, native)['info']
        descriptions=json.loads((FIXTURES/(task+'.json')).read_text())
        for kind in ('seen','unseen'):
            assert descriptions[kind]
            for text in descriptions[kind]:
                assert set(re.findall(r'\{[A-Za-z]\}', text)) <= info.keys()
        if task == 'place_bread_basket':
            native.bread=native.bread[:1];native.bread_id=native.bread_id[:1]
            single=readonly_instruction_metadata(task,native)['info']
            assert '{C}' not in single
            assert any(set(re.findall(r'\{[A-Za-z]\}',text))<=single.keys() for text in descriptions['unseen'])


@pytest.mark.parametrize('positions,ids', [([],[]),([-.2],[0,1]),([float('nan')],[0]),([-.2],[2]),([0.],[0]),([0.,.2],[0,1])])
def test_missing_or_invalid_scene_fields_fail_closed(monkeypatch, positions, ids):
    native=environment('place_bread_basket',monkeypatch)
    native.basket_id=0;native.bread=[actor(x) for x in positions];native.bread_id=ids
    with pytest.raises(ValueError):readonly_instruction_metadata('place_bread_basket',native)


def test_source_drift_is_rejected_and_unknown_tasks_do_not_call_get_info(monkeypatch,tmp_path):
    native=environment('blocks_ranking_size',monkeypatch)
    source=tmp_path/'changed.py';source.write_text('# unreviewed native revision\n')
    monkeypatch.setattr(sys.modules[type(native).__module__],'__file__',str(source))
    with pytest.raises(ValueError,match='source changed'):readonly_instruction_metadata('blocks_ranking_size',native)
    assert readonly_instruction_metadata('unreviewed_task',native) is None
