"""CPU-only SE(3) correction tests; no simulator, model or GPU dependencies."""
from types import SimpleNamespace

import numpy as np
import pytest
import transforms3d as t3d
import yaml

from services.robotwin.acceptance_replay import install_base_frame_adapter
from services.robotwin.acceptance import validate_replay_adapter


@pytest.fixture
def native(tmp_path):
    urdf = tmp_path/'robot.urdf'
    parts = ['<robot name="test"><link name="root"/>']
    robot = SimpleNamespace(communication_flag=False, planner_backend='curobo')
    def world_to_root(base, target):
        rotation = t3d.quaternions.quat2mat(base[3:])
        return (rotation.T@(target[:3]-base[:3]),
                t3d.quaternions.mat2quat(rotation.T@t3d.quaternions.quat2mat(target[3:])))
    for arm, xyz, rpy in [('left', [.23, .3, .78], [.1, -.05, .02]),
                          ('right', [.23, -.3, .78], [-.07, .03, .01])]:
        parts.extend([f'<link name="{arm}_base"/><link name="{arm}_ee"/>',
                      f'<joint name="{arm}_fixed" type="fixed"><parent link="root"/><child link="{arm}_base"/>',
                      f'<origin xyz="{" ".join(map(str, xyz))}" rpy="{" ".join(map(str, rpy))}"/></joint>',
                      f'<joint name="{arm}_joint" type="revolute"><parent link="{arm}_base"/>',
                      f'<child link="{arm}_ee"/></joint>'])
        path = tmp_path/(arm+'.yml')
        path.write_text(yaml.safe_dump(dict(robot_cfg=dict(kinematics=dict(
            urdf_path=str(urdf), base_link=arm+'_base', ee_link=arm+'_ee')))))
        setattr(robot, arm+'_planner', SimpleNamespace(yml_path=str(path), frame_bias=-np.asarray(xyz),
                                                       _trans_from_world_to_base=world_to_root))
        setattr(robot, arm+'_urdf_path', str(urdf))
        setattr(robot, arm+'_ee_name', arm+'_joint')
    urdf.write_text(''.join(parts)+'</robot>')
    return SimpleNamespace(robot=robot)


def test_full_se3_world_roundtrip_preserves_position_orientation_and_quaternion_sign(native):
    before = native.robot.left_planner._trans_from_world_to_base
    audit = install_base_frame_adapter(native, install=False)
    assert native.robot.left_planner._trans_from_world_to_base is before
    assert install_base_frame_adapter(native) == audit
    validate_replay_adapter(audit)
    root_p = np.array([.2, -.65, .1])
    root_r = t3d.euler.euler2mat(.2, -.3, .8)
    root_pose = np.r_[root_p, t3d.quaternions.mat2quat(root_r)]
    expected_p = np.array([.4, .03, .12])
    expected_r = t3d.euler.euler2mat(-.4, .5, -.3)
    for arm in ('left', 'right'):
        proof = audit['arms'][arm]
        base_r, base_p = np.asarray(proof['rotation']), np.asarray(proof['xyz'])
        world_p = root_p+root_r@(base_p+base_r@expected_p)
        world_q = t3d.quaternions.mat2quat(root_r@base_r@expected_r)
        planner = getattr(native.robot, arm+'_planner')
        for sign in (-1, 1):
            returned_p, returned_q = planner._trans_from_world_to_base(root_pose, np.r_[world_p, sign*world_q])
            # Native plan_path adds its unchanged frame_bias after this method.
            np.testing.assert_allclose(returned_p+planner.frame_bias, expected_p, rtol=0, atol=1e-12)
            np.testing.assert_allclose(t3d.quaternions.quat2mat(returned_q), expected_r, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='already installed'):
        install_base_frame_adapter(native)


@pytest.mark.parametrize('invalid', ['communication', 'bias', 'ee_link', 'moving_base', 'different_urdf'])
def test_unsupported_models_fail_before_mutating_either_planner(native, tmp_path, invalid):
    left = native.robot.left_planner._trans_from_world_to_base
    right = native.robot.right_planner
    if invalid == 'communication':
        native.robot.communication_flag = True
    elif invalid == 'bias':
        right.frame_bias[1] += .01
    elif invalid == 'ee_link':
        native.robot.right_ee_name = 'not_the_ee_joint'
    elif invalid == 'moving_base':
        from pathlib import Path
        path = Path(native.robot.right_urdf_path)
        path.write_text(path.read_text().replace('right_fixed" type="fixed', 'right_fixed" type="revolute'))
    else:
        path = tmp_path/'other.urdf'
        path.write_text('<robot name="different"/>')
        native.robot.right_urdf_path = str(path)
    with pytest.raises(ValueError):
        install_base_frame_adapter(native)
    assert native.robot.left_planner._trans_from_world_to_base is left
    assert not getattr(native.robot.left_planner, '_rax_private_base_frame', False)
