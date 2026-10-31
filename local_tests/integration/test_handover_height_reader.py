"""Actual HDF5 recorder -> RoboTwin reader -> decoder -> objective, CPU only.

Use the documented reader harness to avoid unrelated optional dataset imports.
The fixture is explicitly synthetic; it is not a new native successful episode.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.robotwin.handover_mic_height_revision import audit_reference
from services.robotwin.programmatic_annotation import instantiate
from services.robotwin.recording import EpisodeRecorder
from test_robotwin_handover_height_revision import parent, reference
from openwam.dataloader.robotwin import RoboTwinDataset
from openwam.train.constraint_supervision import ActionDecoder, RoboTwinConstraintObjective
from openwam.train.meta_constraints import residual


@pytest.fixture(scope='module')
def recorded(tmp_path_factory):
    root = tmp_path_factory.mktemp('height-reader')
    h = reference(); p = parent(); revised, audit = audit_reference(h, p, reference_id='synthetic')
    pixels = np.full((16, 16, 3), 100, dtype=np.uint8)
    for name, template in [('old', p), ('new', revised)]:
        path = root/name/'handover_mic/aloha-agilex_clean_50/data/episode0.hdf5'
        cs, _ = instantiate(h, template)
        recorder = EpisodeRecorder(path, recording_id='synthetic-duplicate-height', instruction='Synthetic CPU handover fixture')
        for c in cs:
            recorder.upsert(c, recorder.revision)
        for i in range(220):
            recorder.append(dict(endpose={key: h['endpose/'+key][i] for key in
                ['left_endpose', 'right_endpose', 'left_gripper', 'right_gripper']},
                joint_action=dict(vector=np.zeros(14)), images={cam: pixels for cam in
                ['head_camera', 'left_camera', 'right_camera']}))
        recorder.finish(success=True)
    stats = root/'stats.npy'
    np.save(stats, {'eef': {'mean': np.zeros(20), 'min': np.full(20, -2.), 'max': np.full(20, 3.)}})
    return root, stats, audit


def reader(root, version, stats, *, normalized=False, unified=False):
    indices = np.r_[0:10, 34:44] if unified else np.arange(20)
    ds = RoboTwinDataset(data_root=str(root/version/'handover_mic/aloha-agilex_clean_50/data'),
        num_frames=33, video_stride=8, height=32, width=32, action_mode='eef', variant='clean_50',
        normalize_mode='min-max' if normalized else None,
        normalization_stats_path=str(stats) if normalized else None,
        unify_action=unified, unify_action_map=indices.tolist() if unified else None)
    return ds, indices


@pytest.mark.parametrize('normalized,unified', [(False, False), (False, True), (True, True)])
def test_32_chunk_event_edges_tail_padding_gradients_and_measured_anchor(recorded, normalized, unified, tmp_path):
    root, stats, audit = recorded
    ds, indices = reader(root, 'old', stats, normalized=normalized, unified=unified)
    decoder = ActionDecoder.from_reader(ds)
    # Event is world frame190. Row0 is measured; tail windows keep only real actions.
    for start in [157, 158, 159, 175, 189, 190, 218]:
        sample = ds[ds._window_index.index((0, start))]
        assert sample['action'].shape == (32, 80 if unified else 20)
        cs = sample['trajectory_constraints']['constraints']
        a, b = [next(c for c in cs if c['id'].startswith(prefix)) for prefix in
                ['release_vertical_rise_', 'release_height_']]
        pred = sample['action'].clone()
        event_index = 190-start
        if 1 <= event_index <= 32:
            # perturb ONLY generated event Z by0.03 physical metres, in model coordinates
            pred[event_index-1, indices[2]] += .03*decoder.divisor[2]
        pred.requires_grad_()
        sample['proprio'].requires_grad_(); sample['action'].requires_grad_()
        physical = decoder.decode(torch.cat((sample['proprio'].detach(), pred)))
        ref = decoder.decode(torch.cat((sample['proprio'].detach(), sample['action'].detach())))
        valid = torch.cat((sample['proprio_mask'], sample['action_mask']))[:, decoder.indices]
        values = [residual(physical, ref, valid, c, start_frame=start,
                          rotation_convention='robotwin_wxyz_read_as_xyzw') for c in [a, b]]
        expected = start in audit['removed_to_retained']['eligible_chunk_starts']
        assert (values[0] is not None) == expected == (values[1] is not None)
        if expected:
            torch.testing.assert_close(values[0], values[1], atol=2e-7, rtol=2e-6)
            grads = [torch.autograd.grad(v.sum(), pred, retain_graph=True)[0] for v in values]
            torch.testing.assert_close(grads[0], grads[1], atol=1e-7, rtol=1e-6)
            assert grads[0].count_nonzero() == 1
            assert grads[0][event_index-1, indices[2]] > 0
            # Either position or either adjacent reference gripper invalidates BOTH.
            for row, dim in [(event_index, 2), (event_index, 9), (event_index-1, 9)]:
                masked = valid.clone(); masked[row, dim] = False
                assert all(residual(physical, ref, masked, c, start_frame=start,
                                   rotation_convention='robotwin_wxyz_read_as_xyzw') is None for c in [a, b])
        assert sample['proprio'].grad is None and sample['action'].grad is None
        if start == 218:
            assert sample['action_mask'][:, indices].all(-1).sum() == 1
    assert 190 not in audit['removed_to_retained']['eligible_chunk_starts']


@pytest.mark.parametrize('overrides', [{}, {'axis_displacement_interval': .01, 'axis_position_interval': .02}])
@pytest.mark.parametrize('height_shift', [0., .03])
def test_real_reader_objective_removes_exact_feature_and_weight_not_other_terms(recorded, tmp_path, overrides, height_shift):
    root, stats, _ = recorded
    results = []
    for version in ['old', 'new']:
        ds, _ = reader(root, version, stats)
        sample = ds[ds._window_index.index((0, 175))]
        sample['action'].requires_grad_(); sample['proprio'].requires_grad_()
        pred = sample['action'].detach().clone(); pred[14, 2] += height_shift; pred[14, 10] += .04; pred.requires_grad_()
        objective = RoboTwinConstraintObjective(ds, root/version, weight=1., penalty_scale_overrides=overrides)
        result = objective.bind([sample])(pred[None]); result['loss'].backward()
        assert torch.isfinite(pred.grad).all() and sample['action'].grad is None and sample['proprio'].grad is None
        metrics = result['metrics']
        numerator = sum(metrics[k] for k in metrics if k.endswith('_raw_penalty_sum'))
        denominator = metrics['constraint_weight_sum']
        torch.testing.assert_close(result['loss'], numerator/denominator)
        results.append((result, numerator, denominator))
    old, new = results
    removed = old[0]['metrics']['axis_displacement_interval_raw_penalty_sum']
    assert (removed > 0) == (height_shift > 0)
    torch.testing.assert_close(old[1]-new[1], removed)
    torch.testing.assert_close(old[2]-new[2], torch.tensor(1.))
    for name, value in new[0]['metrics'].items():
        if name.endswith('_raw_penalty_sum') and not name.startswith('axis_displacement_interval_'):
            torch.testing.assert_close(old[0]['metrics'][name], value)
    assert not torch.isclose(old[0]['loss'], new[0]['loss'])


def minimal_payload(root, version, rank):
    """Unequal rank-local active weights exercise real DDP global denominator."""
    from openwam.train.constraint_schema import SCHEMA_VERSION
    base = dict(subject='left_eef', reference_frame=dict(origin_m=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.]),
        parameters=dict(axis=[0.,0.,1.],bounds=[1.,1.1]),
        activation=dict(start=0,end=3,mode='reference_event',event=dict(frame=2,direction='increasing',threshold=.5)),
        penalty=dict(kind='squared_hinge',scale=.1,weight=1.),
        evidence=dict(source='manual',status='accepted',report_id='synthetic',rationale='CPU denominator regression'))
    height=dict(copy.deepcopy(base),id='height',type='axis_position_interval')
    rise=dict(copy.deepcopy(base),id='rise',type='axis_displacement_interval');rise['parameters']['anchor']='reference_frame'
    other=dict(copy.deepcopy(base),id='other',type='axis_position_interval',subject='right_eef')
    other['activation']=dict(start=0,end=8,mode='interval');other['parameters']=dict(axis=[1.,0.,0.],bounds=[0.,.1])
    cs=([rise] if version=='old' else [])+[height,other]
    b=dict(schema_version=SCHEMA_VERSION,action_contract=dict(position='absolute',rotation='absolute',quaternion_order='wxyz',
        gripper='absolute_open_fraction',frame='world'),provenance=dict(source='synthetic',split='train',variant='clean_50',recording_id='ddp'),constraints=cs)
    episode=str(Path(root)/'ep.hdf5')
    ds=SimpleNamespace(action_mode='eef',normalize_mode=None,_normalizer=None,_unify_dst_index=None,variant='clean_50',split='train',
                       _episode_files=[episode],_episode_lengths=[8],_constraint_bundles={episode:b})
    ref=torch.zeros(3,20);ref[:,[3,7,13,17]]=1.;ref[:,2]=1.;ref[:,19]=1.;ref[:,9]=torch.tensor([0.,0.,1.])
    if rank: ref[:,9]=1.
    sample=dict(episode_path=episode,start_frame=3 if rank else 0,proprio=ref[:1],action=ref[1:],
                proprio_mask=torch.ones_like(ref[:1],dtype=torch.bool),action_mask=torch.ones_like(ref[1:],dtype=torch.bool),trajectory_constraints=b)
    return ds,sample


class ScalarAction(torch.nn.Module):
    def __init__(self):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(.2))
    def forward(self, sample):
        base=sample['action'].clone(); mask=torch.zeros_like(base);mask[:,2]=1.;mask[:,10]=2.
        return (base+self.value*mask)[None]


def ddp_worker(rank, init_file, output, version):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    try:
        ds,sample=minimal_payload(output,version,rank)
        model=torch.nn.parallel.DistributedDataParallel(ScalarAction())
        objective=RoboTwinConstraintObjective(ds,output,weight=1.,require_full_coverage=False)
        result=objective.bind([sample])(model(sample));result['loss'].backward()
        Path(output,f'{version}-{rank}.json').write_text(json.dumps(dict(loss=result['loss'].item(),
            gradient=model.module.value.grad.item(),local_weight=result['metrics']['constraint_weight_sum'].item())))
    finally:
        torch.distributed.destroy_process_group()


def test_actual_two_rank_cpu_gloo_matches_pooled_batch_after_revision(tmp_path):
    if not torch.distributed.is_gloo_available():pytest.skip('CPU Gloo unavailable')
    torch.set_num_threads(1)
    for version, weights in [('old',[3.,1.]),('new',[2.,1.])]:
        torch.multiprocessing.start_processes(ddp_worker,args=(str(tmp_path/f'init-{version}'),str(tmp_path),version),
                                             nprocs=2,join=True,start_method='fork')
        rows=[json.loads((tmp_path/f'{version}-{rank}.json').read_text()) for rank in range(2)]
        ds,s0=minimal_payload(tmp_path,version,0);_,s1=minimal_payload(tmp_path,version,1)
        model=ScalarAction();pred=torch.cat([model(s0),model(s1)])
        result=RoboTwinConstraintObjective(ds,tmp_path,weight=1.,require_full_coverage=False).bind([s0,s1])(pred)
        result['loss'].backward()
        assert [r['local_weight'] for r in rows]==weights
        assert sum(r['loss'] for r in rows)/2==pytest.approx(result['loss'].item(),rel=1e-6)
        assert all(r['gradient']==pytest.approx(model.value.grad.item(),rel=1e-6) for r in rows)
