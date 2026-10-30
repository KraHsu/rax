"""Failed or abandoned holdout exposure is permanently excluded next round."""
import copy

import pytest

from services.robotwin.collection import atomic_json
from services.robotwin.constraint_refinement import DEFAULT_PROTOCOL
from services.robotwin.refinement_exposure import consume_holdout, require_active_reservation, reserve_holdout, verify_reservation
from test_robotwin_constraint_refinement import inputs


def reserve(tmp_path, name='first', seeds=range(100, 129)):
    sealed, release, split, dev, rows = inputs(seeds)
    proposal = {k: v for k, v in sealed.items() if k not in ('split_plan_sha256', 'frozen_unix', 'annotation_policy_sha256')}
    proposal['state'] = 'revision_proposed'
    registry = tmp_path/'registry'
    atomic_json(registry/'refinement_requirement.json', dict(version=1, protocol=DEFAULT_PROTOCOL))
    root = tmp_path/name
    sealed, release = reserve_holdout(registry, root, proposal, split, dev, frozen_unix=10)
    return registry, root, sealed, release, rows


def test_final_release_requires_registered_lineage(tmp_path):
    registry, root, _, release, rows = reserve(tmp_path)
    require_active_reservation(release)
    assert verify_reservation(root, release, rows, expected_registry=registry)['prior_revisions'] == 0
    bad = copy.deepcopy(release); bad.pop('exposure_reservation')
    with pytest.raises(ValueError, match='reservation'):
        verify_reservation(root, bad, rows)
    with pytest.raises(ValueError, match='another registry'):
        verify_reservation(root, release, rows, expected_registry=tmp_path/'new_registry')
    with pytest.raises(ValueError, match='previous holdout'):
        reserve(tmp_path, 'second', range(200, 229))


def test_failed_round_burns_all_reserved_seeds_and_observed_scenes(tmp_path):
    registry, root, _, release, rows = reserve(tmp_path)
    atomic_json(root/'holdout_campaigns.json', dict(version=1, campaigns=[str(tmp_path/'campaign')]))
    def verified(_):
        return dict(manifest=dict(scene_sha256='known-scene'), manifest_sha256='a'*64,
                    records_sha256='b'*64, trials=rows[:1], complete=False)
    consume_holdout(root, verify_campaign=verified)
    with pytest.raises(ValueError, match='Consumed'):
        require_active_reservation(release)
    assert verify_reservation(root, release, rows)['prior_revisions'] == 0
    with pytest.raises(ValueError, match='earlier revision'):
        reserve(tmp_path, 'reused', range(100, 129))
    registry, fresh, _, new_release, new_rows = reserve(tmp_path, 'fresh', range(200, 229))
    assert verify_reservation(fresh, new_release, new_rows)['prior_revisions'] == 1
    new_rows[0]['scene_sha256'] = 'known-scene'
    with pytest.raises(ValueError, match='earlier revision'):
        verify_reservation(fresh, new_release, new_rows)


def test_exposure_receipts_cannot_be_reset_or_rewritten(tmp_path):
    _, root, _, _, rows = reserve(tmp_path)
    atomic_json(root/'holdout_campaigns.json', dict(version=1, campaigns=[str(tmp_path/'campaign')]))
    def verified(_):
        return dict(manifest=dict(scene_sha256='known-scene'), manifest_sha256='a'*64,
                    records_sha256='b'*64, trials=rows[:1])
    consume_holdout(root, verify_campaign=verified)
    atomic_json(root/'exposure_receipt.json', {})
    with pytest.raises(ValueError, match='receipt changed'):
        reserve(tmp_path, 'fresh', range(200, 229))
