"""Registry-owned holdout reservations: consumed scenes cannot be reused."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from services.robotwin.acceptance import file_hash, json_hash
from services.robotwin.collection import atomic_json


def read(path):
    import json
    return json.loads(Path(path).read_text())


@contextmanager
def locked(registry, task):
    if not isinstance(task, str) or Path(task).name != task or task in ('.', '..'):
        raise ValueError('Invalid task for exposure ledger')
    path = Path(registry).resolve()/'refinement_exposure'/(task+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = read(path) if path.exists() else dict(version=1, task=task, reservations=[])
        if set(ledger) != {'version', 'task', 'reservations'} or ledger['version'] != 1 or ledger['task'] != task:
            raise ValueError('Invalid shared exposure ledger')
        yield path, ledger


def reserve_holdout(registry, revision_root, revision, split_plan, development, *, frozen_unix=None):
    """Reserve fresh seeds before issuing a runnable final-holdout release.

    Prior rounds must be closed before reserving another. Closing a failed round
    consumes its full seed allocation, not just successful or completed trials.
    """
    from services.robotwin.constraint_refinement import seal_revision
    registry, root = Path(registry).resolve(), Path(revision_root).resolve()
    requirement = read(registry/'refinement_requirement.json')
    if requirement != dict(version=1, protocol=revision['protocol']):
        raise ValueError('Revision differs from the registry-required protocol')
    sealed, release = seal_revision(revision, split_plan, development, frozen_unix=frozen_unix)
    with locked(registry, revision['task']) as (path, ledger):
        if root.exists():
            raise ValueError('Reserve a new immutable revision directory')
        if any(r['state'] != 'consumed' for r in ledger['reservations']):
            raise ValueError('Finish or explicitly consume the previous holdout before reserving another')
        exposed_seeds = {seed for r in ledger['reservations'] for seed in r['holdout_seeds']}
        if exposed_seeds & set(split_plan['holdout_seeds']):
            raise ValueError('Final holdout seeds were exposed/reserved by an earlier revision')
        for entry in ledger['reservations']:
            if file_hash(Path(entry['revision_root'])/'exposure_receipt.json') != entry['exposure_receipt_sha256']:
                raise ValueError('Historical exposure receipt changed')
        reservation = dict(registry_root=str(registry), task=revision['task'],
                           reservation_id=json_hash(dict(release=release, registry_root=str(registry))),
                           prior_history_sha256=json_hash(ledger['reservations']))
        release['exposure_reservation'] = reservation
        root.mkdir(parents=True)
        for name, value in [('sealed_revision', sealed), ('holdout_release', release),
                            ('split_plan', split_plan), ('development_report', development)]:
            atomic_json(root/(name+'.json'), value)
        ledger['reservations'].append(dict(reservation_id=reservation['reservation_id'], state='reserved',
                                           revision_root=str(root), release_sha256=json_hash(release),
                                           holdout_seeds=split_plan['holdout_seeds']))
        atomic_json(path, ledger)
    return sealed, release


def verify_reservation(revision_root, release, rows, *, expected_registry=None):
    receipt = release.get('exposure_reservation')
    if not isinstance(receipt, dict) or set(receipt) != {
            'registry_root', 'task', 'reservation_id', 'prior_history_sha256'}:
        raise ValueError('Final holdout release lacks a registry-owned exposure reservation')
    registry = Path(receipt['registry_root']).resolve()
    if expected_registry is not None and registry != Path(expected_registry).resolve():
        raise ValueError('Holdout reservation belongs to another registry')
    ledger = read(registry/'refinement_exposure'/(release['task']+'.json'))
    indices = [i for i, r in enumerate(ledger['reservations']) if r['reservation_id'] == receipt['reservation_id']]
    if len(indices) != 1:
        raise ValueError('Holdout reservation is missing or duplicated')
    index = indices[0]
    prior, current = ledger['reservations'][:index], ledger['reservations'][index]
    if (receipt['task'] != release['task'] or json_hash(prior) != receipt['prior_history_sha256']
            or current['release_sha256'] != json_hash(release)
            or Path(current['revision_root']).resolve() != Path(revision_root).resolve()):
        raise ValueError('Registered holdout lineage or frozen release changed')
    seeds, scenes = set(), set()
    for entry in prior:
        path = Path(entry['revision_root'])/'exposure_receipt.json'
        if entry['state'] != 'consumed' or file_hash(path) != entry['exposure_receipt_sha256']:
            raise ValueError('Earlier holdout exposure is incomplete or changed')
        value = read(path)
        seeds.update(entry['holdout_seeds']); scenes.update(value['scene_sha256'])
    if any(r['scene_seed'] in seeds or r['scene_sha256'] in scenes for r in rows):
        raise ValueError('Final holdout reuses a scene exposed by an earlier revision')
    return dict(registry_root=str(registry), reservation_id=receipt['reservation_id'],
                prior_revisions=len(prior), prior_exposed_seed_count=len(seeds), prior_exposed_scene_count=len(scenes))


def require_active_reservation(release):
    receipt = release.get('exposure_reservation') if isinstance(release, dict) else None
    if not isinstance(receipt, dict):
        raise ValueError('Final holdout execution requires a registry-owned exposure reservation')
    ledger = read(Path(receipt['registry_root'])/'refinement_exposure'/(release['task']+'.json'))
    matches = [r for r in ledger['reservations'] if r['reservation_id'] == receipt['reservation_id']]
    if len(matches) != 1:
        raise ValueError('Final holdout reservation missing or duplicated')
    entry = matches[0]
    verify_reservation(entry['revision_root'], release, [])
    if entry['state'] != 'reserved':
        raise ValueError('Consumed final holdout cannot collect or resume native trials')


def consume_holdout(revision_root, *, verify_campaign=None):
    """Close an evaluated or deliberately abandoned round without approving it.

    Every declared seed is burned. Verified manifests capture scene hashes even
    when native execution was incomplete; later collection must use a new round.
    No status in this ledger is an acceptance decision.
    """
    if verify_campaign is None:
        from scripts.search_robotwin_constraint_counterexamples import verify_campaign
    root = Path(revision_root).resolve()
    release = read(root/'holdout_release.json')
    inventory = read(root/'holdout_campaigns.json')
    for campaign in inventory['campaigns']:
        state = Path(campaign)/'run.json'
        if state.exists() and read(state).get('status') == 'running_trials':
            raise ValueError('Stop native collection before consuming a holdout round')
    checked = [verify_campaign(Path(p)) for p in inventory['campaigns']]
    rows = [r for campaign in checked for r in campaign['trials']]
    verify_reservation(root, release, rows)
    reservation = release['exposure_reservation']
    value = dict(version=1, release_sha256=json_hash(release),
                 scene_sha256=sorted({c['manifest']['scene_sha256'] for c in checked}),
                 campaigns=[dict(path=str(Path(p).resolve()), manifest_sha256=c['manifest_sha256'],
                                  records_sha256=c['records_sha256'])
                            for p, c in zip(read(root/'holdout_campaigns.json')['campaigns'], checked)],
                 purpose='exposure_only_not_acceptance')
    with locked(reservation['registry_root'], release['task']) as (path, ledger):
        entry = next(r for r in ledger['reservations'] if r['reservation_id'] == reservation['reservation_id'])
        if entry['state'] == 'consumed':
            if read(root/'exposure_receipt.json') != value:
                raise ValueError('Consumed holdout gained new evidence; it cannot be resumed')
            return value
        atomic_json(root/'exposure_receipt.json', value)
        entry.update(state='consumed', exposure_receipt_sha256=file_hash(root/'exposure_receipt.json'))
        atomic_json(path, ledger)
    return value
