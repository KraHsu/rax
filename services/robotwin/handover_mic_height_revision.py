"""Explicit CPU migration for one audited template, never automatic deduplication.

This produces a candidate only. It neither edits a dataset nor changes registry
status. The old template remains valid and no general remove-rule API is added.
"""
from copy import deepcopy
import math

from .constraint_coverage import active_range, chunk_candidate, coverage_report
from .constraint_schema import validate_bundle
from .programmatic_annotation import instantiate, template_hash, validate_template

VERSION = 'handover_mic_reference_height_once_v1'
PARENT_SHA256 = 'e3739d94cec71aa07a3ca76ae24fd40249547cfe352d0b2c57787cc71189ed38'
REMOVED_RULE = 'release_vertical_rise'
RETAINED_RULE = 'release_height'


class HeightRevisionError(ValueError):
    """The narrow migration contract has not been proved for this input."""


def candidate_template(parent):
    """Build a new, inactive data-only candidate from the exact audited parent."""
    validate_template(parent)
    if parent['task'] != 'handover_mic' or template_hash(parent) != PARENT_SHA256:
        raise HeightRevisionError('Only the explicitly audited handover_mic parent is supported')
    revised = deepcopy(parent)
    revised['rules'] = [r for r in revised['rules'] if r['id'] != REMOVED_RULE]
    if len(revised['rules']) != len(parent['rules'])-1:
        raise HeightRevisionError('Expected exactly one named rise rule')
    revised['rationale'] = (
        'Keep the original closure locations, closed/released gripper intervals and release height. '
        'CPU migration handover_mic_reference_height_once_v1 removes the named closure-to-release '
        'rise rule only after every audited reference proves it was reanchored to the same fixed '
        'world-Z interval at the same event. This counts that height feature once. It changes '
        'relative weighted-mean loss contributions, not the demonstrated physical allowed set. '
        'This candidate is not activated, physically calibrated, or a task-success certificate.'
    )
    validate_template(revised)
    return revised


def _physics(constraint):
    return {key: deepcopy(value) for key, value in constraint.items() if key != 'evidence'}


def verify_compiled_pair(rise, height, total_frames):
    """Prove this fixed-reference Z/event pair has the same physical set/masks.

    This intentionally rejects rotations, local points, other primitives, short
    predicted-start spans, unequal penalties or even nonidentical float bounds.
    It does not decide that arbitrary nearby features are interchangeable.
    """
    if type(total_frames) is not int or total_frames < 2:
        raise HeightRevisionError('Invalid complete-reference length')
    if (rise['type'] != 'axis_displacement_interval' or height['type'] != 'axis_position_interval'
            or rise['subject'] != height['subject'] or rise['subject'] not in ('left_eef', 'right_eef')
            or rise['evidence']['status'] != 'accepted' or height['evidence']['status'] != 'accepted'):
        raise HeightRevisionError('Unsupported feature/subject/evidence pair')
    rp, hp = rise['parameters'], height['parameters']
    if (set(rp) != {'axis', 'bounds', 'anchor'} or set(hp) != {'axis', 'bounds'}
            or rp['anchor'] != 'reference_frame' or rp['axis'] != [0., 0., 1.]
            or hp['axis'] != [0., 0., 1.]):
        raise HeightRevisionError('Only fixed-reference world-Z features without local offsets are equivalent here')
    for c in (rise, height):
        frame = c['reference_frame']
        if (frame['quaternion_wxyz'] != [1., 0., 0., 0.] or len(frame['origin_m']) != 3
                or not all(math.isfinite(x) for x in frame['origin_m']+c['parameters']['bounds'])):
            raise HeightRevisionError('Expected finite world-oriented frames')
    a, b = rise['activation'], height['activation']
    if (a['mode'] != 'reference_event' or b['mode'] != 'reference_event' or a['event'] != b['event']
            or a['event']['direction'] != 'increasing' or a['event']['threshold'] != .5):
        raise HeightRevisionError('Event identity differs')
    event = a['event']['frame']
    if (type(event) is not int or not 1 <= event < total_frames
            or any(not 0 <= c['activation']['start'] <= event-1
                   or c['activation']['end'] != event+1 for c in (rise, height))):
        raise HeightRevisionError('Unexpected reference event or activation range')
    bounds = [[v+c['reference_frame']['origin_m'][2] for v in c['parameters']['bounds']]
              for c in (rise, height)]
    if bounds[0] != bounds[1] or bounds[0][0] > bounds[0][1]:
        raise HeightRevisionError('Absolute world-Z allowed intervals are not exactly equal')
    if rise['penalty'] != height['penalty']:
        raise HeightRevisionError('Declared penalty/weight differs; no implicit reweighting claim')
    if active_range(rise) != active_range(height):
        raise HeightRevisionError('Effective activation differs')
    # A reference-frame displacement never needs a generated closure anchor.
    # The fast reader filter must also preserve equality in every 32-action
    # window, including event-at-row0 and truncated final windows.
    starts = []
    for start in range(total_frames):
        horizon = min(32, total_frames-1-start)
        left, right = chunk_candidate(rise, start, horizon), chunk_candidate(height, start, horizon)
        if left != right:
            raise HeightRevisionError('Chunk prefilter differs')
        if left:
            starts.append(start)
    return dict(removed_id=rise['id'], retained_id=height['id'], subject=rise['subject'],
                event=deepcopy(a['event']), effective_activation=list(active_range(rise)),
                world_z_bounds_m=bounds[0], original_activation_starts=[a['start'], b['start']],
                anchor='reference_frame', total_frames=total_frames, horizon=32,
                eligible_chunk_starts=starts, penalty=deepcopy(height['penalty']),
                position_validity_dims_equal=True, reference_event_gripper_validity_equal=True,
                row0_is_measured_not_generated=True)


def audit_reference(handle, parent, *, reference_id):
    """Compile both versions and verify every retained physical rule unchanged."""
    revised = candidate_template(parent)
    old, old_audit = instantiate(handle, parent)
    new, new_audit = instantiate(handle, revised)
    rises = [c for c in old if c['id'].startswith(REMOVED_RULE+'_')]
    heights = [c for c in old if c['id'].startswith(RETAINED_RULE+'_')]
    if len(rises) != len(heights) or len(rises) != 1:
        raise HeightRevisionError('Exactly one releasing-arm height pair is required')
    pair = verify_compiled_pair(rises[0], heights[0], len(handle['endpose/left_endpose']))
    keep = [c for c in old if c['id'] != pair['removed_id']]
    if [_physics(c) for c in keep] != [_physics(c) for c in new]:
        raise HeightRevisionError('Another compiled physical rule or automatic motion guard changed')
    if old_audit['coverage'] != new_audit['coverage']:
        raise HeightRevisionError('Temporal coverage changed')
    return revised, dict(version=VERSION, reference_id=reference_id, parent_sha256=template_hash(parent),
        candidate_sha256=template_hash(revised), removed_to_retained=pair,
        old_constraints_sha256=template_hash(old), new_constraints_sha256=template_hash(new),
        retained_physical_rules_identical=True, automatic_motion_guards_identical=True,
        coverage_identical=True, candidate_only=True, activated=False,
        objective_change='One repeated feature weight is removed from numerator and denominator. '
                         'The weighted-mean loss is intentionally not asserted equal; per-kind scale overrides must be reported separately.')


def candidate_embedded_bundle(bundle, total_frames, *, parent):
    """Remove only the proven pair member from an existing bundle in memory.

    Historical embedded motion guards may predate the current compiler. Preserve
    them verbatim; never conflate this narrowly scoped data revision with full
    reannotation using the current candidate template. No HDF5 writes occur here.
    """
    candidate_template(parent)  # Restrict this operation to the named migration.
    validate_bundle(bundle, total_frames)
    cs = bundle['constraints']
    rise = [c for c in cs if c['id'].startswith(REMOVED_RULE+'_')]
    height = [c for c in cs if c['id'].startswith(RETAINED_RULE+'_')]
    if len(rise) != 1 or len(height) != 1:
        raise HeightRevisionError('Existing bundle must contain exactly the named pair')
    pair = verify_compiled_pair(rise[0], height[0], total_frames)
    revised = deepcopy(bundle)
    revised['constraints'] = [c for c in revised['constraints'] if c['id'] != pair['removed_id']]
    validate_bundle(revised, total_frames)
    if coverage_report(cs, total_frames) != coverage_report(revised['constraints'], total_frames):
        raise HeightRevisionError('Existing bundle temporal coverage changed')
    return revised, dict(version=VERSION, original_bundle_sha256=template_hash(bundle),
        candidate_bundle_sha256=template_hash(revised), removed_to_retained=pair,
        all_other_embedded_fields_identical=True, legacy_motion_guards_preserved=True,
        scope='in_memory_candidate_only_not_current_compiler_reannotation', activated=False)
