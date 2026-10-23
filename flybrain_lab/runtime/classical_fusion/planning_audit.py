"""Offline geometry and collision-probe audit; GT never enters the mapper."""
import time
import json
import numpy as np
import torch
from codex_zbot.tasks.shape_geometry import signed_distance_to_primitives


def audit_snapshot(mapper,source,seed,queries=20000,artifact_dir=None):
    before=time.perf_counter()
    snapshot=mapper.planning_snapshot()
    build_ms=(time.perf_counter()-before)*1000
    rng=np.random.default_rng(seed)
    lower=mapper.lower.cpu().numpy(); upper=lower+mapper.spacing.cpu().numpy()*np.asarray(mapper.shape)
    centers=source['object_centers']; sizes=source['object_sizes']
    quaternion=source['object_quaternions_wxyz']
    ids=source['object_shape_ids'] if 'object_shape_ids' in source else np.zeros(len(centers),dtype=np.int32)
    uniform=rng.uniform(lower,upper,size=(queries//2,3))
    truth=source['gt_object_points'] if 'gt_object_points' in source else source['gt_points']
    near=truth[rng.integers(len(truth),size=queries//2)]+rng.normal(0,.015,size=(queries//2,3))
    points=np.concatenate([uniform,near]).astype(np.float32)
    distance=signed_distance_to_primitives(torch.from_numpy(points),torch.from_numpy(centers.copy()),
        torch.from_numpy(sizes.copy()),torch.from_numpy(quaternion.copy()),ids).numpy()
    before=time.perf_counter()
    # Frozen-state geometry test, NOT a claim about live snapshot freshness.
    clearance,valid=snapshot.clearance(points,now_s=snapshot.built_at)
    query_ms=(time.perf_counter()-before)*1000
    result={'free_evidence_fraction':snapshot.free_fraction,'snapshot_build_ms':build_ms,
            'query_batch_ms':query_ms,'query_count':len(points),'independent_evidence_views':len(mapper.evidence_poses)}
    for radius in (0.,.02):
        label='point' if radius==0 else 'sphere_20mm'
        collision=distance<=radius
        allowed=valid&(clearance>radius)
        missed=int((allowed&collision).sum())
        result[label+'_collision_probes']=int(collision.sum())
        result[label+'_missed_collisions']=missed
        result[label+'_false_free_rate']=float(missed/max(int(collision.sum()),1))
        result[label+'_accepted_fraction']=float(allowed.mean())
        result[label+'_true_free_acceptance']=float((allowed&~collision).sum()/max(int((~collision).sum()),1))
    # Demonstrate a nontrivial continuous tool-sphere sweep in the measured
    # free region. It is not a full-robot trajectory or an actuator command.
    usable=np.flatnonzero(valid[:queries//2]&(clearance[:queries//2]>.03))
    result['example_sweep_length_m']=0.
    result['example_sweep_minimum_slack_m']=None
    if len(usable)>1:
        for _ in range(100):
            a,b=points[rng.choice(usable,size=2,replace=False)]
            length=float(np.linalg.norm(b-a))
            if not .10<=length<=.40:
                continue
            check=snapshot.check_swept_spheres(a[None],b[None],np.asarray([.02]),
                                              now_s=snapshot.built_at)
            if check.allowed:
                result['example_sweep_length_m']=length
                result['example_sweep_minimum_slack_m']=check.minimum_clearance_m
                if artifact_dir is not None:
                    (artifact_dir/'example_sweep.json').write_text(json.dumps(
                        {'start_world_m':a.tolist(),'end_world_m':b.tolist(),'sphere_radius_m':.02,
                         'length_m':length,'minimum_slack_m':check.minimum_clearance_m,
                         'tested_samples':check.tested_samples,
                         'scope':'frozen-map straight tool sphere only; NOT a full-robot path or execution approval'},indent=2))
                break
    return result
