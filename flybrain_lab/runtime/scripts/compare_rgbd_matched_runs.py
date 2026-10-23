"""Matched nominal control comparison; no significance from one pair."""
import argparse,json
from pathlib import Path
import numpy as np


def compare(set_run,pose_run,output):
    roots=[Path(set_run),Path(pose_run)]
    def read(root,name):return json.loads((root/name).read_text(encoding='utf-8'))
    results=[read(p,'result.json') for p in roots];configs=[read(p,'run_configuration.json') for p in roots]
    manifests=[read(p,'source_manifest.json') for p in roots];audits=[read(p,'transfer_audit.json') for p in roots]
    checks=dict(same_source=manifests[0]==manifests[1],source_unchanged=all(r.get('source_unchanged_during_run') for r in results),
        same_scan=configs[0]['initial_scan']==configs[1]['initial_scan'],same_scene=configs[0]['scene']==configs[1]['scene'],
        same_initial_joints=bool(np.allclose(results[0]['initial_joints'],results[1]['initial_joints'],atol=1e-12,rtol=0)),
        modes=[r['mode'] for r in results]==['set','pose'],same_task_request=read(roots[0],'task_request.json')==read(roots[1],'task_request.json'))
    if not all(checks.values()):raise ValueError('not a matched pair: '+str(checks))
    rows=[]
    for kind in ('force','torque'):
        for metric in ('rms','peak','accumulated_magnitude','maximum_window_impulse'):
            a,b=[r['reaction'][kind][metric] for r in results]
            rows.append(dict(metric=kind+'_'+metric,set=a,pose=b,relative_change_percent=100*(a-b)/b))
    for key in ('simulation_time_s','cycle_p99_ms'):
        a,b=[r[key] for r in results];rows.append(dict(metric=key,set=a,pose=b,relative_change_percent=100*(a-b)/b))
    out=dict(scope='One matched development scene; not a randomized population estimate',checks=checks,
        runs=[str(p) for p in roots],both_released_and_retreated=all(a.get('post_release_retreat_audited') for a in audits),
        comparison=rows,online_frames_matched=False,
        warning='Same initial scan and constraints; online images/IK/segment durations differ as outcomes. No all-metrics superiority or statistical significance inferred.')
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# Matched nominal RGB-D control comparison','',out['scope'], '',
        '| Metric | Set | Pose | Change, set vs pose |','|---|---:|---:|---:|']
    lines += [f"| {r['metric']} | {r['set']:.6g} | {r['pose']:.6g} | {r['relative_change_percent']:+.2f}% |" for r in rows]
    lines += ['',out['warning']]
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--set-run',required=True);p.add_argument('--pose-run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();compare(a.set_run,a.pose_run,a.output)
