"""Audit NEW rendered fault evidence and the actual certified-line brakes."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from astronaut_delivery.smooth_path import SmoothPath


def audit_brakes(paths,steps):
    groups=[];current=[]
    for s in steps:
        if s['phase']=='braking':current.append(s)
        elif current:groups.append(current);current=[]
    if current:groups.append(current)
    out=[]
    for group in groups:
        first=group[0];p=next(p for p in reversed(paths) if p['time_s']<first['time_s']+1e-9)
        T=p['duration_s'];original=SmoothPath(np.asarray(p['start']),np.asarray(p['end']),T,np.array([0,0,0,10/T**3,-15/T**4,6/T**5]))
        elapsed=first['time_s']-p['time_s'];brake=original.stop(elapsed)
        errors=[]
        for j,s in enumerate(group):
            expected=original.state(elapsed) if j==0 else brake.state(s['time_s']-first['time_s'])
            errors.append(max(float(np.max(abs(np.asarray(s[k])-expected[i]))) for i,k in enumerate(('q','v','a'))))
        velocity,acceleration=np.linalg.norm(group[-1]['v']),np.linalg.norm(group[-1]['a'])
        bounds=original.bounds() if brake is None else brake.bounds()
        if brake is None:scalar=[original.scalar(elapsed)]
        else:
            from astronaut_delivery.smooth_path import candidates
            scalar=brake.scalar(candidates(brake.coefficients,brake.duration))
        passed=(max(errors)<1e-6 and velocity<1e-6 and acceleration<1e-6 and min(scalar)>=-1e-8
            and max(scalar)<=1+1e-8 and np.all(np.asarray(bounds)<=[.3000001,.5000001,5.0000001]))
        out.append(dict(start_motion_s=first['time_s'],end_motion_s=group[-1]['time_s'],
            start_wall_s=first.get('monotonic_wall_s'),ticks=len(group),maximum_reconstructed_state_error=max(errors),
            terminal_velocity_norm=float(velocity),terminal_acceleration_norm=float(acceleration),
            exact_polynomial_bounds=bounds,certified_line_scalar_range=[float(min(scalar)),float(max(scalar))],passed=bool(passed)))
    return out


def audit(run):
    root=Path(run);read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    cfg=read('run_configuration.json');steps=read('steps.json');paths=read('paths.json');result=read('result.json')
    records=[json.loads(line) for line in (root/'sensor_faults/records.jsonl').read_text().splitlines() if line]
    errors=[];active=[];valid_originals=0
    for row in records:
        path=(root/'sensor_faults'/row['file']).resolve()
        if not path.is_relative_to((root/'sensor_faults').resolve()):raise ValueError('fault record path escapes run')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:errors.append('hash/'+row['file']);continue
        with np.load(path,allow_pickle=False) as z:
            if float(z['requested_wall_s'])!=row['requested_wall_s'] or float(z['exposure_s'])!=row['motion_stamp_s']:errors.append('clock/'+row['file'])
            if row['active']:
                active.append(row);valid_originals+=int(np.any(z['clean_rgb']) and np.any(z['clean_depth']>0))
                if not row['started_wall_s']<=row['capture_completed_wall_s']<row['ends_wall_s']:errors.append('fault_interval/'+row['file'])
                if row['mode'] in ('blackout','rgb_blackout') and np.any(z['rgb']):errors.append('nonblack_rgb/'+row['file'])
                if row['mode']=='blackout' and np.any(z['depth']):errors.append('nonblack_depth/'+row['file'])
                if row['mode']=='rgb_blackout' and not np.array_equal(z['depth'],z['clean_depth']):errors.append('changed_rgb_only_depth/'+row['file'])
                if row['mode']=='delay' and (row['delivery_ready_wall_s']-row['capture_completed_wall_s']<row['delay_wall_s']-1e-6
                    or not np.array_equal(z['rgb'],z['clean_rgb']) or not np.array_equal(z['depth'],z['clean_depth'])):errors.append('delay/'+row['file'])
            elif not np.array_equal(z['rgb'],z['clean_rgb']) or not np.array_equal(z['depth'],z['clean_depth']):errors.append('uncleared_fault/'+row['file'])
    brakes=audit_brakes(paths,steps);clock=read('wall_clock_audit.json');mission=read('mission_audit.json')
    onset=active[0]['started_wall_s'] if active else None;end=active[0]['ends_wall_s'] if active else None
    affected=[] if onset is None else [s for s in steps if onset<=s.get('monotonic_wall_s',-1)<=end]
    related=[] if onset is None else [b for b in brakes if b['start_wall_s'] is not None and onset<=b['start_wall_s']<=end+.25]
    passed=bool(active and valid_originals==len(active) and not errors and all(b['passed'] for b in brakes)
        and clock['passed'] and mission['all_continuous_geometry_passed'] and result['source_unchanged_during_run'])
    out=dict(scope='New sensor-side corruption; not physical occluder insertion or hardware; evidence/stop audit is separate from task success',
        sensor_fault=cfg['sensor_fault'],raw_fault_records=len(records),active_records=len(active),active_records_with_nonempty_original_render=valid_originals,
        fault_period_recorded_motion_ticks=len(affected),associated_braking_groups=len(related),brakes=brakes,
        original_and_estimator_inputs_verified=not errors,errors=errors,recorded_clock_passed=clock['passed'],
        geometry_passed=mission['all_continuous_geometry_passed'],task_completed=mission['scene_all_objects_delivered_audited'],
        controller_error=result.get('error'),evidence_and_stop_audit_passed=passed,
        interpretation='A fault that never overlapped motion cannot establish moving-robot braking response; no robustness success rate from one trial.')
    with (root/'rendered_stress_audit.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();audit(a.run)
