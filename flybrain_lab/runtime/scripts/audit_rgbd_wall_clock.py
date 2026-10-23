"""Independent recorded-clock audit; not a hardware or schedulability proof."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def bounded(value,upper=.25):
    return isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=upper


def evaluate_records(configuration,paths,steps,observations):
    failures=[];total=0
    def check(ok,message):
        nonlocal total
        if not ok:
            total+=1
            if len(failures)<20:failures.append(message)
    check(configuration.get('guidance_clock')=='motion_and_monotonic_wall_with_pre_execution_refresh','unsupported_historical_clock')
    check(bool(paths) and bool(steps) and bool(observations),'no_executed_evidence')
    by_clock={o.get('requested_wall_s'):o for o in observations}
    for i,p in enumerate(paths):
        refresh=p.get('execution_refresh',{});request=refresh.get('requested_wall_s');ex=refresh.get('execution_wall_s')
        o=by_clock.get(request)
        check(o is not None,f'path_{i}_missing_new_raw_record')
        check(bounded(refresh.get('raw_age_s')),f'path_{i}_stale_raw')
        clock_ok=isinstance(request,(int,float)) and isinstance(ex,(int,float)) and bounded(ex-request)
        check(clock_ok,f'path_{i}_request_to_execution_clock')
        rollback=p.get('phase')=='certified_view_rollback_right_rest'
        if not rollback:
            check(bounded(refresh.get('target_age_s')),f'path_{i}_stale_target')
            target={} if o is None else o['targets'].get(refresh.get('target'),{})
            stamp=target.get('wall_stamp')
            check(isinstance(stamp,(int,float)) and isinstance(ex,(int,float)) and bounded(ex-stamp),f'path_{i}_target_clock_not_supported_by_observation')
        if p.get('visual_certificate',{}).get('allowed') is True:
            check((refresh.get('visual') or {}).get('allowed') is True,f'path_{i}_missing_new_visual_certificate')
    normal=[s for s in steps if s.get('phase')!='braking']
    for i,s in enumerate(normal):
        key='raw_age_s' if s.get('phase')=='certified_view_rollback_right_rest' else 'guidance_age_s'
        check(bounded(s.get(key)),f'normal_tick_{i}_expired_or_missing_age')
        check(s.get('feedback_stop_reason') is None,f'normal_tick_{i}_ignored_stop')
    ages=[s['guidance_age_s'] for s in normal if bounded(s.get('guidance_age_s'))]
    return dict(passed=total==0,violations=total,first_violations=failures,paths_checked=len(paths),ticks_checked=len(steps),
        normal_ticks_checked=len(normal),braking_ticks_exempt_from_normal_freshness=len(steps)-len(normal),
        maximum_logged_normal_guidance_age_s=max(ages) if ages else None,
        scope='Recorded pre-execution and normal-tick exposure clocks; braking may outlast image freshness; not an external timestamp or full-stack deadline certificate')


def audit(run):
    folder=Path(run);read=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    obs=read('observations.json');out=evaluate_records(read('run_configuration.json'),read('paths.json'),read('steps.json'),obs)
    mismatches=[]
    for o in obs:
        path=(folder/'raw'/o['file']).resolve()
        if not path.is_relative_to((folder/'raw').resolve()):raise ValueError('raw path escapes run')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=o.get('sha256'):mismatches.append(o['file']);continue
        with np.load(path,allow_pickle=False) as z:
            if ('requested_wall_s' not in z or float(z['requested_wall_s'])!=o.get('requested_wall_s')
                or float(z['exposure_s'])!=o['time_s']):mismatches.append(o['file'])
    out.update(raw_records_checked=len(obs),raw_clock_or_hash_mismatches=mismatches,
        source_unchanged_during_execution=read('result.json').get('source_unchanged_during_run',False))
    out['passed'] &= not mismatches and out['source_unchanged_during_execution']
    output=folder/'wall_clock_audit.json'
    with output.open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    audit(args.run)
