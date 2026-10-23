"""Failure-accounted scene-level report; never pool changed source versions."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest


def wilson(successes,n):
    if n==0:return None
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z/den*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [max(0.,center-half),min(1.,center+half)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if path and Path(path).is_file() else {}


def analyze(evaluation):
    data=read(evaluation);protocol=data['protocol'];rows=data['rows']
    if not protocol.get('whole_mission'):raise ValueError('this report requires the whole-mission audit protocol')
    if not data.get('source_unchanged') or data['finished_trials']!=data['planned_trials']:
        raise ValueError('incomplete or mixed-source batch; not eligible for final paired report')
    seeds=protocol['seeds'];modes=protocol['modes'];indexed={(r['seed'],r['mode']):r for r in rows}
    if len(indexed)!=len(rows) or set(indexed)!={(s,m) for s in seeds for m in modes}:raise ValueError('missing or duplicate trial')
    records={}
    for key,row in indexed.items():
        result=read(row.get('result'));audit=read(row.get('audit'))
        success=all((audit.get('scene_all_objects_delivered_audited',False),
            audit.get('all_continuous_geometry_passed',False),audit.get('wrench_crosscheck_passed',False),
            audit.get('terminal_C2_rest',False),result.get('source_unchanged_during_run',False)))
        clock_required=(protocol.get('split')=='prespecified_holdout'
            or result.get('guidance_clock')=='motion_and_monotonic_wall_with_pre_execution_refresh')
        clock=read(Path(row['result']).parent/'wall_clock_audit.json') if row.get('result') else {}
        clock_passed=clock.get('passed') is True
        missing_clock_failure=success and clock_required and not clock_passed
        success=success and (not clock_required or clock_passed)
        metrics={}
        if result.get('reaction'):
            for kind in ('force','torque'):
                for name in ('rms','peak','accumulated_magnitude','maximum_window_impulse'):
                    metrics[kind+'_'+name]=result['reaction'][kind][name]
            metrics['normalized_mean_square']=(metrics['force_rms']/.35)**2+(metrics['torque_rms']/.15)**2
            metrics['motion_time_s']=result['simulation_time_s']
            metrics['normalized_integral_s']=metrics['normalized_mean_square']*metrics['motion_time_s']
        if result.get('timing'):metrics['task_wall_including_setup_s']=result['timing']['task_wall_s_including_setup']
        if result.get('cycle_p99_ms') is not None:metrics['streaming_p99_ms']=result['cycle_p99_ms']
        records[key]=dict(success=success,metrics=metrics,result=result,status=row['status'],
            recorded_clock_required=clock_required,recorded_clock_passed=clock_passed,
            error=result.get('error') or ('missing_or_failed_recorded_clock_audit' if missing_clock_failure else None))
    # Compare inputs using the retained per-trial configurations, not merely a
    # shared-looking folder name. A process failure remains in the denominator.
    pairing=[]
    for seed in seeds:
        configs=[];initial=[];hashes=[]
        for mode in modes:
            row=indexed[seed,mode];result=records[seed,mode]['result']
            if row.get('result') and result:
                cfg=read(Path(row['result']).parent/'run_configuration.json');cfg.pop('mode',None)
                configs.append(cfg);initial.append(result['initial_joints']);hashes.append(result['source_sha256'])
        equal=all(c==configs[0] for c in configs) and all(q==initial[0] for q in initial) and all(h==hashes[0] for h in hashes)
        if not equal:raise ValueError('mismatched policy inputs for seed '+str(seed))
        pairing.append(dict(seed=seed,completed_result_inputs_matched=equal,results_available=len(configs)))
    summaries={}
    for mode in modes:
        selected=[records[s,mode] for s in seeds];passed=sum(r['success'] for r in selected)
        failures={}
        for r in selected:
            if not r['success']:
                reason=r['error'] or r['status'];failures[reason]=failures.get(reason,0)+1
        summaries[mode]=dict(attempts=len(seeds),successes=passed,success_rate=passed/len(seeds),wilson_95=wilson(passed,len(seeds)),failures=failures)
    comparisons=[];rng=np.random.default_rng(20260910)
    if 'set' in modes:
        for baseline in [m for m in modes if m!='set']:
            a_only=sum(records[s,'set']['success'] and not records[s,baseline]['success'] for s in seeds)
            b_only=sum(records[s,baseline]['success'] and not records[s,'set']['success'] for s in seeds)
            common=[s for s in seeds if records[s,'set']['success'] and records[s,baseline]['success']]
            names=set.intersection(*(set(records[s,m]['metrics']) for s in common for m in ('set',baseline))) if common else set()
            metrics={}
            for name in sorted(names):
                a=np.array([records[s,'set']['metrics'][name] for s in common]);b=np.array([records[s,baseline]['metrics'][name] for s in common]);delta=a-b
                ci=None
                if len(common)>=2:
                    draws=delta[rng.integers(len(common),size=(10000,len(common)))].mean(1);ci=np.quantile(draws,[.025,.975]).tolist()
                metrics[name]=dict(set_mean=float(a.mean()),baseline_mean=float(b.mean()),paired_mean_difference=float(delta.mean()),
                    paired_bootstrap_95=ci,ratio_of_means_percent=float((a.mean()/b.mean()-1)*100) if b.mean()!=0 else None)
            comparisons.append(dict(baseline=baseline,set_only_success=a_only,baseline_only_success=b_only,
                exact_discordance_p=float(binomtest(a_only,a_only+b_only,.5).pvalue) if a_only+b_only else 1.,
                common_success_seeds=common,common_success_count=len(common),conditional_metrics=metrics))
    scope=('Prespecified unseen-scene comparison within the declared sampler; no hardware or universal superiority claim'
        if protocol.get('split')=='prespecified_holdout' else 'Scene-level development comparison, not holdout generalization or hardware validation')
    return dict(scope=scope,protocol=protocol,
        attempts=data['planned_trials'],pairing=pairing,policies=summaries,comparisons=comparisons,
        attempted_trial_metrics=[dict(seed=s,mode=m,**{k:v for k,v in records[s,m].items() if k!='result'}) for s in seeds for m in modes],
        caveats=['Failed trials are not assigned zero complete-task cost','Conditional metrics use common audited successes only',
            'Small pilots have imprecise intervals; frames are not independent scenes','Named timing intervals may overlap',
            'No multiplicity correction or population superiority claim from exploratory component tests'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    result=analyze(args.evaluation);path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# RGB-D paired evaluation','',result['scope'],
        '', '| Policy | Audited scene successes / attempts | 95% Wilson interval |','|---|---:|---|']
    for mode,row in result['policies'].items():lines.append(f'| {mode} | {row["successes"]} / {row["attempts"]} | {row["wilson_95"]} |')
    for comparison in result['comparisons']:
        lines.extend(['',f'## Set versus {comparison["baseline"]}','',f'Common audited successes: {comparison["common_success_count"]}.',
            '', '| Metric | Set mean | Baseline mean | Relative difference |','|---|---:|---:|---:|'])
        for name,m in comparison['conditional_metrics'].items():lines.append(f'| {name} | {m["set_mean"]:.6g} | {m["baseline_mean"]:.6g} | {m["ratio_of_means_percent"]} |')
    lines.extend(['','Caveats:',*['- '+c for c in result['caveats']]])
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result['policies'],indent=2))


if __name__=='__main__':main()
