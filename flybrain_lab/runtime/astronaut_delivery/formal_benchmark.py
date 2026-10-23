"""Frozen paired complete-mission experiments; all draws/failures retained."""
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from .scene import ROOT


def fingerprint(paths):return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_scene(folder,seed,methods,timeout,hashes):
    from .randomized_benchmark import draw,make_scene,screen
    folder.mkdir(exist_ok=False);cfg=draw(seed,'combined');scene=make_scene(folder,cfg)
    admitted,checks=screen(scene,cfg)
    result=dict(seed=seed,admitted=admitted,checks=checks,methods={},order=list(methods if seed%2 else methods[::-1]))
    (folder/'admission.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not admitted:return result
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    for method in result['order']:
        current=fingerprint([ROOT/p for p in hashes])
        if current!=hashes:
            result['methods'][method]=dict(status='SOURCE_CHANGED',changed=[p for p in hashes if current[p]!=hashes[p]])
            continue
        out=folder/method;began=time.perf_counter()
        cmd=[sys.executable,'-B','-m','astronaut_delivery.sufficiency_delivery','--output',str(out),
             '--scene',str(scene),'--task-config',str(folder/'task.json'),'--seed',str(seed),'--allocation',method,
             '--objective','wrench','--anytime-feasible','--map-policy','local']
        row=dict(status='PROCESS_FAILED',success=False)
        try:
            with (folder/(method+'.log')).open('w',encoding='utf-8') as log:
                process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            report=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else {}
            row.update(status=report.get('status','PROCESS_FAILED'),error=report.get('error'),
                       exit_code=process.returncode,completed=len(report.get('completed',[])),
                       simulation_time_s=report.get('simulation_time_s'),reaction=report.get('reaction_metrics'))
            if report.get('status')=='PASS' and process.returncode==0:
                with (folder/(method+'_audit.log')).open('w',encoding='utf-8') as log:
                    check=subprocess.run([sys.executable,'-B','-m','astronaut_delivery.cooperative_audit','--run',str(out)],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                row['audit_pass']=check.returncode==0;row['success']=check.returncode==0
            if (out/'events.json').exists():
                events=json.loads((out/'events.json').read_text())
                row['capture_gaps_m']=[e['capture_gap_m'] for e in events if e['event']=='attached']
                row['recoveries']=sum(e['event']=='bounded_visual_recovery' for e in events)
        except subprocess.TimeoutExpired:row.update(status='TIMEOUT',success=False)
        row['wall_s']=time.perf_counter()-began
        row['source_unchanged_after_execution']=fingerprint([ROOT/p for p in hashes])==hashes
        if not row['source_unchanged_after_execution']:row.update(status='SOURCE_CHANGED',success=False)
        result['methods'][method]=row
        (folder/'paired_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,method=method,**row)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--count',type=int,default=50)
    p.add_argument('--seed-start',type=int,default=2000);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--timeout',type=float,default=600);p.add_argument('--methods',nargs='+',choices=['pose','set'],default=['pose','set'])
    args=p.parse_args();base=Path(args.output).resolve();base.mkdir(parents=True,exist_ok=False)
    sources=[p for d in ('astronaut_delivery','classical_fusion','codex_zbot','scripts') for p in (ROOT/d).rglob('*.py')]
    sources+=list((ROOT/'configs').glob('*.json'));hashes=fingerprint(sources)
    protocol=dict(seeds=list(range(args.seed_start,args.seed_start+args.count)),methods=args.methods,
                  workers=args.workers,timeout_s=args.timeout,source_sha256=hashes,python=sys.version,platform=platform.platform(),
                  primary_endpoint='Audited complete mission success per scene; retain all failures',
                  secondary_endpoints='Paired scene-level force/torque/impulse/time conditional on common success, with cohort counts',
                  fixed_before_execution=True,external_preregistration=False,
                  scope='Three known 20-mm docking interfaces; randomized body geometry/position/yaw/table height. Kinematic simulation.',
                  timing='Concurrent wall time is NOT a real-time benchmark. No success-conditioned redrawing.')
    (base/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_scene,base/f'scene_{seed}',seed,args.methods,args.timeout,hashes):seed for seed in protocol['seeds']}
        for job in as_completed(jobs):
            try:row=job.result()
            except Exception as exc:row=dict(seed=jobs[job],admitted=None,status='RUNNER_ERROR',error=repr(exc),methods={})
            rows.append(row);rows.sort(key=lambda x:x['seed'])
            (base/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps(dict(completed_scenes=len(rows),requested_scenes=args.count)),flush=True)


if __name__=='__main__':main()
