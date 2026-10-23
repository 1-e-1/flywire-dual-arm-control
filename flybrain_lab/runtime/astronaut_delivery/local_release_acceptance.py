"""Separate software delivery gates from still-open scientific claims."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .scene import ROOT


def evaluate():
    reports=ROOT/'reports'
    def load(path):
        p=reports/path
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    gates={};runs={}
    for folder in ('local_wrench_nominal_02','local_release_async_01','local_recovery_1003'):
        r=load(folder+'/result.json');audit=load(folder+'/independent_audit.json')
        gates[folder]=r.get('status')=='PASS' and len(r.get('completed',[]))==3 and audit.get('status')=='PASS'
        changes=r.get('map_change_statistics',[])
        gates[folder+'/map_event_replay']=len(changes)==6 and [x['epoch'] for x in changes]==list(range(1,7)) and all(x['policy']=='local' for x in changes)
        kept=[s['known_preserved']/max(1,s['known_before']) for e in changes for s in e['statistics']]
        gates[folder+'/preserved_evidence']=len(kept)==6 and min(kept)>0
        allocations=load(folder+'/visual_allocations.json') or []
        times=[a['optimizer_wall_s']*1000 for a in allocations if a.get('eligible')]
        runs[folder]=dict(status=r.get('status','MISSING'),simulation_time_s=r.get('simulation_time_s'),
            reaction=r.get('reaction_metrics'),map_points=r.get('map_points'),
            preserved_fraction_min=min(kept) if kept else None,preserved_fraction_max=max(kept) if kept else None,
            optimizer_p95_ms=float(np.percentile(times,95)) if times else None,
            independent_audit=audit)
    r=load('local_release_async_01/result.json');mesh=load('local_release_async_01/measured_surface.json')
    gates['async_streaming_budget']=r.get('cycle_deadline_misses')==0 and r.get('cycle_max_ms',float('inf'))<1000/60
    gates['measured_complex_surface']=mesh.get('triangles',0)>0 and mesh.get('valid_differentials',0)>0
    r=load('local_wrench_fault_01/result.json');steps=load('local_wrench_fault_01/control_steps.json') or []
    gates['fault_stop']=bool(steps) and r.get('status')=='STOPPED' and r.get('stopped_smoothly',False) and np.linalg.norm(steps[-1]['v'])<1e-8 and np.linalg.norm(steps[-1]['a'])<1e-8
    gates['fault_derivative_limits']=bool(steps) and all(np.max(np.abs([s[k] for s in steps]))<=lim+1e-8 for k,lim in [('v',.3),('a',.5),('jerk',5.)])
    p=reports/'local_update_final_tests.xml';suites=ET.parse(p).getroot().findall('testsuite') if p.exists() else []
    gates['tests']=bool(suites) and all(int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0 for s in suites)
    pilot=load('local_wrench_holdout_01/benchmark_summary.json');fresh=load('local_recovery_fresh_01/benchmark_summary.json')
    fresh_audits=[]
    for t in fresh.get('trials',[]):
        extended=load(f"local_recovery_fresh_01/combined_{t['seed']}/run/independent_audit_extended.json")
        passed=extended.get('status')=='PASS' if extended else t.get('audit_pass',False)
        fresh_audits.append(dict(seed=t['seed'],original_audit_pass=t.get('audit_pass'),
                                 extended_audit=extended or None,final_audit_pass=bool(passed)))
    gates['fresh_recovery_scenes']=len(fresh_audits)==2 and all(t['status']=='PASS' for t in fresh['trials']) and all(t['final_audit_pass'] for t in fresh_audits)
    gates={k:bool(v) for k,v in gates.items()}
    return dict(status='SCOPED_SIMULATION_DELIVERY_PASS' if all(gates.values()) else 'SCOPED_DELIVERY_GATES_INCOMPLETE',
        complete_research_project=False,gates=gates,runs=runs,tests=sum(int(s.attrib['tests']) for s in suites),
        original_eight_scene_pilot=pilot.get('groups'),post_change_two_fresh_scenes=fresh.get('groups'),
        additional_independent_axis_search=fresh_audits,
        failed_original_case_preserved='local_wrench_holdout_01/combined_1003',
        failure_regression_not_independent_holdout='local_recovery_1003',
        unproven=['Strict zero reaction and continuous minimum visual quality','Recursive feasibility under dynamic occlusion',
                  'Unknown-map-only millimetre collision safety','Overall superiority over old controller at equal task performance',
                  'Full 50-scene frozen comparison and external validation','Hardware calibration and tests, paper acceptance'],
        note='Software gates are not scientific claim gates. Original pilot failure is not erased by a successful regression.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'reports/local_release_acceptance.json'));args=p.parse_args()
    result=evaluate();Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    if result['status']!='SCOPED_SIMULATION_DELIVERY_PASS':raise SystemExit(1)


if __name__=='__main__':main()
