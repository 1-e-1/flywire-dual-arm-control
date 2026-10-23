"""Collect executable benchmark gates without claiming hardware certification."""
import json,argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .scene import ROOT

def evaluate():
    base=ROOT/'reports';load=lambda p:json.loads(p.read_text());gates={};runs={}
    for name in ['cooperative_baseline_final','cooperative_reaction_final','cooperative_async_final','cooperative_seed41']:
        report=load(base/name/'result.json');audit=load(base/name/'independent_audit.json')
        gates[name]=report['status']=='PASS' and report['completed']==['bolt','washer','bracket'] and audit['status']=='PASS'
        runs[name]=dict(result=str(base/name/'result.json'),audit=str(base/name/'independent_audit.json'),simulation_time_s=report['simulation_time_s'],reaction_metrics=report['reaction_metrics'])
    async_report=load(base/'cooperative_async_final/result.json')
    gates['streaming_computation_budget']=async_report['cycle_deadline_misses']==0 and async_report['cycle_max_ms']<1000/60
    fault=load(base/'cooperative_dual_fault/result.json');steps=load(base/'cooperative_dual_fault/control_steps.json')
    gates['dual_arm_smooth_fault_stop']=fault['status']=='STOPPED' and fault['stopped_smoothly'] and fault['simultaneous_moving_ticks']>0 and np.linalg.norm(steps[-1]['v'])<1e-8 and np.linalg.norm(steps[-1]['a'])<1e-8
    gates['fault_derivative_limits']=all(np.max(np.abs([s[k] for s in steps]))<=lim+1e-8 for k,lim in [('v',.3),('a',.5),('jerk',5.)])
    suites=ET.parse(base/'cooperative_tests.xml').getroot().findall('testsuite')
    gates['regression_tests']=bool(suites) and all(int(s.attrib['failures'])==0 and int(s.attrib['errors'])==0 for s in suites)
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(str(ROOT/'isaacsim_model/astronaut_delivery_cooperative_demo.usd'))
    gates['replay_units_and_axis']=UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
    gates={key:bool(value) for key,value in gates.items()}
    result=dict(status='SIMULATION_BENCHMARK_PASS' if all(gates.values()) else 'FAIL',gates=gates,runs=runs,
                tests=sum(int(s.attrib['tests']) for s in suites),
                incomplete_research_and_hardware_requirements=['Strict zero reaction not demonstrated','Hardware COM/inertia/hand-eye/latency uncalibrated',
                'Hard real-time including planning/transport not certified','Unknown-scene collision avoidance from point cloud alone not validated',
                'Native RTX/dynamic astronaut/contact/flexible base not validated'],
                scope='Complete known-workcell kinematic delivery benchmark, not complete physical deployment')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'reports/cooperative_acceptance.json'));args=p.parse_args()
    result=evaluate();Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
    if result['status']!='SIMULATION_BENCHMARK_PASS':raise SystemExit(1)

if __name__=='__main__':main()
