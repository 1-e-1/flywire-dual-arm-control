"""Matched initial state, RIGHT curve and clock; no reaction retiming confound.

Independent segment counterfactuals, NOT complete missions. Preserve ineligible,
unsafe and measured-visibility failure cases. Truth goes only to scene rendering.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,Gf
from threadpoolctl import threadpool_limits
from .motion import CollisionGuard,Hull,world
from .simultaneous import TightBoundArm,SimultaneousGuard
from .reaction import CombinedReaction,reaction_metrics
from .smooth_path import SmoothPath
from .visual_sufficiency import VisualSetAllocator,SufficiencyConfig
from .run import DualArmKinematics
from .vision import VisualGuide
from .d435i import D435iCamera,load_profile


def experiment(source,folder,objective='momentum'):
    source=Path(source);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    load=lambda n:json.loads((source/n).read_text())
    result=load('result.json');paths=load('certified_paths.json');observations=load('telemetry/observations.json');records=load('trajectory.json')
    stage=Usd.Stage.Open(result['scene_path']);stage.SetEditTarget(stage.GetSessionLayer());robot=DualArmKinematics(stage)
    left,right=TightBoundArm(stage,'left'),TightBoundArm(stage,'right');model=CombinedReaction(stage,left,right)
    guard=CollisionGuard(stage,left,right);paired=SimultaneousGuard(guard);allocator=VisualSetAllocator(left,model,SufficiencyConfig(objective=objective))
    camera=D435iCamera();guide=VisualGuide({p['name']:p['color'] for p in result['config']['parts']},sensor_profile=load_profile());guide.fit_circular_cap=True
    initial={p.GetName():np.array(world(p)).T for p in stage.GetPrimAtPath('/World/Parts').GetChildren()}
    local={n:(h.points-initial[n][:3,3])@initial[n][:3,:3] for n,h in guard.parts.items()};rows=[]
    for index,path in enumerate(paths):
        phase=path['phase'];stamp=path['start_time_s'];name=phase.split('/')[0]
        if path['held'] or not ('/approach' in phase or '/servo_' in phase):continue
        row=dict(segment=index,phase=phase,start_time_s=stamp,eligible=False)
        past=[o for o in observations if o['exposure_s']<=stamp+1e-8 and name in o['estimates']]
        if not past:row['reason']='no_measurement';rows.append(row);continue
        e=past[-1]['estimates'][name];q=np.array(path['start']).reshape(2,8);rg=np.array(path['end'])[8:]
        try:context=allocator.context(q[0],np.array(e['position']),np.array(e['covariance']),e['pixels'],stamp-e['stamp'])
        except ValueError as exc:row['reason']=str(exc);rows.append(row);continue
        state=[r for r in records if r['time_s']<=stamp+1e-8][-1]
        for part,pose in state['parts'].items():
            t=np.array(pose).T;guard.parts[part]=Hull.make('part/'+part,local[part]@t[:3,:3].T+t[:3,3])
            UsdGeom.Xformable(stage.GetPrimAtPath('/World/Parts/'+part)).GetOrderedXformOps()[0].Set(Gf.Matrix4d(pose))
        goals={'frozen':q[0].copy()};diagnostics={}
        for mode in ('pose','set'):goals[mode],diagnostics[mode]=allocator.propose(q,rg,context,mode)
        curves={mode:SmoothPath.make(q.ravel(),np.r_[goal,rg]) for mode,goal in goals.items()}
        duration=max(path['duration_s'],max(p.duration for p in curves.values()))
        row.update(eligible=True,common_duration_s=duration,methods={})
        for mode,curve in curves.items():
            curve=curve.retime(duration/curve.duration);started=time.perf_counter()
            safe=paired.segment(curve.start,curve.end,contact_part=path['contact_part'],support_contact=path['support_contact'])
            certificate_wall=time.perf_counter()-started
            dt=duration/120;W=np.array([model.wrench(*curve.state((j+.5)*dt)[:3]) for j in range(120)])
            probes=[]
            for j,u in enumerate((0.,.5,1.)):
                robot.set_positions(*curve.state(duration*u)[0].reshape(2,8));guide.estimates.clear()
                obs=camera.capture_roi(stage,left.camera(curve.state(duration*u)[0][:8]),stamp+duration*u,context['target'],seed=1000+index*10+j)
                guide.update(obs,integrate_map=False);estimate=guide.target(name,obs.stamp)
                quality=False
                if estimate is not None:
                    try:allocator.context(curve.state(duration*u)[0][:8],estimate.position,estimate.covariance,estimate.pixels,0.);quality=True
                    except ValueError:pass
                probes.append(dict(fraction=u,detected=estimate is not None,measured_quality=quality,pixels=estimate.pixels if estimate else 0))
            row['methods'][mode]=dict(collision_certified=safe,certificate_wall_s=certificate_wall,reaction=reaction_metrics(W,dt),
                                      measured_visibility_probes=probes,all_measured_probes_sufficient=all(p['measured_quality'] for p in probes),
                                      optimization=diagnostics.get(mode),start=curve.start.tolist(),end=curve.end.tolist())
        rows.append(row);print('matched_segment',index,flush=True)
        (folder/'segments.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    eligible=[r for r in rows if r['eligible']];paired_rows=[r for r in eligible if all(m['collision_certified'] for m in r['methods'].values())]
    summaries={}
    for mode in ('frozen','pose','set'):
        ms=[r['methods'][mode] for r in paired_rows]
        summaries[mode]=dict(count=len(ms),all_probe_quality_cases=sum(m['all_measured_probes_sufficient'] for m in ms),
                            mean_force_peak_N=float(np.mean([m['reaction']['force']['peak'] for m in ms])) if ms else None,
                            mean_torque_peak_Nm=float(np.mean([m['reaction']['torque']['peak'] for m in ms])) if ms else None,
                            mean_force_window_impulse_Ns=float(np.mean([m['reaction']['force']['maximum_window_impulse'] for m in ms])) if ms else None,
                            optimizer_p95_ms=float(np.percentile([m['optimization']['optimizer_wall_s']*1000 for m in ms],95)) if ms and mode!='frozen' else None)
    summary=dict(source=str(source),objective=objective,candidate_segments=len(rows),eligible_segments=len(eligible),all_methods_collision_feasible=len(paired_rows),methods=summaries,
                 per_method_collision_feasible={mode:sum(r['methods'][mode]['collision_certified'] for r in eligible) for mode in summaries},
                 scope='Independent matched segments, identical right joint curve and duration; no reaction retiming. Three measured visibility probes are NOT continuous occlusion guarantees.',
                 selection='All pick segments retained; eligibility and collision filtering counts explicitly reported; not complete mission success.',
                 method_status='Experimental geometric visual sufficiency, not calibrated task accuracy or recursive feasibility')
    (folder/'segments.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='reports/cooperative_dashboard');p.add_argument('--output',required=True)
    p.add_argument('--objective',choices=['momentum','wrench'],default='momentum');args=p.parse_args()
    with threadpool_limits(1):experiment(args.source,args.output,args.objective)

if __name__=='__main__':main()
