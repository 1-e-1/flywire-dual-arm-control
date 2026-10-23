"""Bounded simultaneous visual-approach / reaction-control experiment.

Two modes: deterministic simulated sensor delay, or asynchronous wall-clock
sensor/control workers. Neither mode is a physical arm driver or a full delivery
mission. Only the initial q is loaded from a legacy run, never its trajectory.
"""
import argparse,copy,json,time
import queue,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd,UsdGeom
from threadpoolctl import threadpool_limits
from .scene import ROOT
from .online_scene import ONLINE_SCENE
from .motion import Arm,CollisionGuard,world
from .reaction import CombinedReaction,reaction_metrics
from .simultaneous import SimultaneousGuard,TightBoundArm
from .online_control import JointOnlineController,LatestObservation,VisualSnapshot
from .d435i import D435iCamera,load_profile
from .vision import VisualGuide
from .run import DualArmKinematics

class SensorWorker:
    """Own USD stage and camera: never read a stage while control mutates it."""
    def __init__(self,cfg):self.cfg=cfg;self.initialized=False;self.rows=[];self.frames=[]
    def capture(self,q,exposure):
        if not self.initialized:
            self.stage=Usd.Stage.Open(str(ONLINE_SCENE));self.stage.SetEditTarget(self.stage.GetSessionLayer())
            self.robot=DualArmKinematics(self.stage);self.left=Arm(self.stage,'left');self.camera=D435iCamera()
            delivery=json.loads((ROOT/'configs/astronaut_delivery.json').read_text())
            self.guide=VisualGuide({p['name']:p['color'] for p in delivery['parts']},sensor_profile=load_profile())
            self.initialized=True
        start=time.perf_counter();q=np.asarray(q).reshape(2,8);self.robot.set_positions(*q)
        prediction=self.guide.estimates.get(self.cfg['target'])
        roi=prediction is not None
        if roi:obs=self.camera.capture_roi(self.stage,self.left.camera(q[0]),exposure,prediction.position,seed=self.cfg['seed']+len(self.rows))
        else:obs=self.camera.capture(self.stage,self.left.camera(q[0]),exposure,seed=self.cfg['seed']+len(self.rows))
        # Precision phase: use the full bootstrap map, prioritize local tracking.
        # Background full-map refresh during delivery is not implemented here.
        self.guide.update(obs,integrate_map=not roi);target=self.guide.target(self.cfg['target'],exposure)
        self.rows.append(dict(exposure_s=exposure,wall_s=time.perf_counter()-start,valid_ratio=self.camera.last_quality['valid_depth_ratio'],detected=target is not None,
                             roi=roi,shape=list(obs.depth.shape),intrinsics=obs.K.tolist()))
        self.frames.append((obs.rgb*255).astype(np.uint8))
        return None if target is None else VisualSnapshot(target.position.copy(),target.covariance.copy(),target.stamp,exposure)

def experiment(cfg,folder,clock='simulated'):
    folder=Path(folder)
    if folder.exists() and any(folder.iterdir()):raise ValueError('Use an empty output directory')
    folder.mkdir(parents=True,exist_ok=True)
    stage=Usd.Stage.Open(str(ONLINE_SCENE));stage.SetEditTarget(stage.GetSessionLayer())
    left,right=TightBoundArm(stage,'left'),TightBoundArm(stage,'right');reaction=CombinedReaction(stage,left,right)
    guard=SimultaneousGuard(CollisionGuard(stage,left,right));robot=DualArmKinematics(stage)
    records=json.loads((ROOT/cfg['initial_condition_report']).read_text())
    initial=[r for r in records if r['phase']==cfg['initial_condition_phase']][-1]
    q=np.array(initial['q']).reshape(16);v=np.zeros(16);dt=cfg['dt_s']
    if not guard.box(q,q)[0]:raise RuntimeError('Invalid initial work configuration')
    robot.set_positions(*q.reshape(2,8));control=JointOnlineController(left,right,reaction,guard,cfg)
    worker=SensorWorker(cfg);broker=LatestObservation(cfg['maximum_observation_age_s']);pool=ThreadPoolExecutor(max_workers=1)
    events=[];trajectory=[];steps=[];wrenches=[];durations=[];pending=None;error=None;braking=None;emergency=False;goal_brake=False
    sensor_stop=threading.Event();state_lock=threading.Lock();shared_q=q.copy();inbox=queue.SimpleQueue()
    def event(kind,**kw):
        row=dict(event=kind,**kw);events.append(row);print(json.dumps(row),flush=True)
    def record(t,phase):
        parts={p.GetName():np.array(world(p)).tolist() for p in stage.GetPrimAtPath('/World/Parts').GetChildren()}
        trajectory.append(dict(time_s=float(t),q=q.reshape(2,8).tolist(),parts=parts,phase=phase,held=None))
    try:
        # Initialization is at rest. Bootstrap is not billed as online throughput.
        first=pool.submit(worker.capture,q.copy(),0.).result()
        if first is None:raise RuntimeError('No bootstrap visual target')
        # Warm ROI geometry and linear algebra outside the online deadline test.
        for _ in range(2):
            warm=pool.submit(worker.capture,q.copy(),0.).result()
            if warm is not None:first=warm
        control.propose(q,v,first.position);control.previous_acceleration.fill(0)
        broker.publish(first);target=broker.read(0.);initial_target=target.position.copy()
        initial_tcp=right.tcp(q[8:])[:3,3].copy();initial_range=np.linalg.norm(left.camera(q[:8])[:3,3]-target.position)
        wall_start=time.perf_counter();next_capture=0.;record(0.,'reset_work_pose')
        if clock=='wall':
            def produce():
                while not sensor_stop.is_set():
                    started=time.perf_counter()
                    with state_lock:pose=shared_q.copy()
                    data=worker.capture(pose,started-wall_start)
                    if data is not None:inbox.put(VisualSnapshot(data.position,data.covariance,data.exposure_s,time.perf_counter()-wall_start))
                    sensor_stop.wait(max(0.,1/cfg['perception_rate_hz']-(time.perf_counter()-started)))
            pending=pool.submit(produce)
        event('start',clock=clock,reaction_weight=cfg['reaction_weight'],initial_range_m=float(initial_range))
        t=0.;tick=0;last_progress=wall_start
        while t<cfg['duration_s'] or braking is not None:
            tick_start=time.perf_counter();now=t if clock=='simulated' else tick_start-wall_start
            if clock=='wall':
                if pending.done():pending.result();raise RuntimeError('Sensor producer exited unexpectedly')
                while not inbox.empty():broker.publish(inbox.get())
            elif t>=next_capture and t<cfg['duration_s']:
                data=pool.submit(worker.capture,q.copy(),t).result()
                if data is not None:broker.publish(VisualSnapshot(data.position,data.covariance,data.exposure_s,t+cfg['sensor_latency_s']))
                next_capture+=1/cfg['perception_rate_hz']
            control_start=time.perf_counter();snap=broker.read(now)
            phase='online_approach';step_dt=dt
            if braking is None and snap is not None and t>=cfg['minimum_online_duration_s']:
                # This is a pre-grasp approach experiment, not indefinite camera
                # repositioning after arrival. Finish only via a certified stop.
                stop_duration=float(np.max(abs(v))/cfg['emergency_deceleration_rad_s2'])
                if np.isfinite(stop_duration):
                    stopped=q+.5*stop_duration*v
                    xr_stop,_,xc_stop,_=control.tasks(stopped,snap.position)
                    desired_stop=snap.position+[0,0,cfg['right_target_height_m']]
                    if np.linalg.norm(xr_stop[:3]-desired_stop)<=cfg['approach_completion_tolerance_m'] and np.max(abs(xc_stop[:2]))<.25 and xc_stop[3]>=cfg['camera_minimum_elevation_sine']:
                        stop=guard.brake(q,v,cfg['emergency_deceleration_rad_s2'])
                        if stop is not None:
                            braking=stop;goal_brake=True
                            event('goal_stop_certified',time_s=t,predicted_endpoint_error_m=float(np.linalg.norm(xr_stop[:3]-desired_stop)))
            if braking is None and t>=cfg['duration_s']-dt*.1:
                braking=guard.brake(q,v,cfg['emergency_deceleration_rad_s2']);phase='finish_brake'
                if braking is None:raise RuntimeError('No end braking path')
            if braking is None:
                a=control.propose(q,v,snap.position) if snap is not None else None
                if a is None:
                    reason=control.last_status if snap is not None else 'stale_visual_observation'
                    braking=guard.brake(q,v,cfg['emergency_deceleration_rad_s2']);emergency=True
                    event('safety_brake',time_s=t,reason=reason)
                    if braking is None:raise RuntimeError('No certified emergency brake; no further motion applied')
            if braking is not None:
                remaining,a=braking;phase='emergency_brake' if emergency else 'finish_brake'
                if remaining<1e-10:break
                step_dt=min(dt,remaining);braking=(max(0.,remaining-step_dt),a)
                if not guard.acceleration(q,v,a,step_dt):raise RuntimeError('Certified brake became invalid')
            midpoint=q+.5*step_dt*v+.125*step_dt*step_dt*a
            midpoint_v=v+.5*step_dt*a
            wrench=reaction.wrench(midpoint,midpoint_v,a);wrenches.append(wrench);durations.append(step_dt)
            q=q+step_dt*v+.5*step_dt*step_dt*a;v=v+step_dt*a
            robot.set_positions(*q.reshape(2,8));t+=step_dt;tick+=1
            with state_lock:shared_q=q.copy()
            elapsed=time.perf_counter()-control_start
            steps.append(dict(time_s=t,v=v.tolist(),a=np.asarray(a).tolist(),wrench=wrench.tolist(),phase=phase,
                              observation_age_s=None if snap is None else now-snap.exposure_s,control_and_check_wall_s=elapsed,
                              iteration_wall_s=time.perf_counter()-tick_start))
            record(t,phase)
            if time.perf_counter()-last_progress>15:
                event('progress',time_s=t,frames=len(worker.rows),phase=phase);last_progress=time.perf_counter()
            if clock=='wall':
                remaining_wall=dt-(time.perf_counter()-tick_start)
                if remaining_wall>0:time.sleep(remaining_wall)
        wall_elapsed=time.perf_counter()-wall_start
        final_tcp=right.tcp(q[8:])[:3,3];desired=initial_target+[0,0,cfg['right_target_height_m']]
        initial_error=np.linalg.norm(initial_tcp-desired);final_error=np.linalg.norm(final_tcp-desired)
        final_range=np.linalg.norm(left.camera(q[:8])[:3,3]-initial_target)
        simultaneous=sum(np.linalg.norm(np.array(s['v'])[:8])>1e-4 and np.linalg.norm(np.array(s['v'])[8:])>1e-4 for s in steps)
        metrics=dict(initial_right_error_m=float(initial_error),final_right_error_m=float(final_error),
                     right_error_reference='bootstrap RGB-D estimate, not ground truth or hardware accuracy',
                     right_error_reduction_fraction=float(1-final_error/max(initial_error,1e-12)),
                     initial_camera_range_m=float(initial_range),final_camera_range_m=float(final_range),
                     simultaneous_moving_ticks=int(simultaneous),total_ticks=len(steps),goal_stop_certified=goal_brake,
                     executed_simulation_time_s=t)
        if emergency:error='Safety stop; experiment did not complete its scheduled approach'
        elif final_error>.015:error='Visual approach residual exceeds 15 mm experiment threshold'
        elif simultaneous==0:error='No simultaneous motion demonstrated'
    except Exception as exc:
        error=str(exc);metrics={};wall_elapsed=None;event('error',error=error)
    finally:
        # Only our own worker is joined; no process termination or external writes.
        sensor_stop.set()
        pool.shutdown(wait=True)
    timing=np.array([s['control_and_check_wall_s'] for s in steps]);solver=np.array(control.timing)
    result=dict(status='PASS' if error is None else 'STOPPED',error=error,scene_path=str(ONLINE_SCENE),controller='joint_online_acceleration_qp',
                sensor_profile=load_profile(),clock_mode=clock,config=cfg,inertia=reaction.manifest(),metrics=metrics,
                reaction_metrics=reaction_metrics(wrenches,durations) if wrenches else None,
                wall_elapsed_s=wall_elapsed,solver_p95_ms=float(np.percentile(solver,95)*1000) if len(solver) else None,
                control_check_p95_ms=float(np.percentile(timing,95)*1000) if len(timing) else None,
                control_check_p99_ms=float(np.percentile(timing,99)*1000) if len(timing) else None,
                deadline_misses=int(np.sum(timing>dt)),certified_intervals=guard.certificates,
                iteration_deadline_misses=sum(s['iteration_wall_s']>dt for s in steps),
                validation_scope='coarse_pregrasp_approach_only',
                configured_control_hz=1/dt,
                real_time_60hz_verified=bool(clock=='wall' and abs(dt-1/60)<1e-8 and len(timing)>0 and all(s['iteration_wall_s']<=dt for s in steps) and error is None),
                configured_deadline_verified=bool(clock=='wall' and len(timing)>0 and all(s['iteration_wall_s']<=dt for s in steps) and error is None),
                limitations=['Bounded visual approach only, no completed capture/transfer mission','Initial work pose is a reset, not an online path from stow',
                             'QP uses joint hard bounds plus separate conservative collision/braking acceptance; not a globally complete collision planner',
                             'Mass user-confirmed; USD COM and inertia unvalidated','No physical hardware or native RTX validation',
                             'ROI tracking assumes a horizontal docking cap; bootstrap global map is not refreshed during approach',
                             'Certified constant-deceleration stop may violate normal jerk limits',
                             'Wall-clock overruns are logged, not tolerated by a hardware driver; fixed simulation dt is not elapsed hardware time'])
    for name,data in [('result.json',result),('events.json',events),('trajectory.json',trajectory),('control_steps.json',steps),('sensor_frames.json',worker.rows)]:
        (folder/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    if worker.frames:
        images=[Image.fromarray(x) for x in worker.frames];images[0].save(folder/'left_camera.gif',save_all=True,append_images=images[1:],duration=67,loop=0)
    event('finished',status=result['status'],metrics=metrics,reaction=result['reaction_metrics'],p95_ms=result['control_check_p95_ms'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--clock',choices=['simulated','wall'],default='simulated')
    p.add_argument('--reaction-weight',type=float);p.add_argument('--duration',type=float);p.add_argument('--control-hz',type=float);args=p.parse_args()
    cfg=json.loads((ROOT/'configs/online_coordination.json').read_text())
    if args.reaction_weight is not None:cfg['reaction_weight']=args.reaction_weight
    if args.duration is not None:cfg['duration_s']=args.duration
    if args.control_hz is not None:
        if args.control_hz<=0:raise ValueError('Control rate must be positive')
        cfg['dt_s']=1/args.control_hz
    if cfg['reaction_weight']<0 or cfg['duration_s']<=0:raise ValueError('Invalid experiment arguments')
    with threadpool_limits(1):result=experiment(cfg,args.output,args.clock)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
