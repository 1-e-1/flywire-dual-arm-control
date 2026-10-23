"""Complete image-guided dual-arm delivery with certified streaming paths.

Reuses scene/perception/latch, never replays a stored mission. Slow planning and
continuous CAD certification are separated from the cheap trajectory executor.
This is a kinematic simulation, not a hardware real-time driver.
"""
import argparse,json,time,hashlib,gc
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .run import Delivery
from .scene import ROOT
from .online_scene import ONLINE_SCENE
from .motion import CollisionGuard
from .simultaneous import SimultaneousGuard,TightBoundArm
from .reaction import CombinedReaction,DifferentialArm,reaction_metrics,skew
from .smooth_path import SmoothPath,reaction_scale

class CooperativeDelivery(Delivery):
    def __init__(self,output,device='cuda:0',reaction=True,seed=20260908,fault_at=None,asynchronous=False,scene_path=None,task_config=None,telemetry=False,map_policy='reset'):
        super().__init__(output,device,True,'d435i',scene_override=scene_path or ONLINE_SCENE)
        if task_config is not None:self.cfg.update(task_config)
        self.cfg['seed']=seed;self.rng=np.random.default_rng(seed)
        self.coop=json.loads((ROOT/'configs/cooperative_delivery.json').read_text())
        self.cfg['control_dt_s']=self.coop['control_dt_s'];self.cfg['camera_rate_hz']=self.coop['camera_rate_hz']
        self.camera_ticks=max(1,round(1/(self.cfg['control_dt_s']*self.cfg['camera_rate_hz'])))
        self.guide.max_age=self.coop['maximum_observation_age_s'];self.guide.docking_cap_radius=self.coop['docking_cap_radius_m']
        self.source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'astronaut_delivery').glob('*.py')}
        for directory in ('classical_fusion','configs'):
            for p in (ROOT/directory).glob('*.py' if directory=='classical_fusion' else '*.json'):
                self.source_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        self.guide.reacquire_stale=True;self.guide.fit_circular_cap=True
        self.left,self.right=TightBoundArm(self.stage,'left'),TightBoundArm(self.stage,'right')
        self.guard=CollisionGuard(self.stage,self.left,self.right,self.cfg['collision_margin_m'])
        self.latch.guard=self.guard;self.paired=SimultaneousGuard(self.guard)
        self.reaction=CombinedReaction(self.stage,self.left,self.right);self.use_reaction=reaction
        self.steps=[];self.paths=[];self.sensor_rows=[];self.plan_wall=[];self.active_target=None
        self.fault_at=fault_at;self.fault_fired=False;self.stopped_smoothly=False
        self.started=time.perf_counter();self.command_times=[];self.mapping_while_moving=0
        self.vision_service=None;self.map_epoch=0;self.streaming=False;self.cycle_times=[];self.observation_ages=[]
        if map_policy not in ('reset','local'):raise ValueError('unknown map policy')
        self.map_policy=map_policy;self.map_changes=[];self.map_change_statistics=[]
        self.current_phase='initialization';self.telemetry=None
        if telemetry:
            from .telemetry import TelemetryRecorder
            self.telemetry=TelemetryRecorder(self.output)
        if asynchronous:
            from .vision_service import VisionService
            self.vision_service=VisionService(self.output,seed,device,scene_path=self.scene_path,task_config=self.cfg,telemetry=telemetry)

    def accept_observation(self,data):
        if data is None or data['epoch']!=self.map_epoch:return
        self.guide.estimates=data['estimates'];self.guide.frames=data['frames'];self.guide.map_updates=data['map_updates']
        self.camera.last_quality=data['quality'];self.last_obs=SimpleNamespace(rgb=data['rgb'].astype(np.float32)/255,depth=data['depth'],robot_mask=data['robot_mask'])
        self.mapping_while_moving=data['moving_updates']
        self.sensor_rows.append(dict(time_s=data['time_s'],received_simulation_time_s=self.time,wall_s=data['wall_s'],targets=list(data['estimates']),map_updates=data['map_updates']))
        self.observation_ages.append(self.time-data['time_s'])
        if len(self.frames)<180 and len(self.sensor_rows)%12==1:self.frames.append(data['rgb'])

    def invalidate_map(self,center=None,reason='transfer'):
        from .map_maintenance import action_region
        self.map_epoch+=1
        regions=[action_region(center)] if self.map_policy=='local' and center is not None else None
        event=dict(epoch=self.map_epoch,target=self.active_target,regions=regions,
                   policy='local' if regions is not None else 'reset',reason=reason,time_s=self.time)
        self.map_changes.append(event)
        stats=self.guide.invalidate_after_transfer(regions,self.active_target)
        if self.vision_service is None:self.map_change_statistics.append(dict(**event,statistics=stats))

    def observe(self):
        if self.vision_service is not None:
            from .motion import world
            state=dict(q=self.q.tolist(),parts={p.GetName():np.array(world(p)).tolist() for p in self.stage.GetPrimAtPath('/World/Parts').GetChildren()},
                       time_s=self.time,held=self.latch.payload.name if self.latch.payload else None,epoch=self.map_epoch,
                       moving=bool(self.steps and np.linalg.norm(self.steps[-1]['v'])>1e-5),full=not self.streaming,focus=None,
                       active_target=self.active_target,phase=self.current_phase,map_changes=self.map_changes.copy(),map_policy=self.map_policy)
            if self.latch.payload is not None:state['focus']=self.right.tcp(self.q[1])[:3,3].tolist()
            elif self.active_target in self.guide.estimates:state['focus']=self.guide.estimates[self.active_target].position.tolist()
            request=self.vision_service.submit(blocking=not self.streaming,**state)
            data=self.vision_service.poll() if self.streaming else self.vision_service.wait(request)
            self.accept_observation(data);return
        start=time.perf_counter()
        focus=None
        if self.streaming and self.latch.payload is not None:focus=self.right.tcp(self.q[1])[:3,3]
        elif self.streaming and self.active_target in self.guide.estimates:focus=self.guide.estimates[self.active_target].position
        if focus is None:obs=self.camera.capture(self.stage,self.left.camera(self.q[0]),self.time,seed=self.cfg['seed']+self.tick)
        else:obs=self.camera.capture_roi(self.stage,self.left.camera(self.q[0]),self.time,focus,seed=self.cfg['seed']+self.tick)
        if self.latch.payload is not None:
            # Known grasp identity is an action state, not an oracle pixel label.
            # Mask measured payload colour to prevent carried geometry becoming
            # static-map occupancy. The collision model still includes payload.
            paint=np.array(self.guide.palette[self.latch.payload.name]);paint/=paint.sum()
            chroma=obs.rgb/np.maximum(obs.rgb.sum(2,keepdims=True),1e-6)
            own=np.linalg.norm(chroma-paint,axis=2)<.06
            obs.robot_mask=obs.robot_mask|own
            if self.map_policy=='local':
                from .map_maintenance import mask_carried_depth
                mask_carried_depth(obs,self.right.tcp(self.q[1])[:3,3])
        before=self.guide.map_updates;self.guide.update(obs,integrate_map=True);self.last_obs=obs
        if self.steps and np.linalg.norm(self.steps[-1]['v'])>1e-5:self.mapping_while_moving+=self.guide.map_updates-before
        self.sensor_rows.append(dict(time_s=self.time,wall_s=time.perf_counter()-start,targets=list(self.guide.estimates),map_updates=self.guide.map_updates))
        if self.telemetry is not None:self.telemetry.record(obs,self.guide,self.active_target,self.q,self.current_phase,self.map_epoch)
        if self.guide.frames%12==1 and len(self.frames)<180:self.frames.append((obs.rgb*255).astype(np.uint8))

    def camera_goal(self,right_goal,phase):
        target=None
        if self.latch.payload is not None:
            # Calibrated attachment + encoder FK predicts where the held target
            # will move. No simulator world pose enters the visual estimator.
            target=(self.right.tcp(self.q[1])[:3,3]+self.right.tcp(right_goal)[:3,3])*.5
        elif self.active_target in self.guide.estimates:
            target=self.guide.estimates[self.active_target].position.copy()
        if target is None:return self.q[0].copy()
        cp=self.left.camera(self.q[0])[:3,3];offset=cp-target;distance=np.linalg.norm(offset)
        direction=offset/max(distance,1e-9);direction[2]=max(direction[2],.42);direction/=np.linalg.norm(direction)
        desired=target+direction*max(.65,min(distance,.9));desired=.7*cp+.3*desired
        axis=target-desired;axis/=np.linalg.norm(axis)
        goal=self.left.solve(desired,axis,seed=self.q[0],tries=4,rng=self.rng,
                             valid=lambda q:self.guard.check('left',q,self.q[1],moving_part=self.latch.payload.name if self.latch.payload else None)[0])
        if goal is None:return self.q[0].copy()
        delta=goal-self.q[0];goal=self.q[0]+delta*min(1.,self.coop['camera_joint_step_rad']/max(abs(delta).max(),1e-9))
        if self.use_reaction:
            # Preserve camera endpoint pose to first order; spend redundant joint
            # motion on combined momentum reduction. Acceptance still uses exact
            # FK/range and full two-arm geometric certificates below.
            state=DifferentialArm(self.left).evaluate(self.q[0])
            _,Jv,Jw=state.point('b8',self.left.tool_local[:3,3]);J=np.vstack([Jv,Jw])
            N=np.eye(8)-np.linalg.pinv(J,rcond=1e-5)@J
            A=self.reaction.matrix(self.q.reshape(16));D=np.diag([1,1,1,10,10,10])
            d=np.r_[goal-self.q[0],right_goal-self.q[1]];B=D@A[:,:8]@N
            correction=-N@np.linalg.solve(B.T@B+.1*np.eye(8),B.T@(D@A@d))
            correction*=min(1.,.035/max(abs(correction).max(),1e-9));goal=np.clip(goal+correction,-np.pi,np.pi)
        T=self.left.camera(goal);dist=np.linalg.norm(T[:3,3]-target)
        if not .45<=dist<=1.2:return self.q[0].copy()
        return goal

    def move(self,side,goal,phase,contact_part=None,support_contact=False):
        if goal is None:raise RuntimeError(phase+': no collision-free IK solution')
        start=time.perf_counter();index=0 if side=='left' else 1
        opts=dict(contact_part=contact_part,held=self.latch.payload,support_contact=support_contact)
        legacy_opts=dict(opts);legacy_opts['held']=self.latch.payload if side=='right' else None
        path=self.guard.plan(side,self.q[index],goal,self.q[1-index],seed=self.cfg['seed']+len(self.paths),**legacy_opts)
        if path is None:raise RuntimeError(phase+': no safe path')
        if len(path)>2:
            short=[path[0]];i=0
            while i<len(path)-1:
                chosen=i+1
                for j in range(len(path)-1,i+1,-1):
                    a=self.q.copy();b=self.q.copy();a[index]=path[i];b[index]=path[j]
                    if self.paired.segment(a.reshape(16),b.reshape(16),**opts):chosen=j;break
                short.append(path[chosen]);i=chosen
            path=np.array(short)
        for knot in path[1:]:
            q0=self.q.reshape(16).copy();q1=self.q.copy();q1[index]=knot
            if side=='right':q1[0]=self.camera_goal(knot,phase)
            accepted=None
            for scale in (1.,.5,.25,0.):
                candidate=q1.copy()
                if side=='right':candidate[0]=self.q[0]+scale*(q1[0]-self.q[0])
                if self.candidate_acceptable(q0,candidate.reshape(16),phase) and self.paired.segment(q0,candidate.reshape(16),**opts):accepted=candidate.reshape(16);break
            if accepted is None:
                # Re-plan against the UPDATED other-arm pose, never trust the
                # old static-other path after the camera has moved.
                repaired=self.guard.plan(side,self.q[index],knot,self.q[1-index],seed=self.cfg['seed']+1000+len(self.paths),resolution=.01,**legacy_opts)
                if repaired is None or len(repaired)<2:raise RuntimeError(phase+': simultaneous certificate unavailable')
                if len(repaired)>2:
                    for middle in repaired[1:]:self.move(side,middle,phase,contact_part,support_contact)
                    continue
                raise RuntimeError(phase+': continuous certificate unavailable')
            motion=SmoothPath.make(q0,accepted,self.coop['maximum_velocity_rad_s'],self.coop['maximum_acceleration_rad_s2'],self.coop['maximum_jerk_rad_s3'])
            if self.use_reaction:motion,_=reaction_scale(motion,self.reaction,self.coop['reaction_design_force_N'],self.coop['reaction_design_torque_Nm'],self.coop['reaction_samples_per_segment'])
            self.paths.append(dict(phase=phase,start=q0.tolist(),end=accepted.tolist(),duration_s=motion.duration,
                                   start_time_s=self.time,
                                   continuous_certificate=True,limits=motion.bounds(),held=self.latch.payload.name if self.latch.payload else None,
                                   contact_part=contact_part,support_contact=support_contact))
            self.plan_wall.append(time.perf_counter()-start)
            self.event('joint_segment_certified',phase=phase,duration_s=motion.duration,both_moving=bool(np.linalg.norm(accepted[:8]-q0[:8])>1e-6 and np.linalg.norm(accepted[8:]-q0[8:])>1e-6))
            self.execute(motion,phase)
            start=time.perf_counter()
        self.record(phase);self.event('arrived',phase=phase)

    def candidate_acceptable(self,start,end,phase):
        """Extension gate; legacy behaviour remains unchanged."""
        return True

    def execute(self,motion,phase):
        self.current_phase=phase
        enabled=gc.isenabled();gc.disable()
        try:self._execute(motion,phase)
        finally:
            self.streaming=False
            if enabled:gc.enable()

    def _execute(self,motion,phase):
        elapsed=0.;dt=self.cfg['control_dt_s'];braking=False;self.streaming=True
        pending_stop=None
        while elapsed<motion.duration-1e-10:
            cycle_start=time.perf_counter()
            if pending_stop is not None or self.estop or (self.fault_at is not None and self.time>=self.fault_at and not self.fault_fired):
                self.fault_fired=True;self.estop=False
                brake=motion.stop(elapsed,self.coop['maximum_velocity_rad_s'],self.coop['maximum_acceleration_rad_s2'],self.coop['maximum_jerk_rad_s3'])
                if brake is None:break
                motion=brake;elapsed=0.;braking=True
                self.event('smooth_safety_brake',remaining_s=motion.duration,reason=pending_stop or 'injected_or_requested_stop');pending_stop=None
            h=min(dt,motion.duration-elapsed);started=time.perf_counter()
            midpoint=motion.state(elapsed+h/2);W=self.reaction.wrench(*midpoint[:3])
            q,v,a,j=motion.state(elapsed+h);self.q=q.reshape(2,8)
            self.robot.set_positions(*self.q);self.latch.update(self.right.tcp(self.q[1]))
            self.time+=h;self.tick+=1;elapsed+=h
            command_wall=time.perf_counter()-started;self.command_times.append(command_wall)
            self.steps.append(dict(time_s=self.time,dt_s=h,q=q.tolist(),v=v.tolist(),a=a.tolist(),jerk=j.tolist(),wrench=W.tolist(),
                                   phase='safety_brake' if braking else phase,command_wall_s=command_wall))
            try:
                if not braking and self.tick%self.camera_ticks==0:self.observe()
                if not braking and self.vision_service is not None:self.accept_observation(self.vision_service.poll())
            except Exception as exc:
                pending_stop='perception_failure/'+str(exc);self.event('perception_fault',reason=str(exc))
            if self.tick%8==0:self.record('safety_brake' if braking else phase)
            cycle_wall=time.perf_counter()-cycle_start;self.cycle_times.append(cycle_wall)
            self.steps[-1]['cycle_wall_s']=cycle_wall
            if self.vision_service is not None and cycle_wall<h:time.sleep(h-cycle_wall)
        self.streaming=False
        if braking or pending_stop is not None:
            self.stopped_smoothly=True;self.record('safe_stop');raise RuntimeError('fault_injection_safe_stop')

    def run(self,count=3):
        if not 1<=count<=len(self.cfg['parts']):raise ValueError('Invalid part count')
        self.survey()  # Initial unobserved scene bootstrap only, not complete mapping.
        for spec in self.cfg['parts'][:count]:
            name=spec['name'];self.active_target=name;estimate=self.acquire_target(name)
            self.move('right',self.right_ik(estimate.position+[0,0,.11],[0,0,-1]),name+'/approach')
            self.servo_contact(name)
            # Remove pre-grasp occupancy; carried points are masked from the new
            # static map while the rest of the environment continues updating.
            observed=self.guide.estimates.get(name)
            self.invalidate_map(observed.position.copy() if observed is not None else self.right.tcp(self.q[1])[:3,3],reason='attached_source')
            offset=self.latch.payload.local_pose;pos=(self.right.tcp(self.q[1])@offset)[:3,3]
            self.move('right',self.right_ik(pos+[0,0,.08],[0,0,1],support=True,offset=offset),name+'/lift',support_contact=True)
            destination=np.array(spec['destination'])
            self.move('right',self.right_ik(destination+[0,0,.08],[0,0,1],offset=offset),name+'/transfer')
            self.move('right',self.right_ik(destination,[0,0,1],support=True,offset=offset),name+'/place',support_contact=True)
            ok,reason=self.latch.release(self.right.tcp(self.q[1]),destination)
            if not ok:raise RuntimeError('release_failed/'+reason)
            self.completed.append(name);self.record('released');self.event('released',part=name)
            if self.map_policy=='local':self.invalidate_map(destination,reason='released_destination')
            self.move('right',self.right_ik(self.right.tcp(self.q[1])[:3,3]+[0,0,.12],[0,0,-1],contact=name),name+'/retract',contact_part=name)
            if self.map_policy=='reset':self.invalidate_map(reason='released_destination')
            self.observe()
        self.event('complete',parts=self.completed)

    def save(self,error=None):
        if self.map_policy=='local' and self.vision_service is None:
            from .measured_surface import export_surface
            export_surface(self.guide.mapper,self.output)
        if self.telemetry is not None and self.vision_service is None:self.telemetry.close()
        super().save(error)
        async_result=None
        if self.vision_service is not None:
            try:async_result=self.vision_service.close()
            except Exception as exc:error=error or ('vision_shutdown_failure/'+str(exc))
        result=json.loads((self.output/'result.json').read_text())
        result['error']=error;result['status']='STOPPED' if error else 'PASS'
        timing=np.array(self.command_times);W=[s['wrench'] for s in self.steps];dt=[s['dt_s'] for s in self.steps]
        both=int(sum(np.linalg.norm(s['v'][:8])>1e-4 and np.linalg.norm(s['v'][8:])>1e-4 for s in self.steps))
        result.update(controller='cooperative_certified_stream',reaction_control=self.use_reaction,inertia=self.reaction.manifest(),
                      config=self.cfg,coordination_config=self.coop,source_sha256=self.source_hashes,
                      map_policy=self.map_policy,map_action_events=self.map_changes,
                      map_change_statistics=async_result.get('map_changes',[]) if async_result else self.map_change_statistics,
                      reaction_metrics=reaction_metrics(W,dt) if W else None,simultaneous_moving_ticks=both,total_ticks=len(self.steps),
                      map_updates_while_moving=self.mapping_while_moving,stopped_smoothly=self.stopped_smoothly,
                      command_p95_ms=float(np.percentile(timing,95)*1000) if len(timing) else None,
                      command_p99_ms=float(np.percentile(timing,99)*1000) if len(timing) else None,
                      command_max_ms=float(timing.max()*1000) if len(timing) else None,
                      planning_wall_s=self.plan_wall,total_wall_s=time.perf_counter()-self.started,
                      real_time_verified=False,validation_scope='complete_kinematic_mission_with_synchronous_sensor',
                      limitations=['Known CAD collision model; not unknown-scene/hardware certification','Synchronous sensor and slow planning excluded from command timing',
                                   'Reaction design limits checked at samples, not continuous wrench guarantees','Mass confirmed; COM/inertia not hardware calibrated',
                                   'No contact/payload/camera inertia; initial camera bootstrap may precede right motion'])
        if async_result is not None:
            cycles=np.array(self.cycle_times)
            result.update(map_points=async_result['map_points'],map_updates_while_moving=async_result['moving_updates'],
                          validation_scope='complete_kinematic_mission_with_isolated_async_sensor',
                          cycle_p95_ms=float(np.percentile(cycles,95)*1000),cycle_p99_ms=float(np.percentile(cycles,99)*1000),
                          cycle_max_ms=float(cycles.max()*1000),cycle_deadline_misses=int((cycles>self.cfg['control_dt_s']).sum()),
                          received_observation_age_p95_s=float(np.percentile(self.observation_ages,95)) if self.observation_ages else None,
                          streaming_compute_budget_verified=bool(error is None and np.all(cycles<=self.cfg['control_dt_s'])),
                          real_time_verified=False,
                          timing_scope='Whole streaming iteration, including sensing IPC and logging, excluding intentional pacing sleep and between-segment planning',
                          limitations=['Known CAD safety model, not unknown-scene or hardware certification','Slow planning occurs at C2 rest endpoints; streaming deadline is a separate gate',
                                       'Physical OS scheduling and communication not certified','Reaction design targets sampled; COM/inertia uncalibrated; tool/payload inertia excluded'])
        for name,data in [('result.json',result),('control_steps.json',self.steps),('certified_paths.json',self.paths),('sensor_frames.json',self.sensor_rows)]:
            (self.output/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps(dict(event='cooperative_result',status=result['status'],completed=self.completed,both_ticks=both,command_p95_ms=result['command_p95_ms'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--parts',type=int,default=3)
    p.add_argument('--seed',type=int,default=20260908);p.add_argument('--no-reaction',action='store_true');p.add_argument('--fault-at',type=float)
    p.add_argument('--async-sensor',action='store_true');p.add_argument('--scene');p.add_argument('--task-config');p.add_argument('--telemetry',action='store_true');args=p.parse_args()
    with threadpool_limits(1):
        task_config=json.loads(Path(args.task_config).read_text()) if args.task_config else None
        run=CooperativeDelivery(args.output,reaction=not args.no_reaction,seed=args.seed,fault_at=args.fault_at,asynchronous=args.async_sensor,
                                scene_path=args.scene,task_config=task_config,telemetry=args.telemetry)
        try:run.run(args.parts)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()

if __name__=='__main__':main()
