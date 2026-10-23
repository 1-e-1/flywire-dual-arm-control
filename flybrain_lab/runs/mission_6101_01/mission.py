"""Direct FlyWire-circuit joint proposals; NO QP, PPO, IK or teacher fallback.

RGB-D integration, measured task queue, deterministic contact state machine and
unchanged safety certificates are peripheral numerical utilities, not fly neurons.
"""
import argparse,hashlib,json,sys,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
LAB=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'runtime'))
from astronaut_delivery.rgbd_mission import RGBDMission
from astronaut_delivery.smooth_path import SmoothPath
from connectome import FlyController,observation

class FlyMission(RGBDMission):
    def __init__(self,*args,checkpoint,**kwargs):
        self.fly=FlyController(checkpoint);self.fly_events=[]
        self.residual_profile=dict(type='direct_full_supplied_flywire_rate_circuit',
            checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            neurons=self.fly.circuit.W.shape[0],edges=self.fly.circuit.W.nnz,
            no_classical_motion_proposal_fallback=True)
        super().__init__(*args,**kwargs)

    def _fly_step(self,target,standoff,look_at=None,right_rest=False):
        began=time.perf_counter();q0=self.q.ravel().copy();tcp=self.right.tcp(self.q[1])
        desired=target.position+np.array([0.,0.,standoff])
        env,snapshot=self.make_environment(target,standoff)
        if not right_rest and np.linalg.norm(desired-tcp[:3,3])<(.001 if standoff<.035 else .004):
            cert=env.segment(q0,q0)
            if not cert.allowed or not self.self_guard.box(q0,q0,**self.robot_guard_arguments())[0]:return None,'arrival_not_certified'
            self.rest_certificate=dict(certificate=asdict(cert),q=self.q.copy());return None,'arrived'
        try:context=self.allocation_context(target)
        except ValueError as exc:return None,'visual_set/'+str(exc)
        visual=context['target'] if look_at is None else np.asarray(look_at)
        goal=tcp[:3,3] if right_rest else desired
        x=observation(self.q,self.left,self.right,goal,visual,self.transfer_phase,self.held is not None)
        delta=self.fly.action(x)
        if right_rest:delta[8:]=0. # Operational task constraint, not a hidden planner.
        inference_s=time.perf_counter()-began
        for scale in (1.,.5,.25,.125):
            end=q0+scale*delta
            if np.any(abs(end)>np.pi):continue
            candidate_tcp=self.right.tcp(end[8:])
            if not right_rest:
                # Reject task divergence or orientation violation; never repair by IK.
                if np.linalg.norm(desired-candidate_tcp[:3,3])>=np.linalg.norm(desired-tcp[:3,3])-1e-6:continue
                if self.held is not None or self.transfer_phase=='retreat':
                    if np.linalg.norm(candidate_tcp[2,:3]-self.source_tcp[2,:3])>.001:continue
                elif np.linalg.norm(candidate_tcp[:3,2]-self.right_task_axis())>.025:continue
            c=env.segment(q0,end);v=self.certify_visual_path(q0[:8],end[:8],context)
            reserve=self.allocator.endpoint_valid(q0[:8],end[:8],context)
            robot=self.self_guard.segment(q0,end,**self.robot_guard_arguments()) if c.allowed and v.allowed else False
            self.decisions.append(dict(time_s=self.time,proposal_mode='flywire_direct',scale=scale,
                environment=asdict(c),visual=v.dictionary(),robot_geometry=robot,
                visual_endpoint_reserve_passed=reserve,inference_and_setup_s=inference_s))
            if c.allowed and v.allowed and reserve and robot and np.linalg.norm(end-q0)>1e-7:
                motion=SmoothPath.make(q0,end,minimum=1.2)
                self.paths.append(dict(start=q0.tolist(),end=end.tolist(),duration_s=motion.duration,time_s=self.time,
                    target=target.position.tolist(),standoff_m=standoff,map_revision=snapshot.revision,
                    environment_certificate=asdict(c),visual_certificate=v.dictionary(),robot_certificate=True,
                    visual_target_name=self.active,visual_target_position=context['target'].tolist(),
                    controller='flywire_direct',scope='Neural joint proposal; original continuous RGBD safety gates'))
                self.fly_events.append(dict(time_s=self.time,accepted=True,scale=scale,inference_and_setup_s=inference_s))
                return motion,'accepted'
        self.fly_events.append(dict(time_s=self.time,accepted=False,delta=delta.tolist(),inference_and_setup_s=inference_s))
        return None,'no_certified_flywire_action'

    def plan_step(self,target,standoff,step_m=.02):
        return self._fly_step(target,standoff)

    def plan_observation_recovery(self,target,look_at=None,require_view_gain=False,**kwargs):
        if self.recovery_count>=6:return None
        motion,_=self._fly_step(target,.10,look_at=look_at,right_rest=True)
        if motion is not None:
            self.recovery_count+=1;self.recovery_count_total+=1
            self.paths[-1]['phase']='flywire_observation_recovery'
        return motion

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        (self.output/'flywire_controller.json').write_text(json.dumps(dict(events=self.fly_events,
            profile=self.residual_profile,scope='Circuit proposes all approach/travel/recovery joint actions; classical perception, state machine, safety braking retained'),indent=2),encoding='utf-8')
        return result

def main():
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--seed',type=int,default=6101);a=p.parse_args()
    parent=LAB.parent;base=parent/('reports/rgbd_paired_mission_pilot_01' if a.seed==6101 else 'reports/rgbd_holdout_01')
    output=Path(a.output).resolve()
    if not output.is_relative_to(LAB):raise ValueError('all experiment output must stay in flybrain_lab')
    with threadpool_limits(1):
        task=FlyMission(output,str(base/f'scene_{a.seed}/rgbd_random_{a.seed}.usda'),str(base/f'recording_{a.seed}.json'),
            f'rgbd_random_{a.seed}','660ece523b85804d9a24fc37be455e4dbcfe9b6a178be2e0bd4254e2488e05f1',
            checkpoint=a.checkpoint,assume_unchanged_scene=True,mode='set',asynchronous=True,
            contact_permission=True,capture=True,local_contact_map=True,static_interval_fusion=True,local_map_cell_budget=16000000)
        for file in ('mission.py','connectome.py','train.py'):
            (output/file).write_bytes((LAB/file).read_bytes())
        task.run(48,.007,'auto')
if __name__=='__main__':main()
