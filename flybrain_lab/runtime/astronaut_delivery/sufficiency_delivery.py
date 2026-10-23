"""Opt-in full mission with visual-set allocation, retaining legacy safety gates."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .cooperative import CooperativeDelivery
from .visual_sufficiency import VisualSetAllocator,SufficiencyConfig


class SufficiencyDelivery(CooperativeDelivery):
    def __init__(self,*args,allocation='set',objective='wrench',anytime_feasible=False,map_policy='local',recovery_attempts=1,continuous_visual=True,**kwargs):
        super().__init__(*args,map_policy=map_policy,**kwargs)
        if recovery_attempts not in (0,1):raise ValueError('recovery budget must be zero or one')
        self.recovery_attempts=recovery_attempts
        self.continuous_visual=continuous_visual;self.visual_certificates=[]
        self.allocator=VisualSetAllocator(self.left,self.reaction,SufficiencyConfig(objective=objective,accept_feasible_iterate=anytime_feasible));self.allocation=allocation;self.allocations=[];self.visual_context=None

    def servo_contact(self,name):
        for attempt in range(self.recovery_attempts+1):
            try:return super().servo_contact(name)
            except RuntimeError as exc:
                if str(exc)!='visual_servo_stale_hold/'+name or attempt>=self.recovery_attempts:raise
                # Only a stale measurement at a C2 rest endpoint may trigger
                # active reacquisition; never catch geometry, latch or IPC faults.
                if self.steps and (np.linalg.norm(self.steps[-1]['v'])>1e-8 or np.linalg.norm(self.steps[-1]['a'])>1e-8):raise
                self.visual_context=None
                self.event('bounded_visual_recovery',part=name,attempt=attempt+1)
                self.acquire_target(name)  # existing finite-view, certified survey

    def camera_goal(self,right_goal,phase):
        self.visual_context=None
        target=self.guide.target(self.active_target,self.time)
        # Once attached, the painted docking cap may intentionally be covered.
        # This first implementation releases DOF in visually guided pick phases.
        # Transport retains the already verified legacy camera controller.
        if self.latch.payload is None and target is not None:
            try:context=self.allocator.context(self.q[0],target.position,target.covariance,target.pixels,self.time-target.stamp)
            except ValueError as exc:self.allocations.append(dict(time_s=self.time,phase=phase,eligible=False,reason=str(exc)))
            else:
                goal,row=self.allocator.propose(self.q,right_goal,context,self.allocation)
                self.visual_context=context;row.update(time_s=self.time,phase=phase,eligible=True)
                row['measurement']=dict(position=target.position.tolist(),covariance=target.covariance.tolist(),
                                        pixels=target.pixels,stamp=target.stamp)
                self.allocations.append(row);return goal
        else:self.allocations.append(dict(time_s=self.time,phase=phase,eligible=False,reason='held_or_no_fresh_target'))
        return super().camera_goal(right_goal,phase)

    def candidate_acceptable(self,start,end,phase):
        if self.visual_context is None:return True
        if not self.continuous_visual:return self.allocator.path_valid(np.asarray(start)[:8],np.asarray(end)[:8],self.visual_context)
        from .continuous_visual import certify
        result=certify(self.allocator,np.asarray(start)[:8],np.asarray(end)[:8],self.visual_context)
        self.visual_certificates.append(dict(time_s=self.time,phase=phase,**result.dictionary()))
        return result.allowed

    def move(self,side,goal,phase,contact_part=None,support_contact=False):
        self.visual_context=None
        return super().move(side,goal,phase,contact_part,support_contact)

    def save(self,error=None):
        super().save(error)
        (self.output/'visual_allocations.json').write_text(json.dumps(self.allocations,indent=2),encoding='utf-8')
        (self.output/'visual_certificates.json').write_text(json.dumps(self.visual_certificates,indent=2),encoding='utf-8')
        path=self.output/'result.json';result=json.loads(path.read_text());result['controller']='visual_sufficiency_'+self.allocation
        result['visual_constraint_scope']='Fresh unheld pick targets only; interval-certified geometric proxy when enabled. Legacy fallback otherwise; no continuous occlusion guarantee.'
        result['continuous_geometric_visual_gate']=self.continuous_visual
        result['allocation_objective']=self.allocator.cfg.objective
        result['stale_visual_recovery_budget']=self.recovery_attempts
        result['sufficiency_config']=asdict(self.allocator.cfg)
        result['minimum_visual_quality_proven']=False;result['recursive_feasibility_proven']=False
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--allocation',choices=['set','pose'],default='set');p.add_argument('--seed',type=int,default=20260908)
    p.add_argument('--scene');p.add_argument('--task-config');p.add_argument('--async-sensor',action='store_true');p.add_argument('--telemetry',action='store_true')
    p.add_argument('--objective',choices=['momentum','wrench'],default='wrench');p.add_argument('--anytime-feasible',action='store_true')
    p.add_argument('--map-policy',choices=['reset','local'],default='local');p.add_argument('--fault-at',type=float)
    p.add_argument('--recovery-attempts',type=int,choices=[0,1],default=1);p.add_argument('--sampled-visual',action='store_true');args=p.parse_args()
    cfg=json.loads(Path(args.task_config).read_text()) if args.task_config else None
    with threadpool_limits(1):
        run=SufficiencyDelivery(args.output,allocation=args.allocation,objective=args.objective,anytime_feasible=args.anytime_feasible,map_policy=args.map_policy,recovery_attempts=args.recovery_attempts,continuous_visual=not args.sampled_visual,fault_at=args.fault_at,seed=args.seed,scene_path=args.scene,task_config=cfg,asynchronous=args.async_sensor,telemetry=args.telemetry)
        try:run.run(3)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()

if __name__=='__main__':main()
