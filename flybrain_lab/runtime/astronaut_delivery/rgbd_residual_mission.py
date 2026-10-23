"""Experimental residual policy on the unchanged RGB-D mission safety shield.

Only approach/transfer plan_step candidates are perturbed. Recovery, contact,
release and initial commissioning remain classical. No privileged scene inputs.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .rgbd_mission import RGBDMission
from .residual_ppo import ResidualPPO,ACTION_RADIUS


class RGBDResidualMission(RGBDMission):
    def __init__(self,*args,training=True,policy_checkpoint=None,policy_seed=41,**kwargs):
        if not training and not policy_checkpoint:raise ValueError('evaluation requires a trained checkpoint')
        self.policy=ResidualPPO(policy_seed,training)
        if policy_checkpoint:self.policy.load(policy_checkpoint,resume=training)
        self.initial_update_count=len(self.policy.updates)
        self.residual_profile=dict(training=training,seed=policy_seed,observation_size=107,
            radius_rad=ACTION_RADIUS,hidden=[64,64],lr=.0003,rollout=16,epochs=4,gamma=.99,gae=.95,
            checkpoint_sha256=None if not policy_checkpoint else hashlib.sha256(Path(policy_checkpoint).read_bytes()).hexdigest(),
            initialization='scratch' if not policy_checkpoint else self.policy.resume_kind,
            scope='Bounded task-projected candidate; original safety gates and classical fallback')
        self.pending=None;self.previous_action=np.zeros(16);self.policy_events=[]
        super().__init__(*args,**kwargs)
        self.policy.save(self.output/'policy_initial.pt')

    def policy_observation(self,base,target,standoff,context):
        q=self.q.ravel();v=np.zeros(16) if not self.steps else np.asarray(self.steps[-1]['v'])
        wrench=np.zeros(6) if not self.steps else np.asarray(self.steps[-1]['wrench'])/np.array([.35]*3+[.15]*3)
        phase=getattr(self,'transfer_phase',None) or 'approach'
        phases=[float(phase==name) for name in ('approach','lift','transport','place','retreat')]
        error=(target.position+np.array([0,0,standoff])-self.right.tcp(self.q[1])[:3,3])/.1
        quality=[context['sigma']/.005,context['pixels']/100.,(self.time-target.stamp)/.25]
        rejected=float(bool(self.decisions and self.decisions[-1].get('status')=='PRE_EXECUTION_MEASUREMENT_REJECTED'))
        x=np.r_[np.sin(q),np.cos(q),v,(base-q)/.3,error,quality,
            self.allocator.margins(self.q[0],context),phases,float(self.held is not None),wrench,self.previous_action/ACTION_RADIUS,rejected]
        if x.shape!=(107,) or not np.isfinite(x).all():raise ValueError('invalid policy observation')
        return np.clip(x,-10,10).astype(np.float32)

    def finish_pending(self,next_observation,terminal=False,success=False):
        if self.pending is None:return
        p=self.pending;recent=self.steps[p['step_index']:]
        progress=(p['distance']-np.linalg.norm(p['goal']-self.right.tcp(self.q[1])[:3,3]))/.02
        # Time-normalized modeled wrench cost on actually executed ticks; no truth
        # contacts or collision oracle. Completion is the observed task queue only.
        dt=np.array([r['dt_s'] for r in recent]);w=np.array([r['wrench'] for r in recent])
        reaction=0. if not len(recent) else float(np.average(np.sum((w/np.array([.35]*3+[.15]*3))**2,axis=1),weights=dt))
        decisions=self.decisions[p['decision_index']:]
        policy_rows=[r for r in decisions if r.get('proposal_mode')=='residual_ppo']
        accepted=any(r.get('robot_geometry') and r['environment']['allowed'] and r['visual']['allowed'] and r.get('visual_endpoint_reserve_passed') for r in policy_rows)
        pre_rejections=sum(r.get('status')=='PRE_EXECUTION_MEASUREMENT_REJECTED' for r in decisions)
        completed=len(self.memory.completed)-p['completed']
        reward=float(np.clip(progress,-2,2)+5*completed-.1*reaction-.25*(not accepted)-.25*pre_rejections-.05*(not len(recent)))
        if terminal:reward+=5. if success else -2.
        self.policy.finish(p['sample'],reward,next_observation,terminal)
        self.policy_events.append(dict(reward=reward,progress=float(progress),modeled_wrench_cost=reaction,
            policy_candidate_certified=accepted,pre_execution_rejections=pre_rejections,executed_ticks=len(recent),
            observed_completions=completed,terminal=terminal,sample=p['sample'],action=p['action'].tolist()))
        self.pending=None

    def residual_proposal(self,base,target,standoff,context):
        cost,scale,end,row,mode=base
        observation=self.policy_observation(end,target,standoff,context)
        self.finish_pending(observation)
        action,sample=self.policy.sample(observation);self.previous_action=action.copy()
        goal=target.position+np.array([0,0,standoff])
        self.pending=dict(sample=sample,action=action,goal=goal,distance=np.linalg.norm(goal-self.right.tcp(self.q[1])[:3,3]),
            completed=len(self.memory.completed),step_index=len(self.steps),decision_index=len(self.decisions))
        endpoint=end.copy();endpoint[:8]+=action[:8]
        options=self.right_ik_arguments();options['max_delta']=ACTION_RADIUS
        seed=np.clip(end[8:]+action[8:],-np.pi+1e-8,np.pi-1e-8)
        projected=self.right.solve(self.right.tcp(end[8:])[:3,3],self.right_task_axis(),seed=seed,tries=1,**options)
        if projected is None:return None
        endpoint[8:]=projected
        # Projection may move farther than its seed: reject, never widen residual.
        if (np.max(abs(endpoint-end))>ACTION_RADIUS+1e-8 or np.any(abs(endpoint)>np.pi)
            or np.max(abs(endpoint[:8]-self.q[0]))>self.allocator.cfg.joint_step+1e-8
            or np.max(abs(endpoint[8:]-self.q[1]))>.30+1e-8):return None
        delta=endpoint-self.q.ravel();u=np.array([.2113248654,.5,.7886751346])
        s=10*u**3-15*u**4+6*u**5;v=30*u**2-60*u**3+30*u**4;a=60*u-180*u**2+120*u**3
        W=self.reaction.wrench_batch(self.q.ravel()+s[:,None]*delta,v[:,None]*delta,a[:,None]*delta)
        cost=float(np.sum((W/np.array([.35]*3+[.15]*3))**2)+.001*np.dot(delta[:8],delta[:8]))
        return cost,1.,endpoint,dict(mode='residual_ppo',action=action.tolist(),scope='Proposal only; no motion authority'),'residual_ppo'

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        self.finish_pending(np.zeros(107,dtype=np.float32),True,bool(self.task_metadata.get('observed_queue_exhausted')))
        self.policy.save(self.output/'policy_final.pt')
        (self.output/'residual_training.json').write_text(json.dumps(dict(profile=self.residual_profile,events=self.policy_events,
            updates=self.policy.updates,new_updates=self.policy.updates[self.initial_update_count:],
            scope='On-policy development training/evaluation; not held-out performance'),indent=2),encoding='utf-8')
        return result


def main():
    p=argparse.ArgumentParser()
    for name in ('output','scene','initial-scan','scene-token','calibration-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--assume-unchanged-scene',action='store_true');p.add_argument('--checkpoint');p.add_argument('--seed',type=int,default=41)
    p.add_argument('--resume-training',action='store_true',help='Continue online learning from checkpoint; legacy checkpoints restore weights only')
    a=p.parse_args()
    with threadpool_limits(1):
        task=RGBDResidualMission(a.output,a.scene,a.initial_scan,a.scene_token,a.calibration_id,
            training=a.resume_training or a.checkpoint is None,policy_checkpoint=a.checkpoint,policy_seed=a.seed,
            assume_unchanged_scene=a.assume_unchanged_scene,mode='set',asynchronous=True,
            contact_permission=True,capture=True,local_contact_map=True,static_interval_fusion=True,local_map_cell_budget=16000000)
        task.run(48,.007,'auto')


if __name__=='__main__':main()
