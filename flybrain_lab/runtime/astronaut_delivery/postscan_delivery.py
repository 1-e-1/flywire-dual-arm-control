"""Measured-map coverage diagnostic: bounded, certified post-delivery views.

Reported separately from simultaneous delivery time; never presented as free
coverage obtained during the original mission or as unknown-world safety.
"""
import argparse,json
import numpy as np
from threadpoolctl import threadpool_limits
from .high_resolution_delivery import HighResolutionDelivery
from .separation_recovery_delivery import BoundedSeparationGuard,RetrySeparation


class PostScanDelivery(HighResolutionDelivery):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,adaptive_free=True,**kwargs)
        self.guard=BoundedSeparationGuard(self.stage,self.left,self.right,self.cfg['collision_margin_m'])
        self.latch.guard=self.guard;self.paired=RetrySeparation(self.guard)

    def observe_hold(self,frames=10):
        ticks=int(np.ceil(.51/self.cfg['control_dt_s']))
        for _ in range(frames):
            self.tick+=ticks;self.time+=ticks*self.cfg['control_dt_s'];self.observe()

    def run(self,count=3):
        super().run(count);self.active_target=None;began=self.time;results=[]
        # Workspace fixture prior, not part poses or an unseen geometry oracle.
        target=np.asarray(self.camera.profile['workspace_look_at_m'])
        views=self.camera.profile['view_candidates_m']+[[.70,.30,1.30]]
        for i,position in enumerate(views):
            position=np.asarray(position);direction=target-position;direction/=np.linalg.norm(direction)
            q=self.left.solve(position,direction,seed=self.q[0],tries=6,rng=self.rng,
                 valid=lambda x:self.guard.check('left',x,self.q[1])[0])
            if q is None:results.append(dict(view=i,status='IK_REJECTED'));continue
            try:self.move('left',q,f'postscan_{i}')
            except RuntimeError as exc:
                if self.steps and (np.linalg.norm(self.steps[-1]['v'])>1e-8 or np.linalg.norm(self.steps[-1]['a'])>1e-8):raise
                results.append(dict(view=i,status='MOTION_REJECTED',reason=str(exc)));continue
            # Explicit simulated observation time; exact extrinsics permit
            # complementing missing pixels without inventing independent views.
            self.observe_hold()
            results.append(dict(view=i,status='OBSERVED',map_updates=self.guide.map_updates))
        self.event('postscan_complete',elapsed_s=self.time-began,views=results,
                   scope='Additional bounded post-delivery mapping, excluded from original cooperative mission latency')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    with threadpool_limits(1):
        run=PostScanDelivery(args.output,anytime_feasible=True,telemetry=True)
        try:run.run(3)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()


if __name__=='__main__':main()
