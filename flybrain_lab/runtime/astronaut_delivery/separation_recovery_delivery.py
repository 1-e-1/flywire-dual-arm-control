"""Bounded extra separating-axis proposals after a conservative path rejection.

An experimental repair, NOT the controller in the frozen primary benchmark.
All candidates still need all-vertex unit-axis projection at unchanged margins.
"""
import argparse
from threadpoolctl import threadpool_limits
from .sensor_stress_delivery import StressDelivery
from .motion import CollisionGuard
from .simultaneous import SimultaneousGuard
from .convex_separation import certified_axis


class BoundedSeparationGuard(CollisionGuard):
    extra_budget=0
    extra_calls=0

    def separated(self,key,va,na,vb,nb,margin):
        if super().separated(key,va,na,vb,nb,margin):return True
        if self.extra_budget<=0:return False
        self.extra_budget-=1;self.extra_calls+=1
        axis=certified_axis(va,vb,margin)
        if axis is None:return False
        self.separation_axes[key]=axis;return True


class RetrySeparation:
    def __init__(self,guard):self.guard=guard;self.original=SimultaneousGuard(guard)
    def segment(self,*args,**kwargs):
        if self.original.segment(*args,**kwargs):return True
        self.guard.extra_budget=32
        try:return self.original.segment(*args,**kwargs)
        finally:self.guard.extra_budget=0


class SeparationRecoveryDelivery(StressDelivery):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.guard=BoundedSeparationGuard(self.stage,self.left,self.right,self.cfg['collision_margin_m'])
        self.latch.guard=self.guard;self.paired=RetrySeparation(self.guard)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--stress',default='noise2',choices=['noise2','dropout20','blank_transient','blank_permanent'])
    args=p.parse_args()
    with threadpool_limits(1):
        run=SeparationRecoveryDelivery(args.output,stress=args.stress,anytime_feasible=True,telemetry=True)
        try:run.run(3)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()
        print('extra_axis_proposals',run.guard.extra_calls,flush=True)


if __name__=='__main__':main()
