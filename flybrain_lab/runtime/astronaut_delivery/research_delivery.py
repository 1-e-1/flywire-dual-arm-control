"""Research CLI defaults to the strict RGB-D commissioning gate.

ResearchDelivery below is the preserved historical CAD integration. It is
reachable from this CLI only with --known-cad-baseline, never as a fallback.
"""
import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .sufficiency_delivery import SufficiencyDelivery
from .separation_recovery_delivery import BoundedSeparationGuard,RetrySeparation
from .configured_vision_service import ConfiguredVisionService


class ResearchDelivery(SufficiencyDelivery):
    def __init__(self,*args,**kwargs):
        if kwargs.pop('asynchronous',False):raise ValueError('This entry always uses its configured asynchronous worker')
        super().__init__(*args,**kwargs)
        self.guard=BoundedSeparationGuard(self.stage,self.left,self.right,self.cfg['collision_margin_m'])
        self.latch.guard=self.guard;self.paired=RetrySeparation(self.guard)
        self.vision_service=ConfiguredVisionService(self.output,self.cfg['seed'],scene_path=self.scene_path,task_config=self.cfg,telemetry=self.telemetry is not None)

    def save(self,error=None):
        super().save(error);path=self.output/'result.json';r=json.loads(path.read_text())
        r['research_configuration']=dict(async_high_resolution_map=True,voxel_m=.004,adaptive_free_patch=True,
             per_voxel_exact_view_provenance=True,additional_axis_budget_per_retry=32,
             additional_axis_calls=self.guard.extra_calls,primary_collision_authority='KNOWN_CAD',
             evaluated_in_primary_frozen_batch=False)
        path.write_text(json.dumps(r,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=20260908)
    p.add_argument('--scene');p.add_argument('--task-config')
    p.add_argument('--initial-scan');p.add_argument('--scene-token');p.add_argument('--calibration-id')
    p.add_argument('--assume-unchanged-scene',action='store_true')
    p.add_argument('--known-cad-baseline',action='store_true',help='Explicit historical comparison ONLY; does not satisfy the RGB-D-only requirement')
    args=p.parse_args()
    if not args.known_cad_baseline:
        if args.task_config:raise ValueError('Strict RGB-D startup does not accept legacy task object metadata')
        from .rgbd_bootstrap import run as strict_startup
        with threadpool_limits(1):result=strict_startup(args.output,scene=args.scene,initial_scan=args.initial_scan,
            scene_token=args.scene_token,calibration_id=args.calibration_id,assume_unchanged_scene=args.assume_unchanged_scene)
        print('RGB-D-only commissioning gate; full delivery is not yet certified. No environment CAD fallback.',flush=True)
        raise SystemExit(2 if result['status']=='BLOCKED_INITIAL_RGBD_COVERAGE' else 3)
    if args.initial_scan:raise ValueError('Do not combine RGB-D commissioning evidence with the historical CAD baseline')
    cfg=json.loads(Path(args.task_config).read_text()) if args.task_config else None
    with threadpool_limits(1):
        run=ResearchDelivery(args.output,seed=args.seed,scene_path=args.scene,task_config=cfg,anytime_feasible=True,telemetry=True)
        try:run.run(3)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()


if __name__=='__main__':main()
