"""Separate resolution ablation; frozen benchmark controller remains untouched."""
import argparse,json
from pathlib import Path
from dataclasses import replace
from threadpoolctl import threadpool_limits
from .sufficiency_delivery import SufficiencyDelivery
from classical_fusion.planning_map import PlanningTSDF


class HighResolutionDelivery(SufficiencyDelivery):
    def __init__(self,*args,voxel_m=.004,adaptive_free=False,**kwargs):
        if kwargs.get('asynchronous'):raise ValueError('Resolution ablation uses synchronous mapper only')
        super().__init__(*args,**kwargs)
        if not .002<=voxel_m<=.012:raise ValueError('Unsupported resolution')
        cfg=replace(self.guide.mapper.cfg,voxel_size_m=voxel_m,truncation_m=3*voxel_m)
        if adaptive_free:
            from .adaptive_planning_map import AdaptivePlanningTSDF
            self.guide.mapper=AdaptivePlanningTSDF(cfg,device='cuda:0')
        else:self.guide.mapper=PlanningTSDF(cfg,device='cuda:0')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--voxel-m',type=float,default=.004)
    p.add_argument('--adaptive-free',action='store_true');args=p.parse_args()
    with threadpool_limits(1):
        run=HighResolutionDelivery(args.output,voxel_m=args.voxel_m,adaptive_free=args.adaptive_free,anytime_feasible=True,telemetry=True)
        try:run.run(3)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()
        path=Path(args.output)/'result.json';result=json.loads(path.read_text())
        result['resolution_ablation']=dict(voxel_m=args.voxel_m,adaptive_free=args.adaptive_free,map_device='cuda:0',known_CAD_remains_collision_authority=True)
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
