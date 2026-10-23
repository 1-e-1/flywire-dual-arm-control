"""Post-freeze regression for accumulated exact-view pixel evidence."""
import argparse,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .postscan_delivery import PostScanDelivery
from .temporal_evidence_map import TemporalEvidenceTSDF


class TemporalPostScanDelivery(PostScanDelivery):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.guide.mapper=TemporalEvidenceTSDF(self.guide.mapper.cfg,device='cuda:0')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    with threadpool_limits(1):
        run=TemporalPostScanDelivery(args.output,anytime_feasible=True,telemetry=True)
        try:run.run(3)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()
        path=Path(args.output)/'result.json';r=json.loads(path.read_text())
        r['temporal_evidence']=dict(exact_pose_intrinsics_only=True,minimum_valid_depth=True,
            scene_change_cache_clear=True,maximum_cached_images=8,CAD_collision_authority=True,
            additional_postscan_time_is_not_free=True)
        path.write_text(json.dumps(r,indent=2),encoding='utf-8')


if __name__=='__main__':main()
