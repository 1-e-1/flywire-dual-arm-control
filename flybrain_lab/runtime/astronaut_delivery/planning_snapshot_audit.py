"""Offline utility and false-free probes for the exact saved planning evidence."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pxr import Usd
from classical_fusion.planning_map import PlanningSnapshot,PlanningMapConfig
from .surface_accuracy_audit import truth_mesh,area_samples


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();root=Path(args.run)
    z=np.load(root/'planning_map_state.npz');cfg=PlanningMapConfig(**json.loads((root/'measured_surface.json').read_text())['config'])
    free=(z['free_views']>=cfg.minimum_free_views)&~z['surface_veto']&((z['weight']==0)|(z['distance']>0))
    mapper=SimpleNamespace(cfg=cfg,revision=0,latest_observation_time=time.monotonic(),
                           lower=torch.from_numpy(z['lower']),spacing=torch.from_numpy(z['spacing']),shape=free.shape)
    snapshot=PlanningSnapshot(mapper,free);rng=np.random.default_rng(19321)
    ids=np.argwhere(free);ids=ids[rng.choice(len(ids),min(100000,len(ids)),replace=False)] if len(ids) else ids
    points=z['lower']+(ids+.5)*z['spacing'];clearance,valid=snapshot.clearance(points,now_s=snapshot.built_at)
    vertices,faces,groups=truth_mesh(Usd.Stage.Open(str(root/'final_state.usdc')))
    surface,_=area_samples(vertices,faces,groups,200000,734)
    inside=((surface>=cfg.workspace_min)&(surface<=cfg.workspace_max)).all(axis=1);surface=surface[inside]
    direction=rng.normal(size=surface.shape);direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    radius=.010;centers=surface+direction*rng.uniform(0,.009,(len(surface),1))
    # Each 10-mm sphere contains the sampled exact CAD surface point by
    # construction; this needs no unreliable whole-mesh inside/outside sign.
    probe_clearance,_=snapshot.clearance(centers,now_s=snapshot.built_at)
    result=dict(scope='Saved evidence snapshot utility/known-colliding sphere probes; audit clock is frozen. Not an online full-arm trajectory certificate.',
        known_free_cells=int(free.sum()),total_cells=free.size,known_free_fraction=float(free.mean()),
        sampled_free_centers=len(points),maximum_sampled_clearance_m=float(clearance.max()) if len(clearance) else 0.,
        sphere_admissible_centers={str(r):int(np.sum(clearance>r)) for r in (.005,.010,.020,.040)},
        known_colliding_probes=len(centers),incorrectly_allowed_colliding_probes=int(np.sum(probe_clearance>radius)),
        probe_radius_m=radius,unknown_is_blocked=True,
        caveat='Zero false-free probes is weak evidence when map utility is near zero. Disconnected free centers do not imply a feasible full-link path.')
    (root/'planning_snapshot_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
