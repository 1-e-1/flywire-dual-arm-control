"""Independent environment truth audit, never imported by the planner."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pxr import Usd,UsdGeom
from classical_fusion.planning_map import PlanningMapConfig,PlanningSnapshot
from .surface_accuracy_audit import truth_mesh,area_samples,query_distance,distance_metrics


def audit(run,scene,static_interval_fusion=False,motion_run=None,persistent_surface=False):
    import warp as wp
    from .scene import ROOT
    wp.config.kernel_cache_dir=str(ROOT/'logs/surface_audit_warp_cache')
    wp.init()
    root=Path(run);meta=json.loads((root/'measured_surface.json').read_text());cfg=PlanningMapConfig(**meta['config'])
    z=np.load(root/'planning_map_state.npz');free=(z['free_views']>=cfg.minimum_free_views)&((z['weight']==0)|(z['distance']>0))
    if not static_interval_fusion:free &= ~z['surface_veto']
    mapper=SimpleNamespace(cfg=cfg,revision=0,latest_observation_time=time.monotonic(),lower=torch.from_numpy(z['lower']),spacing=torch.from_numpy(z['spacing']),shape=free.shape)
    snapshot=PlanningSnapshot(mapper,free)
    stage=Usd.Stage.Open(str(Path(scene).resolve()));stage.SetEditTarget(stage.GetSessionLayer())
    moved=[]
    state_file=root/'scene_state_at_export.json'
    if state_file.exists() or motion_run:
        from .audit_scene_state import apply_recorded_final_state
        moved=apply_recorded_final_state(stage,json.loads((state_file if state_file.exists() else Path(motion_run)/'result.json').read_text()))
    # C is the robot mounting interface, not an environmental obstacle. The
    # astronaut is NOT hidden; its true surface remains in this audit.
    UsdGeom.Imageable(stage.GetPrimAtPath('/World/C_structure')).CreateVisibilityAttr().Set('invisible')
    vertices,faces,groups=truth_mesh(stage);surface,labels=area_samples(vertices,faces,groups,200000,5531)
    inside=((surface>=cfg.workspace_min)&(surface<=cfg.workspace_max)).all(1);surface,labels=surface[inside],labels[inside]
    rng=np.random.default_rng(781);direction=rng.normal(size=surface.shape);direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    centers=surface+direction*rng.uniform(0,.009,(len(surface),1));clear,_=snapshot.clearance(centers,now_s=snapshot.built_at)
    selected=np.argwhere(free)
    if len(selected)>100000:selected=selected[rng.choice(len(selected),100000,replace=False)]
    p=z['lower']+(selected+.5)*z['spacing'];utility,_=snapshot.clearance(p,now_s=snapshot.built_at)
    mesh=np.load(root/('persistent_measured_surface.npz' if persistent_surface else 'measured_surface.npz'))
    precision=query_distance(vertices,faces,mesh['vertices'])
    recall=query_distance(mesh['vertices'],mesh['triangles'],surface)
    part_surface,_=area_samples(vertices,faces[groups==1],groups[groups==1],50000,8187)
    part_surface=part_surface[((part_surface>=cfg.workspace_min)&(part_surface<cfg.workspace_max)).all(1)]
    part_recall=query_distance(mesh['vertices'],mesh['triangles'],part_surface)
    dense_ids=np.floor((part_surface-z['lower'])/z['spacing']).astype(int)
    valid_ids=((dense_ids>=0)&(dense_ids<np.array(free.shape))).all(1);dense_ids=dense_ids[valid_ids]
    false_free=int(free[tuple(dense_ids.T)].sum()) if len(dense_ids) else 0
    # Equal dense samples per object prevent a large object hiding a missed
    # small one. This is audit-only and never supplies a controller count.
    from .vision import triangulate
    from .motion import transformed
    per_object={};cache=UsdGeom.XformCache()
    parts=stage.GetPrimAtPath('/World/Parts')
    for body in (parts.GetChildren() if parts else []):
        vv=[];ff=[];offset=0
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Imageable):continue
            im=UsdGeom.Imageable(prim)
            if im.ComputeVisibility()=='invisible' or im.ComputePurpose()=='guide' or 'collision' in str(prim.GetPath()).lower():continue
            geometry=triangulate(prim)
            if geometry is None:continue
            v,f=geometry;vv.append(transformed(v,np.array(cache.GetLocalToWorldTransform(prim))));ff.append(f+offset);offset+=len(v)
        if not vv:continue
        v,f=np.concatenate(vv),np.concatenate(ff)
        samples,_=area_samples(v,f,np.zeros(len(f),int),20000,8188)
        inside=((samples>=cfg.workspace_min)&(samples<cfg.workspace_max)).all(1)
        distances=query_distance(mesh['vertices'],mesh['triangles'],samples[inside]) if inside.any() else np.zeros(0)
        per_object[body.GetName()]=dict(samples_before_ROI=20000,ROI_fraction=float(inside.mean()),coverage=distance_metrics(distances))
    report=dict(scope='Offline saved evidence audit with frozen query clock; actual environment USD is audit-only. Not an online path certificate.',
        excluded_from_environment=['LeftArm','RightArm','RightMagnet','C_structure'],astronaut_retained=True,
        final_motion_run=None if motion_run is None else str(motion_run),actual_moved_objects=moved,persistent_surface=persistent_surface,
        scene_state_at_export=str(state_file) if state_file.exists() else None,per_object_coverage=per_object,
        known_free_fraction=float(free.mean()),known_colliding_10mm_sphere_probes=len(centers),
        incorrectly_allowed_colliding_probes=int(np.sum(clear>.01)),
        tested_free_centers=len(p),sphere_admissible_centers={str(r):int(np.sum(utility>r)) for r in (.01,.02,.04)},
        surface_precision=distance_metrics(precision),surface_coverage=distance_metrics(recall),
        parts_coverage=distance_metrics(recall[labels==1]),
        static_interval_fusion=static_interval_fusion,
        dense_parts_coverage=distance_metrics(part_recall),
        dense_true_part_surface_voxels_tested=len(dense_ids),true_part_surface_samples_in_certified_free=false_free,
        caveat='Finite collision probes do not prove universal safety; free-center utility is not a full robot trajectory. Truth is never used to fill the map.')
    name='rgbd_static_interval_audit' if static_interval_fusion else 'rgbd_environment_audit'
    if persistent_surface:name+='_persistent'
    (root/(name+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);p.add_argument('--static-interval-fusion',action='store_true');p.add_argument('--motion-run');p.add_argument('--persistent-surface',action='store_true');args=p.parse_args();audit(args.run,args.scene,args.static_interval_fusion,args.motion_run,args.persistent_surface)


if __name__=='__main__':main()
