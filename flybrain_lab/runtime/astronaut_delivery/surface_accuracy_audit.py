"""Independent post-hoc triangle-distance audit, never used to alter the map."""
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from pxr import Usd,UsdGeom
from .scene import ROOT
from .vision import triangulate
from .motion import transformed


@wp.kernel
def distances(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),output:wp.array(dtype=float)):
    i=wp.tid();p=points[i];query=wp.mesh_query_point(mesh,p,100.0)
    output[i]=100.0
    if query.result:
        closest=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
        output[i]=wp.length(closest-p)


def query_distance(vertices,faces,points,device='cuda:0'):
    mesh=wp.Mesh(wp.array(np.asarray(vertices,np.float32),dtype=wp.vec3,device=device),
                 wp.array(np.asarray(faces,np.int32).ravel(),dtype=wp.int32,device=device))
    pts=wp.array(np.asarray(points,np.float32),dtype=wp.vec3,device=device);out=wp.empty(len(points),dtype=float,device=device)
    wp.launch(distances,dim=len(points),inputs=[mesh.id,pts,out],device=device)
    return out.numpy()


def truth_mesh(stage):
    verts=[];faces=[];groups=[];offset=0;cache=UsdGeom.XformCache()
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if path.startswith(('/World/LeftArm/','/World/RightArm/','/World/RightMagnet/')):continue
        if not prim.IsA(UsdGeom.Imageable):continue
        im=UsdGeom.Imageable(prim)
        if im.ComputePurpose()=='guide' or im.ComputeVisibility()=='invisible' or 'collision' in path.lower():continue
        geometry=triangulate(prim)
        if geometry is None:continue
        v,f=geometry;verts.append(transformed(v,np.array(cache.GetLocalToWorldTransform(prim))))
        faces.append(f+offset);offset+=len(v)
        groups.extend([1 if path.startswith('/World/Parts/') else 2 if path.startswith('/World/Station/') else 3]*len(f))
    return np.concatenate(verts),np.concatenate(faces),np.asarray(groups)


def area_samples(vertices,faces,groups,count,seed):
    rng=np.random.default_rng(seed);tri=vertices[faces]
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    ids=rng.choice(len(faces),count,p=areas/areas.sum());uv=rng.random((count,2));s=np.sqrt(uv[:,0])
    points=(1-s[:,None])*tri[ids,0]+(s*(1-uv[:,1]))[:,None]*tri[ids,1]+(s*uv[:,1])[:,None]*tri[ids,2]
    return points,groups[ids]


def distance_metrics(d):
    if not len(d):return dict(count=0)
    return dict(count=len(d),mean_m=float(np.mean(d)),median_m=float(np.median(d)),p95_m=float(np.quantile(d,.95)),
                maximum_m=float(np.max(d)),within_5mm=float(np.mean(d<=.005)),within_10mm=float(np.mean(d<=.010)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--samples',type=int,default=200000)
    args=p.parse_args();root=Path(args.run);wp.config.kernel_cache_dir=str(ROOT/'logs/surface_audit_warp_cache');wp.init()
    stage=Usd.Stage.Open(str(root/'final_state.usdc'));v,f,g=truth_mesh(stage)
    measured=np.load(root/'measured_surface.npz');mv,mf=measured['vertices'],measured['triangles']
    meta=json.loads((root/'measured_surface.json').read_text());cfg=meta['config']
    precision=query_distance(v,f,mv)
    # Coverage is deliberately not limited to visible faces; unknown surfaces
    # remain failures of coverage, not fabricated completion or free space.
    samples,sg=area_samples(v,f,g,args.samples,92831)
    inside=((samples>=cfg['workspace_min'])&(samples<=cfg['workspace_max'])).all(axis=1)
    samples,sg=samples[inside],sg[inside];recall=query_distance(mv,mf,samples)
    out=dict(scope='Unsigned distance to final static USD rendered triangles, excluding robot/tool meshes. Truth is audit-only. Not physical accuracy or watertightness.',
             voxel_m=cfg['voxel_size_m'],measured_vertex_precision=distance_metrics(precision),
             area_sampled_workspace_coverage=distance_metrics(recall),
             coverage_by_group={name:distance_metrics(recall[sg==i]) for i,name in [(1,'parts'),(2,'station'),(3,'other_static')]},
             sample_seed=92831,truth_samples_before_workspace_filter=args.samples,
             caveat='Parts coverage can have few samples under whole-scene area sampling; report sample count. No unseen-surface completion.',
             source_run=str(root.resolve()))
    np.savez_compressed(root/'surface_accuracy_distances.npz',measured_distance=precision,truth_sample_points=samples,truth_distance=recall,truth_group=sg)
    (root/'surface_accuracy_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
