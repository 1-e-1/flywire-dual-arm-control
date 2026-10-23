"""SIMULATION pre-deployment RGB-D recording, not online autonomous surveying.

An external commissioning scanner supplies fixed, declared optical poses.
Only removable robot hardware is absent; astronaut and environmental objects
are retained. Environment meshes are used solely by the sensor renderer.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
from threadpoolctl import threadpool_limits
from .d435i import D435iCamera
from .online_scene import ONLINE_SCENE


def optical_pose(position,target):
    p=np.asarray(position,float);forward=np.asarray(target,float)-p
    forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,forward);T=np.eye(4);T[:3,:3]=np.column_stack([right,up,-forward]);T[:3,3]=p
    return T


def record(output,scene=None,repeat=3,device='cuda:0',coverage_report=None,dense_workspace=False,overhead_only=False,start_time=0.,right_corridor_only=False):
    if not np.isfinite(start_time) or start_time<0:raise ValueError('nonnegative acquisition time required')
    folder=Path(output);folder.mkdir(parents=True,exist_ok=False)
    source=Path(scene or ONLINE_SCENE);stage=Usd.Stage.Open(str(source));stage.SetEditTarget(stage.GetSessionLayer())
    excluded=['/World/LeftArm','/World/RightArm','/World/RightMagnet','/World/C_structure']
    for path in excluded:UsdGeom.Imageable(stage.GetPrimAtPath(path)).CreateVisibilityAttr().Set('invisible')
    camera=D435iCamera(device);target=[.25,0.,1.05]
    positions=[[1.25,-.95,1.05],[1.25,0.,1.05],[1.25,.95,1.05],
               [.25,.85,1.05],[-.75,.95,1.05],[-.75,0.,1.05],[-.75,-.95,1.05],[.25,-.85,1.05],
               [1.1,-.75,1.9],[1.1,.75,1.9],[-.5,.75,1.9],[-.5,-.75,1.9]]
    targets=[target for _ in positions]
    # Explicit extra commissioning observations of the robot installation
    # workspace, not hidden object poses or a truth-derived free-space mask.
    positions += [[.7,.7,1.25],[.7,-.7,1.25],[.1,.85,1.3],[.1,-.85,1.3]]
    targets += [[.01,.33,.97],[.01,-.33,.97],[.01,.33,.97],[.01,-.33,.97]]
    for side in (-1,1):
        base=np.array([.01,side*.33,.97])
        for dx in (.45,.8):
            for dy in (.32,.55):
                for dz in (.02,.35):
                    positions.append((base+[dx,side*dy,dz]).tolist());targets.append(base.tolist())
    for target_tool in ([.55,.55,1.10],[.55,-.45,1.10]):
        t=np.array(target_tool)
        for offset in ([.55,0,.3],[-.55,0,.3],[0,.45,.35],[0,-.45,.35],[.4,.3,.1],[.4,-.3,.1]):
            positions.append((t+offset).tolist());targets.append(t.tolist())
    if dense_workspace or overhead_only:
        if overhead_only:positions=[];targets=[]
        # Frozen workcell lattice, independent of part positions and of which
        # test trial failed. Every scene uses exactly this acquisition budget.
        for x in (.3,.65):
            for y in (-.55,-.2,.3,.6):
                for z in (.9,1.3):
                    t=np.array([x,y,z])
                    offsets=([.08,0,.65],[-.08,0,.65])
                    if not overhead_only:offsets=([.6,0,.25],[-.6,0,.25],[0,.6,.25],[0,-.6,.25],[.45,.45,.35],[.45,-.45,.35])+offsets
                    for offset in offsets:
                        positions.append((t+offset).tolist());targets.append(t.tolist())
    if right_corridor_only:
        positions=[];targets=[]
        # Fixed installation-workspace lattice, not object poses and not
        # selected successful trajectories. Extra raw acquisition is reported.
        for x in (.2,.4,.6):
            for y in (-.55,-.35,-.15):
                for z in (1.1,1.3):
                    t=np.array([x,y,z])
                    for offset in ([.55,0,.10],[.40,.35,.10],[.40,-.35,.10],[.12,0,.55]):
                        positions.append((t+offset).tolist());targets.append(t.tolist())
    # Optional evidence-directed commissioning: targets are rejected measured
    # map cells, never simulator object positions or ground-truth surfaces.
    coverage_source=None
    if coverage_report:
        payload=Path(coverage_report).read_bytes();report=json.loads(payload)
        if report.get('environment_authority')!='RGBD_ONLY':raise ValueError('need RGB-D coverage diagnostic')
        points=[]
        for row in report.get('candidates',[]):
            diagnostic=row.get('rejection_diagnostic') or {}
            points.extend(diagnostic.get('voxel_centers_m',[]))
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('no valid measured coverage targets')
        unique=np.unique(points,axis=0)[:8]
        for t in unique:
            for offset in ([.5,0,.2],[-.5,0,.2],[0,.45,.2],[0,-.45,.2],[.4,.3,.15],[.4,-.3,.15]):
                positions.append((t+offset).tolist());targets.append(t.tolist())
        coverage_source=dict(file=str(Path(coverage_report).resolve()),sha256=hashlib.sha256(payload).hexdigest(),targets_m=unique.tolist())
    rows=[]
    for index,position in enumerate(positions):
        T=optical_pose(position,targets[index])
        for k in range(repeat):
            i=len(rows);obs=camera.capture(stage,T,start_time+i/30,seed=12000+i);path=folder/f'rgbd_{i:03d}.npz'
            np.savez_compressed(path,rgb=obs.rgb,depth=obs.depth,K=obs.K,camera_world=T,exposure_s=obs.stamp)
            rows.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),view=index,
                valid_depth_fraction=float(np.mean(obs.depth>0))))
        print(json.dumps(dict(view=index,frames=len(rows),valid_fraction=rows[-1]['valid_depth_fraction'])),flush=True)
    calibration=hashlib.sha256(json.dumps(camera.profile,sort_keys=True).encode()).hexdigest()
    result=dict(format='rgbd-initial-map-v1',scene_token=source.stem,calibration_id=calibration,records=rows,
        acquisition='SIMULATION external pre-deployment commissioning scanner; not physical data and not an online left-arm trajectory',
        declared_optical_positions_m=positions,look_at_m=targets,excluded_robot_hardware=excluded,
        measured_coverage_target_source=coverage_source,
        fixed_dense_workspace_protocol=dense_workspace,
        overhead_only=overhead_only,acquisition_start_s=start_time,
        fixed_right_corridor_protocol=right_corridor_only,
        environment_CAD_exported=False,renderer_instance_labels_exported=False,
        assumptions=['Environmental surfaces remain unchanged from commissioning to startup.',
            'Only listed known robot hardware is subsequently installed; this hardware needs independent robot-only collision checking.',
            'Optical scan poses are externally supplied initial-data acquisition conditions, not claimed as autonomously reached safe poses.'])
    (folder/'recording.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene');p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--coverage-report');p.add_argument('--dense-workspace',action='store_true');p.add_argument('--overhead-only',action='store_true');p.add_argument('--right-corridor-only',action='store_true');p.add_argument('--start-time',type=float,default=0.);args=p.parse_args()
    if args.repeat<1:raise ValueError('positive repeats required')
    with threadpool_limits(1):record(args.output,args.scene,args.repeat,coverage_report=args.coverage_report,dense_workspace=args.dense_workspace,overhead_only=args.overhead_only,start_time=args.start_time,right_corridor_only=args.right_corridor_only)


if __name__=='__main__':main()
