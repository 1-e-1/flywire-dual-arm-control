"""Read-only release rejection geometry diagnostic; no evidence or motion writes."""
import argparse,json,sys
from pathlib import Path
from itertools import combinations,product
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from pxr import Usd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from astronaut_delivery.simultaneous import TightBoundArm
from scripts.probe_rgbd_handover import saved_map
from classical_fusion.measured_contact import MeasuredContactPermit
from classical_fusion.local_tool_guard import LocalToolEnvironmentGuard
from classical_fusion.robot_query_snapshot import RobotQuerySnapshot
from classical_fusion.static_interval_snapshot import StaticIntervalSnapshot


def probe(run):
    root=Path(run);r=json.loads((root/'result.json').read_text())
    cfg=json.loads((root/'run_configuration.json').read_text())
    changes=[e for e in r['transfer_events'] if e['event']=='local_evidence_retired_to_UNKNOWN']
    v=np.asarray(changes[-1]['physical_evidence_bounds']['evidence_vertices'])
    eq=ConvexHull(v).equations
    decisions=json.loads((root/'decisions.json').read_text())
    def find(o):
        if isinstance(o,dict):
            if o.get('link')=='magnet' and o.get('voxel_centers_m'):return o
            for value in reversed(list(o.values())):
                found=find(value)
                if found:return found
        if isinstance(o,list):
            for value in reversed(o):
                found=find(value)
                if found:return found
    rejection=find(decisions)
    if rejection is None:raise ValueError('No recorded magnet voxel rejection')
    centers=np.asarray(rejection['voxel_centers_m']);spacing=np.asarray(rejection['spacing_m'])
    right=TightBoundArm(Usd.Stage.Open(cfg['scene']),'right');q=np.asarray(r['final_joints'])[1]
    body=[];distance_axes=[]
    partial=[]
    footprint=ConvexHull(v[:,:2]).equations
    for name,(p,n) in right.collision_hulls_world(q):
        if name!='magnet':continue
        hull=ConvexHull(p)
        edges=np.array([p[j]-p[i] for face in hull.simplices for i,j in combinations(face,2)])
        axes=np.concatenate([n,np.eye(3),np.cross(edges[:,None,:],np.eye(3)[None,:,:]).reshape(-1,3)])
        lengths=np.linalg.norm(axes,axis=1);axes=axes[lengths>1e-10]/lengths[lengths>1e-10,None]
        axes=np.unique(np.round(axes,12),axis=0)
        support=np.abs(axes)@(spacing*.5);pc=p@axes.T;cc=centers@axes.T
        gap=np.maximum(cc-support-pc.max(0),pc.min(0)-cc-support).max(1)
        body.append(gap)
        refined=[]
        for c in centers:
            lo=c-spacing*.5;hi=c+spacing*.5
            def fun(x):
                d=x-np.clip(x,lo,hi)
                return .5*np.dot(d,d),d
            e=hull.equations
            fit=minimize(fun,p.mean(0),jac=True,method='SLSQP',constraints=[dict(type='ineq',
                fun=lambda x:-(e[:,:3]@x+e[:,3]),jac=lambda x:-e[:,:3])],
                options=dict(ftol=1e-15,maxiter=100))
            axis=fit.x-np.clip(fit.x,lo,hi);length=np.linalg.norm(axis)
            if length<1e-12:refined.append(0.);continue
            axis/=length
            refined.append(float((p@axis).min()-c@axis-np.abs(axis)@(spacing*.5)))
        distance_axes.append(refined)
        violations=[]
        corners=np.array(list(product([-.5,.5],repeat=3)))
        for c in centers:
            cube=c+corners*spacing;bad=[]
            for face in footprint:
                d=cube[:,:2]@face[:2]+face[2]
                if d.max()<0:continue
                clipped=list(cube[d>=0])
                for i,j in combinations(range(8),2):
                    if np.count_nonzero(corners[i]!=corners[j])!=1 or d[i]*d[j]>=0:continue
                    clipped.append(cube[i]+(cube[j]-cube[i])*d[i]/(d[i]-d[j]))
                clipped=np.asarray(clipped)
                simple=np.concatenate([n,np.eye(3),np.c_[footprint[:,:2],np.zeros(len(footprint))]])
                simple/=np.linalg.norm(simple,axis=1)[:,None]
                proj=clipped@simple.T;robot=p@simple.T
                gap=np.maximum(proj.min(0)-robot.max(0),robot.min(0)-proj.max(0)).max()
                bad.append(float(gap))
            violations.append(min(bad) if bad else 1.)
        partial.append(violations)
    sep=(centers@eq[:,:3].T+eq[:,3]-np.abs(eq[:,:3])@(spacing*.5)).max(1)
    print(json.dumps(dict(scope='diagnostic only; no map changes or execution',rejection=rejection,
        whole_voxel_outside_released_envelope_m=sep.tolist(),
        minimum_over_tool_components_SAT_separation_m=np.min(body,axis=0).tolist(),
        minimum_verified_distance_axis_separation_m=np.min(distance_axes,axis=0).tolist()),indent=2))
    print('minimum separation to footprint-exterior voxel portions',np.min(partial,axis=0).tolist())
    attached=[e for e in r['transfer_events'] if e['event']=='measured_payload_attached'][-1]
    release=[e for e in r['transfer_events'] if e['event']=='measured_release_command'][-1]
    source=np.asarray(attached['tcp']);destination=np.asarray(release['tcp'])
    cap=np.asarray(json.loads((root/'contact_permits.json').read_text())[-1]['target'])
    cap=(cap-source[:3,3])@source[:3,:3]@destination[:3,:3].T+destination[:3,3]
    permit=MeasuredContactPermit(cap,footprint)
    stage=Usd.Stage.Open(cfg['scene']);left=TightBoundArm(stage,'left')
    broad=RobotQuerySnapshot(saved_map(root));fine=StaticIntervalSnapshot(saved_map(root/'local_map'),assume_static_epoch=True)
    guard=LocalToolEnvironmentGuard(left,right,broad,fine,permit,max_boxes=127,max_depth=8)
    joints=np.asarray(r['final_joints']).ravel()
    print('current stationary complete environment certificate',guard.segment(joints,joints))
    print('remaining rejection',guard.last_rejection)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    probe(parser.parse_args().run)
