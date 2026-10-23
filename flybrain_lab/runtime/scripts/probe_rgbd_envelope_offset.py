"""Same retained measured core, same radius; compare outer offset geometry.

Actual object geometry is used ONLY by this post-execution audit, not by the
offset constructor or planner. No pose is proposed or executed here.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from classical_fusion.euclidean_envelope import outer_euclidean_offset
from astronaut_delivery.source_freeze import fingerprint,unchanged


def probe(run,output):
    root=Path(run);out=Path(output)
    if out.exists():raise FileExistsError(out)
    read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    result=read('result.json');cfg=read('run_configuration.json');hashes=fingerprint(ROOT)
    from pxr import Usd
    from astronaut_delivery.simultaneous import TightBoundArm
    from astronaut_delivery.separation_recovery_delivery import BoundedSeparationGuard
    stage=Usd.Stage.Open(cfg['scene']);left,right=[TightBoundArm(stage,s) for s in ('left','right')]
    truth=BoundedSeparationGuard(stage,left,right,.003);rows=[]
    for a in [e for e in result['transfer_events'] if e['event']=='measured_payload_attached']:
        T=np.asarray(a['tcp']);sources=[root/a['retained_measured_surface'],root/a['envelope_file']]
        with np.load(sources[0]) as z:p=z['local_vertices']@T[:3,:3].T+T[:3,3]
        with np.load(sources[1]) as z:old=z['vertices'];radius=float(z['margin_m']);support=float(z['support_z'])
        floor=p.copy();floor[:,2]=support;core=np.r_[p,floor]
        vertices,h=outer_euclidean_offset(core,radius)
        n=h.equations[:,:3];s=(core@n.T).max(0)+radius*np.linalg.norm(n,axis=1)+h.equations[:,3]
        signed=truth.parts[a['name']].points@n.T+h.equations[:,3]
        old_volume=ConvexHull(old).volume
        rows.append(dict(name=a['name'],same_euclidean_radius_m=radius,old_cube_offset_volume_m3=old_volume,
            new_outer_offset_volume_m3=h.volume,volume_ratio=h.volume/old_volume,
            old_payload_vertices=len(a['local_points']),new_payload_vertices=len(vertices),
            maximum_error_ball_support_violation_m=float(s.max()),actual_simulated_payload_contained=bool((signed<=1e-8).all()),
            maximum_actual_geometry_violation_m=float(signed.max()),
            measured_input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
    report=dict(scope='SAME SAVED MEASUREMENTS/RADIUS, POST-EXECUTION GEOMETRY AUDIT ONLY; not mission progress or newly free evidence',
        source_unchanged=unchanged(ROOT,hashes),source_hashes=hashes,run=str(run),objects=rows)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(dict(scope=report['scope'],objects=rows),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(1):probe(a.run,a.output)
