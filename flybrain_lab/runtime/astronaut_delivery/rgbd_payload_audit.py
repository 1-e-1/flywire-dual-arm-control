"""Measure object-envelope containment AFTER reconstruction; truth is audit-only."""
import argparse,json
from pathlib import Path
import numpy as np
from pxr import Usd
from classical_fusion.measured_object import propose_envelope
from .motion import CollisionGuard
from .simultaneous import TightBoundArm


def audit(run,scene):
    root=Path(run);surface=np.load(root/'measured_surface.npz')['vertices']
    observations=json.loads((root/'observations.json').read_text());targets=observations[-1]['targets']
    stage=Usd.Stage.Open(str(scene));truth=CollisionGuard(stage,*[TightBoundArm(stage,s) for s in ('left','right')])
    rows={}
    for name,measurement in targets.items():
        try:
            proposal=propose_envelope(surface,measurement['position'])
            points=truth.parts[name].points;inside=proposal.contains(points)
            np.savez_compressed(root/(name+'_measured_envelope.npz'),vertices=proposal.vertices,faces=proposal.faces,equations=proposal.equations)
            rows[name]=dict(observed_points=proposal.observed_points,margin_m=proposal.margin_m,support_z=proposal.support_z,
                truth_vertices=len(points),truth_vertices_contained=int(inside.sum()),all_truth_vertices_contained=bool(inside.all()),
                maximum_signed_support_violation_m=float(np.max(points@proposal.equations[:,:3].T+proposal.equations[:,3])),
                unseen_containment_proven=False)
        except Exception as exc:rows[name]=dict(error=str(exc),all_truth_vertices_contained=False)
    result=dict(scope='Offline containment audit of measured proposals; does not authorize a grasp or supply CAD geometry to planning',objects=rows)
    (root/'payload_envelope_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);a=p.parse_args();audit(a.run,a.scene)


if __name__=='__main__':main()
