"""Decompose ACTUAL logged arm reactions; not an executable baseline policy."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Usd
from astronaut_delivery.simultaneous import TightBoundArm
from astronaut_delivery.reaction import CombinedReaction,reaction_metrics


def analyze(run,scene):
    folder=Path(run);steps=json.loads((folder/'steps.json').read_text())
    states=[s['wrench_state'] for s in steps]
    q=np.array([s['q'] for s in states]);v=np.array([s['v'] for s in states]);a=np.array([s['a'] for s in states])
    dt=np.array([s['dt_s'] for s in steps]);logged=np.array([s['wrench'] for s in steps])
    stage=Usd.Stage.Open(str(scene));arms=[TightBoundArm(stage,x) for x in ('left','right')];model=CombinedReaction(stage,*arms)
    pieces=[]
    for side in (0,1):
        vi=np.zeros_like(v);ai=np.zeros_like(a);vi[:,8*side:8*(side+1)]=v[:,8*side:8*(side+1)];ai[:,8*side:8*(side+1)]=a[:,8*side:8*(side+1)]
        pieces.append(model.wrench_batch(q,vi,ai))
    left,right=pieces
    result=dict(scope='Additive contributions on the actual logged dual-arm trajectory. Right contribution alone is NOT a separately feasible controller baseline.',
        samples=len(steps),left=reaction_metrics(left,dt),right=reaction_metrics(right,dt),combined=reaction_metrics(logged,dt),
        decomposition_max_component_error=float(abs(left+right-logged).max()),components={})
    for name,sl in [('force',slice(0,3)),('torque',slice(3,6))]:
        l,r=left[:,sl],right[:,sl];nl=np.linalg.norm(l,axis=1);nr=np.linalg.norm(r,axis=1)
        active=(nl>1e-5)&(nr>1e-5);dot=np.sum(l*r,axis=1)
        result['components'][name]=dict(opposed_direction_time_fraction=float(np.sum(dt[active]*(dot[active]<0))/np.sum(dt[active])) if active.any() else None,
            combined_lower_than_right_contribution_time_fraction=float(np.sum(dt*(np.linalg.norm(l+r,axis=1)<nr))/dt.sum()),
            mean_cross_power=float(np.sum(dt*dot)/dt.sum()))
    np.savez_compressed(folder/'reaction_sharing.npz',time_s=[s['time_s'] for s in steps],left=left,right=right,combined=logged,dt_s=dt)
    (folder/'reaction_sharing.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--scene',required=True);a=p.parse_args()
    with threadpool_limits(1):analyze(a.run,a.scene)
