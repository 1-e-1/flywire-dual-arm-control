"""Offline actual-arm residual certificates; no changes to frozen controller."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from pxr import Usd
from .online_scene import ONLINE_SCENE
from .motion import Arm
from .reaction import CombinedReaction
from .visual_sufficiency import VisualSetAllocator
from .reaction_feasibility import bounded_residual


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--count',type=int,default=100)
    p.add_argument('--output',required=True);args=p.parse_args()
    stage=Usd.Stage.Open(str(ONLINE_SCENE));left,right=Arm(stage,'left'),Arm(stage,'right')
    model=CombinedReaction(stage,left,right);allocator=VisualSetAllocator(left,model)
    steps=json.loads((Path(args.run)/'control_steps.json').read_text());rows=[]
    # Deterministic uniform time-index samples, including low-excitation states.
    indices=np.unique(np.linspace(0,len(steps)-1,args.count).astype(int))
    scale=np.diag([1/.35]*3+[1/.15]*3)
    for i in indices:
        item=steps[i];q,v,a=(np.asarray(item[k],float).reshape(16) for k in ('q','v','a'))
        A,bias=model.coefficients(q,v);B=scale@A[:,:8];b=scale@(A[:,8:]@a[8:]+bias)
        _,Jv,Jw=allocator.tree.evaluate(q[:8]).point('b8',left.tool_local[:3,3]);J=np.vstack([Jv,Jw])
        row=dict(index=int(i),time_s=item['time_s'],actual_normalized_residual=float(np.linalg.norm(scale@(A@a+bias))))
        for name,E,f in [('box_superset',None,None),('preserve_logged_camera_acceleration',J,J@a[:8])]:
            row[name]=bounded_residual(B,b,np.full(8,-.5),np.full(8,.5),E=E,f=f)
        rows.append(row)
    out=dict(scope='Instantaneous left acceleration allocation with right acceleration fixed. Box-only feasibility does NOT certify vision, collision, or an integrable trajectory.',
             normalization=dict(force_N=.35,torque_Nm=.15),rows=rows,
             status_counts={m:dict(Counter(r[m]['status'] for r in rows)) for m in ('box_superset','preserve_logged_camera_acceleration')})
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['status_counts'],indent=2))


if __name__=='__main__':main()
