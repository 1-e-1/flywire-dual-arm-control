"""Single-state streaming backend microbenchmark, not whole-stack timing."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pxr import Usd
from astronaut_delivery.simultaneous import TightBoundArm
from astronaut_delivery.reaction import CombinedReaction


def benchmark(scene,output):
    stage=Usd.Stage.Open(scene);model=CombinedReaction(stage,*[TightBoundArm(stage,s) for s in ('left','right')])
    rng=np.random.default_rng(7291);q=rng.uniform(-1,1,(128,16));v=rng.uniform(-.1,.1,(128,16));a=rng.uniform(-.2,.2,(128,16))
    functions=dict(finite_difference_momentum=model.wrench,analytic_tree=lambda q,v,a:model.wrench_batch(q[None],v[None],a[None])[0])
    results={};values={}
    for name,fn in functions.items():
        for i in range(8):fn(q[i],v[i],a[i])
        times=[];answer=[]
        for x,y,z in zip(q,v,a):
            began=time.perf_counter();answer.append(fn(x,y,z));times.append(time.perf_counter()-began)
        values[name]=np.array(answer);results[name]=dict(median_ms=float(np.median(times)*1000),p99_ms=float(np.quantile(times,.99)*1000))
    reference=np.array([model.reference_wrench(x,y,z) for x,y,z in zip(q,v,a)])
    for name in results:results[name]['maximum_reference_error']=float(np.max(abs(values[name]-reference)))
    out=dict(scope='128 seeded independent states, one BLAS thread; microbenchmark only, not online rollout or hard realtime',backends=results)
    Path(output).write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(1):benchmark(a.scene,a.output)
