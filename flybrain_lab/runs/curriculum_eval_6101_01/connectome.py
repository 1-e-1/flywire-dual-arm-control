"""Full supplied FlyWire topology, signed rate reservoir; NOT biological emulation."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse

LAB=Path(__file__).resolve().parent
INPUTS=47

def build():
    source=LAB/'data/upstream';out=LAB/'data/graph';out.mkdir(exist_ok=False)
    neurons=list(csv.DictReader((source/'Completeness_783.csv').open()))
    ids=np.array([int(row['']) for row in neurons],dtype=np.int64);n=len(ids)
    columns=['Presynaptic_Index','Postsynaptic_Index','Presynaptic_ID','Postsynaptic_ID','Excitatory x Connectivity']
    table=pq.read_table(source/'Connectivity_783.parquet',columns=columns)
    pre,post,pid,qid,weight=[table[c].to_numpy() for c in columns]
    if min(pre.min(),post.min())<0 or max(pre.max(),post.max())>=n:raise ValueError('invalid neuron indices')
    if not np.array_equal(ids[pre],pid) or not np.array_equal(ids[post],qid):raise ValueError('ID/index mismatch')
    # Stored as postsynaptic rows: h_next[j] receives W[j,i]*h[i].
    matrix=sparse.coo_matrix((weight.astype(np.float32),(post,pre)),shape=(n,n)).tocsr()
    matrix.sum_duplicates();counts=np.asarray(abs(matrix).sum(axis=1)).ravel()
    matrix=sparse.diags(.85/np.maximum(counts,1.))@matrix
    matrix=matrix.astype(np.float32).tocsr();sparse.save_npz(out/'signed_normalized.npz',matrix)
    # Engineered robot interface, not a claim these IDs biologically encode joints.
    # Disjoint input/output pools; choose deterministically, no synthetic recurrent edge.
    rng=np.random.default_rng(601);order=rng.permutation(n)
    input_ids=order[:2048];output_ids=order[2048:4096]
    channels=np.arange(2048)%INPUTS;sign=rng.choice([-1.,1.],2048).astype(np.float32)
    np.savez(out/'interface.npz',neuron_ids=ids,input_ids=input_ids,output_ids=output_ids,channels=channels,sign=sign)
    metadata=dict(neurons=n,source_rows=len(pre),aggregated_directed_edges=matrix.nnz,
        source_sha256=hashlib.sha256((source/'Connectivity_783.parquet').read_bytes()).hexdigest(),
        dynamics='Engineered signed tanh rate network; row absolute norm <= .85; leak=.5; 12 internal steps',
        interface='47 robot features injected into 2048 disjoint deterministic ports; linear 2048-to-16 motor readout',
        scope='Entire author-supplied v783 neuron list, not a claim to contain every neuron in current FlyWire release')
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(json.dumps(metadata),flush=True)

class FlyCircuit:
    def __init__(self):
        folder=LAB/'data/graph';self.W=sparse.load_npz(folder/'signed_normalized.npz')
        ports=np.load(folder/'interface.npz');self.inputs=ports['input_ids'];self.outputs=ports['output_ids']
        self.channels=ports['channels'];self.sign=ports['sign']

    def features(self,x):
        x=np.asarray(x,np.float32)
        if x.shape!=(INPUTS,) or not np.isfinite(x).all():raise ValueError('invalid robot measurement')
        drive=np.zeros(self.W.shape[0],np.float32);drive[self.inputs]=np.clip(x[self.channels],-3,3)*self.sign
        h=np.zeros_like(drive)
        for _ in range(12):h=.5*h+.5*np.tanh(self.W@h+drive)
        return h[self.outputs]

def observation(q,left,right,goal,visual_target,phase,held):
    q=np.asarray(q).reshape(2,8);tcp=right.tcp(q[1]);camera=left.camera(q[0])
    relative=camera[:3,:3].T@(visual_target-camera[:3,3])
    return np.r_[np.sin(q.ravel()),np.cos(q.ravel()),(goal-tcp[:3,3])/.1,relative,
        tcp[:3,2],[(phase or 'approach')==v for v in ('approach','lift','transport','place','retreat')],float(held)].astype(np.float32)

class FlyController:
    def __init__(self,path):
        self.circuit=FlyCircuit();data=np.load(path)
        self.mean=data['mean'];self.scale=data['scale'];self.coef=data['coef']
    def action(self,x):
        h=(self.circuit.features(x)-self.mean)/self.scale
        return np.clip(np.r_[h,1.]@self.coef,-np.r_[np.full(8,.12),np.full(8,.3)],np.r_[np.full(8,.12),np.full(8,.3)])

if __name__=='__main__':build()
