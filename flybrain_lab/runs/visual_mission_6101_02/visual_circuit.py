"""RGB-D stream -> full fly topology -> linear joint readout. No point-cloud input."""
import time,json,hashlib
from pathlib import Path
import cv2
import numpy as np
from connectome import FlyCircuit,INPUTS

RETINA_SHAPE=(96,54)
RETINA_SIZE=96*54*8
AUX_SIZE=128

def pack_frame(frame,previous=None,wrench=None,received_wall=None):
    rgb=np.asarray(frame['rgb']);depth=np.asarray(frame['depth'],np.float32)
    if rgb.shape!=depth.shape+(3,):raise ValueError('RGB/depth must be registered')
    valid=np.isfinite(depth)&(depth>0)
    sigma=np.asarray(frame.get('depth_sigma',np.zeros_like(depth)),np.float32)
    color=np.asarray(frame.get('color_valid',np.ones_like(depth)),np.float32)
    robot=np.asarray(frame.get('robot_mask',frame.get('self_mask',np.zeros_like(depth))),np.float32)
    # All delivered pixels contribute via area pooling, but pooling IS lossy.
    channels=np.concatenate([np.nan_to_num(rgb.astype(np.float32))/255.,
        np.where(valid,depth/1.2,0)[...,None],valid[...,None].astype(np.float32),
        (np.nan_to_num(sigma)/.005)[...,None],robot[...,None],color[...,None]],axis=-1)
    retina=np.stack([cv2.resize(channels[...,i],RETINA_SHAPE,interpolation=cv2.INTER_AREA)
                     for i in range(channels.shape[-1])],axis=-1).ravel()
    q=np.asarray(frame.get('q',np.zeros((2,8)))).ravel();stamp=float(frame.get('stamp',frame.get('exposure_s',0.)))
    if 'q' not in frame or (previous is not None and not previous.get('q_available',True)):previous=None
    dt=0. if previous is None else stamp-previous['stamp'];delta=np.zeros(16) if previous is None else q-previous['q']
    v=delta/dt if dt>1e-6 else np.zeros(16)
    acceleration=(v-previous['v'])/dt if previous is not None and dt>1e-6 else np.zeros(16)
    K=np.array(frame['K'],float).copy();K[0]/=depth.shape[1];K[1]/=depth.shape[0]
    wall=float(frame.get('requested_wall_s',0.));received=wall if received_wall is None else float(received_wall)
    age=max(0.,received-wall) if np.isfinite(wall) else 0.
    values=np.r_[np.sin(q),np.cos(q),v,acceleration,delta,
        np.zeros(6) if wrench is None else np.asarray(wrench)/np.array([.35]*3+[.15]*3),
        K.ravel(),np.asarray(frame['camera_world']).ravel(),dt,age,float(dt>1e-6),float(np.isfinite(wall)),
        depth.shape[0]/480.,depth.shape[1]/848.,float('depth_sigma' in frame),float('color_valid' in frame),
        float('robot_mask' in frame or 'self_mask' in frame),float(wrench is not None),float('q' in frame)]
    aux=np.zeros(AUX_SIZE,np.float32);aux[:len(values)]=np.nan_to_num(values)
    return np.clip(np.r_[retina,aux],-5,5).astype(np.float32),dict(q=q.copy(),v=v,stamp=stamp,q_available='q' in frame)

def ingest_initial_scan(circuit,manifest):
    path=Path(manifest).resolve();records=json.loads(path.read_text())['records']
    for record in records:
        source=(path.parent/record['file']).resolve()
        if not source.is_relative_to(path.parent):raise ValueError('initial record escaped root')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('initial RGB-D hash mismatch')
        with np.load(source) as f:circuit.ingest(dict(f))
    circuit.previous=None # External scanner epoch differs from wrist camera clock.
    return len(records)

class VisualCircuit(FlyCircuit):
    def __init__(self):
        super().__init__();rng=np.random.default_rng(8601);ports=rng.permutation(self.W.shape[0])
        self.frame_ports=ports[:RETINA_SIZE+AUX_SIZE]
        self.task_ports=ports[RETINA_SIZE+AUX_SIZE:RETINA_SIZE+AUX_SIZE+INPUTS]
        self.read_ports=ports[-2048:]
        self.state=np.zeros(self.W.shape[0],np.float32);self.drive=np.zeros_like(self.state)
        self.previous=None;self.frames=0

    def ingest(self,frame,**kwargs):
        data,self.previous=pack_frame(frame,self.previous,**kwargs)
        self.drive[:]=0;self.drive[self.frame_ports]=.25*data
        for _ in range(4):self.state=.5*self.state+.5*np.tanh(self.W@self.state+self.drive)
        self.frames+=1

    def features(self,task):
        task=np.asarray(task,np.float32)
        if task.shape!=(INPUTS,) or not np.isfinite(task).all():raise ValueError('invalid task observation')
        if self.frames==0:raise ValueError('no actual camera frame; no geometry-only fallback')
        h=self.state.copy();drive=self.drive.copy();drive[self.task_ports]=np.clip(task,-5,5)
        for _ in range(4):h=.5*h+.5*np.tanh(self.W@h+drive)
        return h[self.read_ports]

class VisualController:
    def __init__(self,path):
        self.circuit=VisualCircuit();data=np.load(path)
        if int(data['visual_schema'])!=3:raise ValueError('v3 complete-observation checkpoint required')
        self.mean=data['mean'];self.scale=data['scale'];self.coef=data['coef']
    def action(self,x):
        h=(self.circuit.features(x)-self.mean)/self.scale
        bounds=np.r_[np.full(8,.12),np.full(8,.3)]
        return np.clip(np.r_[h,1.]@self.coef,-bounds,bounds)
