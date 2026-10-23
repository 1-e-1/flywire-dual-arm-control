"""Replay RGB-D chronologically into fly memory; fit isolated visual motor ports."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
LAB=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'runtime'))
from pxr import Usd
from astronaut_delivery.motion import Arm
from connectome import observation
from visual_circuit import VisualCircuit,ingest_initial_scan

def main():
    output=LAB/'runs/visual_bootstrap_03';output.mkdir(parents=True,exist_ok=False)
    runs=['reports/rgbd_quality_handover_regression_6101_01','reports/rgbd_capacity_repair_82007_set_01',
          'reports/rgbd_quality_control_82015_off_01']
    H=[];Y=[];groups=[];sources=[];began=time.perf_counter();frames=0
    for group,relative in enumerate(runs):
        folder=LAB.parent/relative;config=json.loads((folder/'run_configuration.json').read_text())
        scene=Path(config['scene']);scene=scene if scene.is_absolute() else LAB.parent/scene
        stage=Usd.Stage.Open(str(scene));left,right=Arm(stage,'left'),Arm(stage,'right');circuit=VisualCircuit()
        scan=Path(config['initial_scan']);scan=scan if scan.is_absolute() else LAB.parent/scan
        frames+=ingest_initial_scan(circuit,scan)
        for p in sorted((folder/'commissioning').glob('rgbd_*.npz')):
            with np.load(p) as f:circuit.ingest(dict(f));frames+=1
        obs=json.loads((folder/'observations.json').read_text());paths=json.loads((folder/'paths.json').read_text())
        steps=json.loads((folder/'steps.json').read_text());step_times=np.array([s['time_s'] for s in steps])
        source=dict(run=relative,observation_manifest_sha256=hashlib.sha256((folder/'observations.json').read_bytes()).hexdigest(),
            path_manifest_sha256=hashlib.sha256((folder/'paths.json').read_bytes()).hexdigest(),frames=0)
        index=0
        def ingest(row):
            nonlocal frames
            p=folder/'raw'/row['file']
            if row.get('sha256') and hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('raw image hash mismatch')
            step_index=np.searchsorted(step_times,row.get('received_s',row['time_s']),side='right')-1
            wrench=None if step_index<0 else steps[step_index]['wrench']
            with np.load(p) as f:circuit.ingest(dict(f),wrench=wrench,received_wall=row.get('received_wall_s'))
            frames+=1;source['frames']+=1
        for row in paths:
            # Strictly earlier exposure; tie at startup uses first recorded frame.
            while index<len(obs) and (obs[index]['time_s']<row['time_s']-1e-9 or index==0):
                ingest(obs[index]);index+=1
            if not all(k in row for k in ('target','visual_target_position','start','end')):continue
            phase=row.get('payload_contact_phase',row.get('phase','approach'))
            if phase=='visual_acquisition_right_rest':phase='approach'
            if phase not in ('approach','lift','transport','place','retreat'):continue
            q=np.array(row['start']);goal=np.array(row['target'])+np.array([0,0,row['standoff_m']])
            if row.get('phase')=='visual_acquisition_right_rest':goal=right.tcp(q[8:])[:3,3]
            x=observation(q,left,right,goal,np.array(row['visual_target_position']),phase,'payload_local_points' in row)
            H.append(circuit.features(x));Y.append(np.array(row['end'])-q);groups.append(group)
            if len(H)%25==0:print(json.dumps(dict(samples=len(H),frames=frames,elapsed_s=time.perf_counter()-began)),flush=True)
        while index<len(obs):ingest(obs[index]);index+=1
        sources.append(source)
    H=np.array(H);Y=np.array(Y);groups=np.array(groups);train=groups!=2;valid=~train
    mean=H[train].mean(0);scale=np.maximum(H[train].std(0),1e-4);X=np.c_[(H-mean)/scale,np.ones(len(H))].astype(float)
    Z=X[train];candidates=[];models=[]
    for ridge in (10.,100.,1000.,10000.):
        coef=Z.T@solve(Z@Z.T+ridge*np.eye(len(Z)),Y[train],assume_a='pos')
        rmse=float(np.sqrt(np.mean((X[valid]@coef-Y[valid])**2)));models.append(coef)
        candidates.append(dict(ridge=ridge,validation_joint_rmse_rad=rmse))
    best=int(np.argmin([v['validation_joint_rmse_rad'] for v in candidates]))
    np.savez(output/'controller.npz',visual_schema=3,mean=mean,scale=scale,coef=models[best])
    np.savez(output/'dataset.npz',features=H,labels=Y,groups=groups)
    report=dict(train_samples=int(train.sum()),validation_samples=int(valid.sum()),processed_frames=frames,sources=sources,
        candidates=candidates,selected=candidates[best],zero_action_validation_rmse_rad=float(np.sqrt(np.mean(Y[valid]**2))),
        scope='Validation used for ridge selection, not heldout; imitation bootstrap only; reaction reduction unproven',
        elapsed_s=time.perf_counter()-began)
    (output/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
