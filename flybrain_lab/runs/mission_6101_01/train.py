"""Isolated teacher bootstrap: fit only linear robot readout of real fly circuit."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve
LAB=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'runtime'))
from pxr import Usd
from astronaut_delivery.motion import Arm
from connectome import FlyCircuit,observation

def main():
    output=LAB/'runs/bootstrap_01';output.mkdir(parents=True,exist_ok=False)
    parent=LAB.parent
    # Read only previously recorded encoder/measurement/action data, no network weights.
    runs=['reports/rgbd_quality_handover_regression_6101_01',
          'reports/rgbd_capacity_repair_82007_set_01',
          'reports/rgbd_quality_control_82015_off_01']
    circuit=FlyCircuit();features=[];labels=[];groups=[];records=[];start=time.perf_counter()
    for group,relative in enumerate(runs):
        folder=parent/relative;config=json.loads((folder/'run_configuration.json').read_text())
        scene=Path(config['scene']);scene=scene if scene.is_absolute() else parent/scene
        stage=Usd.Stage.Open(str(scene));left,right=Arm(stage,'left'),Arm(stage,'right')
        source=folder/'paths.json';paths=json.loads(source.read_text())
        records.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        for row in paths:
            if not all(k in row for k in ('target','visual_target_position','start','end')):continue
            q=np.array(row['start']);goal=np.array(row['target'])+np.array([0,0,row['standoff_m']])
            visual=np.array(row['visual_target_position']);phase=row.get('payload_contact_phase',row.get('phase','approach'))
            if phase not in ('approach','lift','transport','place','retreat'):continue
            x=observation(q,left,right,goal,visual,phase,'payload_local_points' in row)
            features.append(circuit.features(x));labels.append(np.array(row['end'])-q);groups.append(group)
            if len(features)%25==0:print(json.dumps(dict(samples=len(features),elapsed_s=time.perf_counter()-start)),flush=True)
    H=np.array(features);Y=np.array(labels);g=np.array(groups)
    # Validation is an entire distinct scene, not adjacent frames from training.
    train=g!=2;valid=~train;mean=H[train].mean(0);scale=np.maximum(H[train].std(0),1e-4)
    X=np.c_[(H-mean)/scale,np.ones(len(H))].astype(np.float64)
    # Dual ridge avoids a costly 2049x2049 system with a small sample set.
    def fit(mask):
        Z=X[mask];return Z.T@solve(Z@Z.T+10.*np.eye(mask.sum()),Y[mask],assume_a='pos')
    coef=fit(train);error=float(np.sqrt(np.mean((X[valid]@coef-Y[valid])**2)))
    np.savez(output/'controller.npz',mean=mean,scale=scale,coef=coef)
    np.savez(output/'dataset.npz',features=H,labels=Y,groups=g)
    report=dict(train_samples=int(train.sum()),validation_samples=int(valid.sum()),validation_joint_rmse_rad=error,
        validation_zero_action_rmse_rad=float(np.sqrt(np.mean(Y[valid]**2))),sources=records,
        scope='Offline teacher imitation only, not task success or reaction reduction; train 6101/82007, validate 82015',
        neurons=circuit.W.shape[0],edges=circuit.W.nnz,elapsed_s=time.perf_counter()-start)
    (output/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
