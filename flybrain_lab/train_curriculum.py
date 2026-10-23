"""Measured-frame visual acquisition curriculum; teacher is offline ONLY."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
LAB=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'runtime'))
from pxr import Usd
from astronaut_delivery.motion import Arm
from astronaut_delivery.visual_sufficiency import VisualSetAllocator,SufficiencyConfig
from curriculum_circuit import CurriculumCircuit,task_observation
from visual_circuit import ingest_initial_scan

RUNS=['reports/rgbd_quality_handover_regression_6101_01','reports/rgbd_capacity_repair_82007_set_01',
      'reports/rgbd_quality_control_82015_off_01']

def main():
    output=LAB/'runs/curriculum_01';output.mkdir(parents=True,exist_ok=False)
    H=[];Y=[];groups=[];kinds=[];margins=[];jacobians=[];witnesses=[];models=[];sources=[]
    began=time.perf_counter();frames=0
    def progress(**extra):
        row=dict(status='collecting',samples=len(H),processed_frames=frames,elapsed_s=time.perf_counter()-began,**extra)
        (output/'progress.json').write_text(json.dumps(row,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
    for group,relative in enumerate(RUNS):
        folder=LAB.parent/relative;config=json.loads((folder/'run_configuration.json').read_text())
        scene=Path(config['scene']);scene=scene if scene.is_absolute() else LAB.parent/scene
        stage=Usd.Stage.Open(str(scene));left,right=Arm(stage,'left'),Arm(stage,'right')
        allocator=VisualSetAllocator(left,None,SufficiencyConfig());models.append((left,right,allocator))
        circuit=CurriculumCircuit();scan=Path(config['initial_scan']);scan=scan if scan.is_absolute() else LAB.parent/scan
        frames+=ingest_initial_scan(circuit,scan)
        for p in sorted((folder/'commissioning').glob('rgbd_*.npz')):
            with np.load(p) as f:circuit.ingest(dict(f));frames+=1
        obs=json.loads((folder/'observations.json').read_text());paths=json.loads((folder/'paths.json').read_text())
        steps=json.loads((folder/'steps.json').read_text());times=np.array([s['time_s'] for s in steps])
        sources.append(dict(run=relative,observations_sha256=hashlib.sha256((folder/'observations.json').read_bytes()).hexdigest()))
        index=0;latest=None;counts=dict(proposed=0,teacher_unavailable=0,invalid_view_examples=0,valid_view_examples=0)
        def ingest(row,index):
            nonlocal frames,latest
            path=folder/'raw'/row['file']
            if row.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('RGB-D hash mismatch')
            with np.load(path) as f:frame=dict(f)
            step=np.searchsorted(times,row.get('received_s',row['time_s']),side='right')-1
            circuit.ingest(frame,wrench=None if step<0 else steps[step]['wrench'],received_wall=row.get('received_wall_s'))
            frames+=1;latest=row
            if index%8!=0 and index>3:return
            q=np.asarray(frame['q']).reshape(2,8)
            for name,target in row['targets'].items():
                if 'covariance' not in target:continue
                cov=np.array(target['covariance']);pos=np.array(target['position']);pixels=target['pixels'];age=row['time_s']-target['stamp']
                try:
                    ctx=allocator.context(q[0],pos,cov,pixels,age,require_feasible=False)
                    base=allocator.margins(q[0],ctx)
                except ValueError:continue
                bad=bool(base.min()<0)
                if not bad and index%40!=0:continue
                counts['proposed']+=1
                end=allocator.restore_view(q[0],pos,cov,pixels,age) if bad else q[0].copy()
                if end is None:counts['teacher_unavailable']+=1;continue
                delta=np.r_[end-q[0],np.zeros(8)]
                x=task_observation(q,left,right,allocator,right.tcp(q[1])[:3,3],pos,cov,pixels,age,None,False,True)
                h=1e-5;G=np.column_stack([(allocator.margins(q[0]+np.eye(8)[j]*h,ctx)-allocator.margins(q[0]-np.eye(8)[j]*h,ctx))/(2*h) for j in range(8)])
                H.append(circuit.features(x));Y.append(delta);groups.append(group);kinds.append(1)
                margins.append(base);jacobians.append(G)
                witnesses.append(dict(group=group,q=q.tolist(),position=pos.tolist(),covariance=cov.tolist(),pixels=pixels,age=age,
                    initially_invalid=bad,raw_file=row['file'],scope='Kinematic view teacher only; no whole-path environment safety claim'))
                counts['invalid_view_examples' if bad else 'valid_view_examples']+=1
        for path in paths:
            while index<len(obs) and (obs[index]['time_s']<path['time_s']-1e-9 or index==0):
                ingest(obs[index],index);index+=1
                if index%300==0:progress(group=group,view_counts=counts)
            if not all(k in path for k in ('target','visual_target_position','start','end','visual_target_name')):continue
            phase=path.get('payload_contact_phase',path.get('phase','approach'))
            if phase not in ('approach','lift','transport','place','retreat'):continue
            t=latest['targets'].get(path['visual_target_name']) if latest else None
            if t is None or 'covariance' not in t:continue
            q=np.array(path['start']);goal=np.array(path['target'])+np.array([0,0,path['standoff_m']])
            try:x=task_observation(q,left,right,allocator,goal,np.array(path['visual_target_position']),np.array(t['covariance']),
                t['pixels'],latest['time_s']-t['stamp'],phase,'payload_local_points' in path,False)
            except ValueError:continue
            H.append(circuit.features(x));Y.append(np.array(path['end'])-q);groups.append(group);kinds.append(0)
            margins.append(np.zeros(8));jacobians.append(np.zeros((8,8)));witnesses.append(None)
        while index<len(obs):ingest(obs[index],index);index+=1
        progress(group=group,view_counts=counts)
    H=np.array(H);Y=np.array(Y);g=np.array(groups);kind=np.array(kinds);M=np.array(margins);J=np.array(jacobians)
    train=g!=2;valid=~train;acq=train&(kind==1);travel=train&(kind==0);val_acq=valid&(kind==1)
    if min(acq.sum(),travel.sum(),val_acq.sum())<2:raise RuntimeError('insufficient separate-scene curriculum examples')
    mean=H[train].mean(0);scale=np.maximum(H[train].std(0),1e-4);X=np.c_[(H-mean)/scale,np.ones(len(H))].astype(np.float32)
    np.savez(output/'dataset.npz',features=H,labels=Y,groups=g,kinds=kind,margins=M,jacobians=J)
    (output/'witnesses.json').write_text(json.dumps(witnesses),encoding='utf-8')
    def ridge(mask,alpha):
        Z=X[mask].astype(float);return (Z.T@solve(Z@Z.T+alpha*np.eye(len(Z)),Y[mask],assume_a='pos')).astype(np.float32)
    travel_coef=ridge(travel,100.)
    torch.set_num_threads(1);torch.manual_seed(9401)
    coef=torch.nn.Parameter(torch.tensor(ridge(acq,10.)[:,:8]));opt=torch.optim.Adam([coef],lr=2e-4)
    XT=torch.tensor(X[acq]);YT=torch.tensor(Y[acq,:8],dtype=torch.float32)
    MT=torch.tensor(M[acq],dtype=torch.float32);JT=torch.tensor(J[acq],dtype=torch.float32)
    # Negative examples get a larger share, preventing the safe-but-useless zero predictor.
    weight=torch.where(MT.min(1).values<0,torch.tensor(4.),torch.tensor(1.))
    candidates=[];best_score=None
    for epoch in range(1001):
        if epoch:
            ids=torch.randint(len(XT),(min(128,len(XT)),));pred=XT[ids]@coef
            imitation=((pred-YT[ids])/.12).square().mean(1)
            required=torch.minimum(MT[ids],torch.full_like(MT[ids],.003))
            required=torch.where(MT[ids]<0,torch.full_like(required,.003),required)
            predicted_margin=MT[ids]+torch.einsum('bij,bj->bi',JT[ids],pred)
            quality=torch.relu(required-predicted_margin).square().mean(1)*200.
            bound=torch.relu(pred.abs()-.12).square().mean(1)*100.
            loss=((imitation+quality+bound)*weight[ids]).mean()+1e-5*coef.square().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([coef],1.);opt.step()
        if epoch%200:continue
        C=np.zeros((X.shape[1],16),np.float32);C[:,:8]=coef.detach().numpy()
        output_delta=np.clip(X@C,-.12,.12);eligible=0;restored=0;valid_preserved=0;already_valid=0
        for i in np.flatnonzero(val_acq):
            w=witnesses[i];q=np.array(w['q']);candidate=q[0]+output_delta[i,:8]
            okay=bool(np.all(abs(candidate)<=np.pi))
            try:models[w['group']][2].context(candidate,w['position'],w['covariance'],w['pixels'],w['age'])
            except ValueError:okay=False
            if w['initially_invalid']:eligible+=1;restored+=int(okay)
            else:already_valid+=1;valid_preserved+=int(okay)
        rmse=float(np.sqrt(np.mean((output_delta[val_acq]-Y[val_acq])**2)))
        record=dict(epoch=epoch,invalid_validation_views=eligible,restored=restored,already_valid=already_valid,
            valid_preserved=valid_preserved,joint_rmse_rad=rmse);candidates.append(record)
        score=(restored/max(eligible,1),valid_preserved/max(already_valid,1),-rmse)
        if best_score is None or score>best_score:
            best_score=score
            np.savez(output/'controller.npz',visual_schema=4,mean=mean,scale=scale,travel=travel_coef,acquire=C)
            best=record.copy()
        print(json.dumps(dict(status='fitting',**record)),flush=True)
        (output/'progress.json').write_text(json.dumps(dict(status='fitting',samples=len(H),processed_frames=frames,
            selected=best,latest=record),indent=2),encoding='utf-8')
    report=dict(processed_frames=frames,train_acquisition_samples=int(acq.sum()),train_travel_samples=int(travel.sum()),
        validation_acquisition_samples=int(val_acq.sum()),validation_travel_samples=int((valid&(kind==0)).sum()),
        updates=1000,selected=best,candidates=candidates,sources=sources,elapsed_s=time.perf_counter()-began,
        scope='Real measured-frame curriculum; view endpoint proxy validation, NOT collision certificate or full-task success; no online teacher fallback')
    (output/'training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (output/'progress.json').write_text(json.dumps(dict(status='complete',**report),indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
if __name__=='__main__':
    with threadpool_limits(1):main()
