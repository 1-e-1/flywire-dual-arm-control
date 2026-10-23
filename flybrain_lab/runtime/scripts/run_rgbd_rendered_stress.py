"""Frozen serial nominal + new-render sensor faults on one development scene."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rgbd_paired_evaluation import command,read,reused_inputs
from astronaut_delivery.source_freeze import freeze,unchanged


def run(output,reuse,seed):
    folder=Path(output).resolve();folder.mkdir(parents=True,exist_ok=False)
    scene,recording,manifest=reused_inputs(reuse,seed,True)
    if not manifest:raise ValueError('authenticated development inputs required')
    spec=dict(seed=seed,cases=['nominal','blackout','rgb_blackout','delay'],mode='set',
        scope='New online renders with reused development commissioning; NOT holdout',trigger_motion_s=5.,duration_wall_s=.6,delay_wall_s=.3)
    hashes=freeze(ROOT,folder,spec);rows=[]
    for case in spec['cases']:
        if not unchanged(ROOT,hashes):raise RuntimeError('source changed during frozen stress batch')
        dest=folder/f'run_{seed}_{case}'
        args=['--output',dest,'--scene',scene,'--initial-scan',recording,'--scene-token',manifest['scene_token'],
            '--calibration-id',manifest['calibration_id'],'--assume-unchanged-scene','--mode','set']
        if case!='nominal':args+=['--sensor-fault',case]
        execution=command('astronaut_delivery.rgbd_mission',args,folder/f'{case}.log')
        row=dict(case=case,run=str(dest),execution=execution,task_completed=False,evidence_audit_passed=False)
        if (dest/'paths.json').exists():
            row['mission_audit_execution']=command('astronaut_delivery.rgbd_mission_audit',['--run',dest,'--scene',scene],folder/f'{case}_audit.log')
            row['clock_audit_execution']=command('scripts.audit_rgbd_wall_clock',['--run',dest],folder/f'{case}_clock.log')
            if case!='nominal' and (dest/'sensor_faults/records.jsonl').exists():
                row['fault_audit_execution']=command('scripts.audit_rgbd_rendered_stress',['--run',dest],folder/f'{case}_fault.log')
                row['evidence_audit_passed']=read(dest/'rendered_stress_audit.json').get('evidence_and_stop_audit_passed',False)
            else:row['evidence_audit_passed']=read(dest/'wall_clock_audit.json').get('passed',False)
            row['task_completed']=read(dest/'mission_audit.json').get('scene_all_objects_delivered_audited',False)
        rows.append(row)
        state=dict(protocol=spec,source_unchanged=unchanged(ROOT,hashes),planned_trials=4,finished_trials=len(rows),rows=rows)
        (folder/'evaluation.json').write_text(json.dumps(state,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
        if not state['source_unchanged']:raise RuntimeError('mixed-version stress evidence excluded')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reuse-inputs-from',required=True);p.add_argument('--seed',type=int,default=6102)
    a=p.parse_args();run(a.output,a.reuse_inputs_from,a.seed)
