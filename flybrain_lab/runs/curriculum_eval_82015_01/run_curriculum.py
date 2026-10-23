"""Declared isolated training -> two frozen closed-loop probes -> audits."""
import json,subprocess,sys,traceback
from pathlib import Path
LAB=Path(__file__).resolve().parent
def main():
    state=dict(status='running',phase='training',episodes=[],scope='Development curriculum, no holdout claim')
    def save():(LAB/'progress_curriculum_01.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    def execute(label,args,cwd=LAB):
        with (LAB/(label+'.log')).open('x',encoding='utf-8') as log:
            return subprocess.run([sys.executable,'-B',*args],cwd=cwd,stdout=log,stderr=subprocess.STDOUT).returncode
    save()
    try:
        if execute('training_curriculum_01',[str(LAB/'train_curriculum.py')]):raise RuntimeError('training process failed')
        state['training']=json.loads((LAB/'runs/curriculum_01/training.json').read_text());save()
        for seed in (6101,82015):
            state.update(phase='closed_loop',active_seed=seed);save()
            destination=LAB/f'runs/curriculum_eval_{seed}_01'
            code=execute(f'curriculum_eval_{seed}_01',[str(LAB/'curriculum_mission.py'),'--seed',str(seed),
                '--output',str(destination),'--checkpoint',str(LAB/'runs/curriculum_01/controller.npz')])
            if code:raise RuntimeError('evaluation process failed')
            result=json.loads((destination/'result.json').read_text())
            row=dict(seed=seed,status=result['status'],error=result.get('error'),completed=result.get('completed_observed_targets',[]))
            base=LAB.parent/('reports/rgbd_paired_mission_pilot_01' if seed==6101 else 'reports/rgbd_holdout_01')
            execute(f'curriculum_audit_{seed}_01',['-m','astronaut_delivery.rgbd_mission_audit','--run',str(destination),
                '--scene',str(base/f'scene_{seed}/rgbd_random_{seed}.usda')],LAB/'runtime')
            execute(f'curriculum_clock_{seed}_01',[str(LAB/'runtime/scripts/audit_rgbd_wall_clock.py'),'--run',str(destination)],LAB/'runtime')
            row['mission_audit']=json.loads((destination/'mission_audit.json').read_text())
            row['visual_input']=json.loads((destination/'visual_input_audit.json').read_text())
            row['clock_audit']=json.loads((destination/'wall_clock_audit.json').read_text())
            state['episodes'].append(row);save()
            # No-motion failure is not a safety pass, but can be followed by next probe.
            if row['mission_audit'].get('paths') and not row['mission_audit']['all_continuous_geometry_passed']:
                raise RuntimeError('executed path failed audit; batch stopped')
            if row['mission_audit'].get('paths') and (not row['mission_audit']['wrench_crosscheck_passed'] or not row['clock_audit']['passed']):
                raise RuntimeError('executed trajectory wrench/clock audit failed')
            if not row['clock_audit']['source_unchanged_during_execution']:raise RuntimeError('runtime changed')
        state.update(status='complete',phase='finished');save()
    except Exception:state.update(status='stopped',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
