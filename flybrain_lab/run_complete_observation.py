"""Private sequential visual training and closed-loop test with persistent status."""
import json,subprocess,sys,traceback
from pathlib import Path
LAB=Path(__file__).resolve().parent
def main():
    state=dict(status='running',phase='training',scope='Isolated full-observation fly circuit experiment')
    def save():(LAB/'progress_v3.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        stages=[('training',[str(LAB/'train_visual.py')]),
                ('closed_loop',[str(LAB/'visual_mission.py'),'--output',str(LAB/'runs/visual_mission_6101_02'),
                    '--checkpoint',str(LAB/'runs/visual_bootstrap_03/controller.npz')])]
        for phase,args in stages:
            state['phase']=phase;save()
            with (LAB/f'{phase}_v3.log').open('x',encoding='utf-8') as log:
                code=subprocess.run([sys.executable,'-B',*args],cwd=LAB,stdout=log,stderr=subprocess.STDOUT).returncode
            if code:raise RuntimeError(f'{phase} process failed: {code}')
        result=json.loads((LAB/'runs/visual_mission_6101_02/result.json').read_text())
        destination=LAB/'runs/visual_mission_6101_02'
        state['phase']='auditing';save()
        commands=[('mission_audit',['-m','astronaut_delivery.rgbd_mission_audit','--run',str(destination),
            '--scene',str(LAB.parent/'reports/rgbd_paired_mission_pilot_01/scene_6101/rgbd_random_6101.usda')]),
            ('wall_clock_audit',[str(LAB/'runtime/scripts/audit_rgbd_wall_clock.py'),'--run',str(destination)])]
        for label,args in commands:
            with (LAB/f'{label}_v3.log').open('x',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-B',*args],cwd=LAB/'runtime',stdout=log,stderr=subprocess.STDOUT)
        state.update(status='complete',task_status=result['status'],task_error=result.get('error'),
            observed_completed=result.get('completed_observed_targets',[]),
            visual_input_audit=json.loads((destination/'visual_input_audit.json').read_text()))
        save()
    except Exception:state.update(status='stopped',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
