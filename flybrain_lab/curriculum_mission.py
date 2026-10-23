"""Trained visual acquisition/readout; still no online teacher/IK/QP fallback."""
import argparse,json
from pathlib import Path
from visual_mission import VisualMission,LAB
from curriculum_circuit import CurriculumController,task_observation

class CurriculumMission(VisualMission):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,controller_class=CurriculumController,**kwargs)
        self.residual_profile.update(visual_schema=4,task_features=60,training='visual-acquisition curriculum')
    def policy_observation(self,target,goal,visual,right_rest=False):
        return task_observation(self.q,self.left,self.right,self.allocator,goal,visual,target.covariance,
            target.pixels,self.time-target.stamp,self.transfer_phase,self.held is not None,right_rest)

def main():
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--seed',type=int,default=6101);a=p.parse_args()
    base=LAB.parent/('reports/rgbd_paired_mission_pilot_01' if a.seed==6101 else 'reports/rgbd_holdout_01')
    if not Path(a.output).resolve().is_relative_to(LAB):raise ValueError('isolated output required')
    with threadpool_limits(1):
        task=CurriculumMission(a.output,str(base/f'scene_{a.seed}/rgbd_random_{a.seed}.usda'),str(base/f'recording_{a.seed}.json'),
            f'rgbd_random_{a.seed}','660ece523b85804d9a24fc37be455e4dbcfe9b6a178be2e0bd4254e2488e05f1',
            checkpoint=a.checkpoint,assume_unchanged_scene=True,mode='set',asynchronous=True,contact_permission=True,
            capture=True,local_contact_map=True,static_interval_fusion=True,local_map_cell_budget=16000000)
        for path in LAB.glob('*.py'):(task.output/path.name).write_bytes(path.read_bytes())
        task.run(48,.007,'auto')
if __name__=='__main__':main()
