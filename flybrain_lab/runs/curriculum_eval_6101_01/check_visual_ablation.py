"""Causal input wiring test, NOT a task-level ablation evaluation."""
import json
import numpy as np
from pathlib import Path
from visual_circuit import VisualController
LAB=Path(__file__).resolve().parent
def main():
    file=LAB.parent/'reports/rgbd_quality_handover_regression_6101_01/raw/rgbd_00000.npz'
    with np.load(file) as data:original=dict(data)
    task=np.zeros(47,np.float32);task[32]=.1
    outputs={}
    for mode in ('observed','black_rgb','missing_depth'):
        frame={k:np.array(v).copy() for k,v in original.items()}
        if mode=='black_rgb':frame['rgb'][:]=0
        if mode=='missing_depth':frame['depth'][:]=0
        model=VisualController(LAB/'runs/visual_bootstrap_02/controller.npz')
        model.circuit.ingest(frame);outputs[mode]=model.action(task).tolist()
    result=dict(actions=outputs,rgb_action_difference_norm=float(np.linalg.norm(np.array(outputs['observed'])-outputs['black_rgb'])),
        depth_action_difference_norm=float(np.linalg.norm(np.array(outputs['observed'])-outputs['missing_depth'])),
        scope='Paired identical proprio/task input; changing actual image changes final policy output. Does not prove useful semantic vision or task success.')
    (LAB/'runs/visual_bootstrap_02/input_ablation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
