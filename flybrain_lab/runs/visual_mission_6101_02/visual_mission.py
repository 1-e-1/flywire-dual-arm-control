"""Every delivered accepted RGB-D frame updates persistent fly-circuit state."""
import argparse,json,time
from pathlib import Path
import numpy as np
from mission import FlyMission,LAB
from visual_circuit import VisualController,ingest_initial_scan

class VisualMission(FlyMission):
    def __init__(self,*args,**kwargs):
        self.visual_frame_log=[]
        super().__init__(*args,controller_class=VisualController,**kwargs)
        self.residual_profile.update(visual_schema=3,retina='96x54x8, lossy area pooling of all delivered pixels')
        self.initial_scan_neural_frames=ingest_initial_scan(self.fly.circuit,self.scan_source[0])
        # Startup commissioning RGB-D are real sensor records, not reconstructed points.
        for path in sorted((self.output/'commissioning').glob('rgbd_*.npz')):
            with np.load(path) as f:self.fly.circuit.ingest(dict(f))
        self.commissioning_neural_frames=self.fly.circuit.frames-self.initial_scan_neural_frames

    def accept_packet(self,packet,integrate=False):
        obs=super().accept_packet(packet,integrate)
        if obs is not None:
            began=time.perf_counter();last=None if not self.steps else self.steps[-1]['wrench']
            self.fly.circuit.ingest(packet,wrench=last,received_wall=began)
            self.visual_frame_log.append(dict(packet_id=packet['id'],stamp=packet['stamp'],
                shape=list(packet['depth'].shape),neurally_ingested=True,wall_s=time.perf_counter()-began))
        elif packet is not None:
            self.visual_frame_log.append(dict(packet_id=packet['id'],neurally_ingested=False,reason='Rejected sensor frame by existing freshness gate'))
        return obs

    def run(self,*args,**kwargs):
        result=super().run(*args,**kwargs)
        (self.output/'visual_input_audit.json').write_text(json.dumps(dict(frames=self.visual_frame_log,
            initial_scan_neural_frames=self.initial_scan_neural_frames,
            commissioning_neural_frames=self.commissioning_neural_frames,total_neural_frames=self.fly.circuit.frames,
            accepted_frames=len(self.observations),scope='Every frame accepted by controller; sensor latest-frame transport may drop frames before delivery; no point-cloud encoder'),indent=2),encoding='utf-8')
        return result

def main():
    from threadpoolctl import threadpool_limits
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
    a=p.parse_args();base=LAB.parent/'reports/rgbd_paired_mission_pilot_01'
    if not Path(a.output).resolve().is_relative_to(LAB):raise ValueError('output must remain isolated')
    with threadpool_limits(1):
        task=VisualMission(a.output,str(base/'scene_6101/rgbd_random_6101.usda'),str(base/'recording_6101.json'),
            'rgbd_random_6101','660ece523b85804d9a24fc37be455e4dbcfe9b6a178be2e0bd4254e2488e05f1',
            checkpoint=a.checkpoint,assume_unchanged_scene=True,mode='set',asynchronous=True,contact_permission=True,
            capture=True,local_contact_map=True,static_interval_fusion=True,local_map_cell_budget=16000000)
        for name in ('visual_circuit.py','visual_mission.py','mission.py'):
            (task.output/name).write_bytes((LAB/name).read_bytes())
        task.run(48,.007,'auto')
if __name__=='__main__':main()
