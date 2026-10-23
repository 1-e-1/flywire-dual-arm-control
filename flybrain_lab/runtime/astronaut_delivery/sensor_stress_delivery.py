"""Held-out engineering sensor stresses, separate from the frozen primary batch."""
import argparse,copy,json
from pathlib import Path
from threadpoolctl import threadpool_limits
from .sufficiency_delivery import SufficiencyDelivery


class StressDelivery(SufficiencyDelivery):
    def __init__(self,*args,stress='noise2',**kwargs):
        super().__init__(*args,**kwargs);self.stress=stress;self.blank_frames=0
        # Do not mutate the shared config or its source hash.
        self.camera.profile=copy.deepcopy(self.camera.profile)
        noise=self.camera.profile['simulation_noise']
        if stress=='noise2':
            noise['floor_sigma_m']*=2;noise['disparity_sigma_px']*=2
        elif stress=='dropout20':noise['dropout_probability']=.2
        elif stress not in ('blank_transient','blank_permanent'):raise ValueError('unknown stress')
        original=self.camera.capture
        def capture(*args,**kw):
            obs=original(*args,**kw)
            blank=(stress=='blank_transient' and 40<=obs.stamp<44) or (stress=='blank_permanent' and obs.stamp>=40)
            if blank:
                obs.rgb.fill(0);obs.depth.fill(0)
                if obs.color_valid is not None:obs.color_valid.fill(False)
                self.blank_frames+=1
            return obs
        # capture_roi calls self.capture internally, covering both native/crop.
        self.camera.capture=capture

    def save(self,error=None):
        super().save(error);p=self.output/'result.json';r=json.loads(p.read_text())
        r['sensor_stress']=dict(mode=self.stress,blank_frames=self.blank_frames,profile=self.camera.profile,
             scope='Synthetic sensor failure/noise experiment, not calibration or a physical occluder model.')
        p.write_text(json.dumps(r,indent=2),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--stress',required=True,choices=['noise2','dropout20','blank_transient','blank_permanent']);args=p.parse_args()
    with threadpool_limits(1):
        run=StressDelivery(args.output,stress=args.stress,anytime_feasible=True,telemetry=True)
        try:run.run(3)
        except Exception as exc:run.save(str(exc));raise
        else:run.save()


if __name__=='__main__':main()
