"""One VLM-selected transfer, followed by the existing RGB-D-only guarded controller."""
import argparse
import json
import subprocess
import sys
import time

from threadpoolctl import threadpool_limits
from .rgbd_transfer import RGBDTransfer
from .vlm_task_contract import prepare_request, validate_response, validate_reobservation, requested_interface, validate_intent


class RGBDVLMTransfer(RGBDTransfer):
    def approach(self, *args, **kwargs):
        if not getattr(self, '_vlm_rebound', False):
            estimate = self.guide.target(self.active, self.time)
            validate_reobservation(self.vlm_binding, estimate, self.time)
            self._vlm_rebound = True
            self.task_metadata['vlm_fresh_binding'] = dict(measured_id=self.active,
                position_m=estimate.position.tolist(), motion_time_s=self.time,
                note='Identity re-observed after model process exits; every execution still refreshes wall-clock guidance')
        return super().approach(*args, **kwargs)

    def run_vlm(self, instruction, model_path, steps=48):
        began=time.perf_counter();folder=self.output/'vlm'
        try:
            requested_interface(instruction)
            obs=self.observe(integrate=True)
            request, bindings=prepare_request(obs, self.guide.estimates,
                self.output/'raw'/self.observations[-1]['file'], instruction, folder)
            # The child receives only RGB crops and language, never USD or controller objects.
            with (folder/'worker.log').open('x', encoding='utf-8') as log:
                child=subprocess.run([sys.executable, '-m', 'astronaut_delivery.vlm_local_worker',
                    '--request-dir', str(folder.resolve()), '--model', str(model_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
            if child.returncode:
                raise RuntimeError('local_vlm_process_failed_see_worker_log')
            selection=validate_response(folder)
            self.vlm_binding=bindings[selection]
            validate_intent(instruction, self.vlm_binding)
            plan=dict(schema='measured-transfer-task-v1', action='transfer_one_marked_object',
                selected_candidate=selection, measured_id=self.vlm_binding['measured_id'],
                destination_policy='nearest_observed_available_mark',
                states=['reobserve','approach','capture','lift','transport','place','release','retreat'],
                state_machine_source='validated existing controller template, NOT VLM-generated executable code',
                instruction=instruction, upstream_wall_s=time.perf_counter()-began,
                semantic_gate='Independent measured known-interface colour check; rejection only, no target fallback',
                vlm_unloaded_before_motion=True, scope='Development integration; one task, not a VLM accuracy benchmark')
            (folder/'validated_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
            self.task_metadata['vlm_task']=plan
        except Exception as exc:
            self.flush_raw()
            if self.sensor_service is not None:self.sensor_service.close()
            self.output.joinpath('vlm_gate_failure.json').write_text(json.dumps(dict(error=str(exc),
                motion_executed=False, hardcoded_target_fallback=False),indent=2),encoding='utf-8')
            raise
        return super().run(steps,.007,self.vlm_binding['measured_id'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--initial-scan',required=True);p.add_argument('--scene-token',required=True);p.add_argument('--calibration-id',required=True)
    p.add_argument('--assume-unchanged-scene',action='store_true');p.add_argument('--instruction',required=True,
        help='Limited validated language: Select the green interface. (red/green/blue); action is fixed transfer, not arbitrary instructions')
    p.add_argument('--model',default='checkpoints/SmolVLM-500M-Instruct');p.add_argument('--steps',type=int,default=48);a=p.parse_args()
    with threadpool_limits(1):
        task=RGBDVLMTransfer(a.output,a.scene,a.initial_scan,a.scene_token,a.calibration_id,
            assume_unchanged_scene=a.assume_unchanged_scene, mode='set', asynchronous=True,
            contact_permission=True,capture=True,local_contact_map=True,static_interval_fusion=True)
        result=task.run_vlm(a.instruction,a.model,a.steps)
        if not result.get('single_transfer_cycle_completed'):raise SystemExit(2)


if __name__=='__main__':main()
