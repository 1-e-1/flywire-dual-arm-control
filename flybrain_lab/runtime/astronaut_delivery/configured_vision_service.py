"""Worker-local map factory adapter, preserving the frozen service implementation.

The adapter changes an import only inside its OWN spawned sensor process. It
does not touch the parent process, the frozen controller, or the USD files.
"""
import multiprocessing as mp
from dataclasses import replace
from .vision_service import VisionService,vision_process


def configured_vision_process(requests,results,folder,seed,device,scene_path,task_config,telemetry):
    from . import vision
    from .adaptive_planning_map import AdaptivePlanningTSDF
    original=vision.VisualGuide
    class ConfiguredGuide(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.mapper=AdaptivePlanningTSDF(replace(self.mapper.cfg,voxel_size_m=.004,truncation_m=.012),device=device)
    vision.VisualGuide=ConfiguredGuide
    try:vision_process(requests,results,folder,seed,device,scene_path,task_config,telemetry)
    finally:vision.VisualGuide=original


class ConfiguredVisionService(VisionService):
    def __init__(self,folder,seed,device='cuda:0',scene_path=None,task_config=None,telemetry=False):
        context=mp.get_context('spawn');self.requests=context.Queue(maxsize=1);self.results=context.Queue(maxsize=2)
        self.process=context.Process(target=configured_vision_process,args=(self.requests,self.results,str(folder),seed,device,scene_path,task_config,telemetry),daemon=True)
        self.process.start();self.serial=0;self.last_id=-1;self.closed=False
