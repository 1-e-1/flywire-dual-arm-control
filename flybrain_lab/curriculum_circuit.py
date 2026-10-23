"""Same full fly recurrent graph; explicit visual-quality ports and task readouts."""
import numpy as np
from visual_circuit import VisualCircuit,RETINA_SIZE,AUX_SIZE
from connectome import observation

TASK_SIZE=60
REST_INDEX=58

def task_observation(q,left,right,allocator,goal,visual,covariance,pixels,age,phase,held,right_rest):
    basic=observation(q,left,right,goal,visual,phase,held)
    context=allocator.context(np.asarray(q).reshape(2,8)[0],visual,covariance,pixels,age,require_feasible=False)
    margins=allocator.margins(np.asarray(q).reshape(2,8)[0],context)
    return np.clip(np.r_[basic,margins,context['sigma']/.005,pixels/100.,age/.25,
        float(right_rest),allocator.cfg.cap_radius_m/.02],-5,5).astype(np.float32)

class CurriculumCircuit(VisualCircuit):
    def __init__(self):
        super().__init__();ports=np.random.default_rng(8601).permutation(self.W.shape[0])
        self.task_ports=ports[RETINA_SIZE+AUX_SIZE:RETINA_SIZE+AUX_SIZE+TASK_SIZE]
        # Read neurons connected to the engineered task ports, not random distant
        # cells with nearly zero task response. No new recurrent edge is added.
        score=np.asarray(abs(self.W[:,self.task_ports]).sum(axis=1)).ravel()
        score[self.frame_ports]=-1.;score[self.task_ports]=-1.
        self.read_ports=np.argsort(score,kind='stable')[-2048:]

    def features(self,task):
        task=np.asarray(task,np.float32)
        if task.shape!=(TASK_SIZE,) or not np.isfinite(task).all():raise ValueError('invalid visual-quality task input')
        if not self.frames:raise ValueError('actual RGB-D frame required')
        h=self.state.copy();drive=self.drive.copy();drive[self.task_ports]=np.clip(task,-5,5)
        for _ in range(4):h=.5*h+.5*np.tanh(self.W@h+drive)
        return h[self.read_ports]

class CurriculumController:
    def __init__(self,path):
        self.circuit=CurriculumCircuit();data=np.load(path)
        if int(data['visual_schema'])!=4:raise ValueError('curriculum checkpoint required')
        self.mean=data['mean'];self.scale=data['scale'];self.travel=data['travel'];self.acquire=data['acquire']
    def action(self,x):
        h=np.r_[(self.circuit.features(x)-self.mean)/self.scale,1.]
        value=h@(self.acquire if x[REST_INDEX]>.5 else self.travel)
        bounds=np.r_[np.full(8,.12),np.full(8,.3)]
        return np.clip(value,-bounds,bounds)
