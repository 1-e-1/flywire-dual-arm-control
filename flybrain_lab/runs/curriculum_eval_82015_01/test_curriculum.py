from types import SimpleNamespace
import numpy as np
from curriculum_circuit import task_observation,CurriculumCircuit,REST_INDEX,TASK_SIZE
from test_visual_input import frame

def test_quality_and_task_mode_are_explicit_inputs():
    arm=SimpleNamespace(tcp=lambda q:np.eye(4),camera=lambda q:np.eye(4))
    allocator=SimpleNamespace(cfg=SimpleNamespace(cap_radius_m=.02),
        context=lambda q,p,cov,pix,age,**kw:dict(sigma=float(np.sqrt(cov[0,0]))),margins=lambda *a:np.arange(8))
    args=(np.zeros((2,8)),arm,arm,allocator,np.zeros(3),np.zeros(3),np.eye(3)*1e-6,50,0.,None,False)
    x=task_observation(*args,False);y=task_observation(*args,True)
    assert x.shape==(TASK_SIZE,) and x[REST_INDEX]==0 and y[REST_INDEX]==1
    assert np.isclose(x[55],.2) and np.isclose(x[56],.5)

def test_curriculum_keeps_graph_and_reads_real_downstream_cells():
    c=CurriculumCircuit()
    assert c.W.nnz==15091983
    assert len(c.task_ports)==TASK_SIZE and len(c.read_ports)==2048
    assert not np.intersect1d(c.read_ports,c.task_ports).size
    assert not np.intersect1d(c.read_ports,c.frame_ports).size
    c.ingest(frame());state=c.state.copy();x=np.zeros(TASK_SIZE,np.float32)
    h=c.features(x);x[REST_INDEX]=1
    assert not np.array_equal(h,c.features(x))
    np.testing.assert_array_equal(c.state,state)
