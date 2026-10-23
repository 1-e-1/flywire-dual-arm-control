import numpy as np
from visual_circuit import pack_frame,VisualCircuit,RETINA_SIZE,AUX_SIZE

def frame():
    return dict(rgb=np.zeros((54,96,3),np.uint8),depth=np.ones((54,96),np.float32),q=np.zeros((2,8)),
        K=np.eye(3),camera_world=np.eye(4),exposure_s=0.,requested_wall_s=0.)

def test_all_visual_modalities_and_proprioception_reach_encoder():
    f=frame();x,history=pack_frame(f);assert x.shape==(RETINA_SIZE+AUX_SIZE,)
    for name in ('rgb','depth','q','K','camera_world'):
        changed={k:np.array(v).copy() for k,v in f.items()}
        changed[name].flat[0]+=1
        y,_=pack_frame(changed)
        assert not np.array_equal(x,y),name
    f['exposure_s']=.1;f['q'][0,0]=.01
    _,history2=pack_frame(f,history);assert np.isclose(history2['v'][0],.1)


def test_native_camera_eight_channel_area_resize():
    f=frame();f['rgb']=np.zeros((480,848,3),np.uint8);f['depth']=np.ones((480,848),np.float32)
    x,_=pack_frame(f)
    assert x.shape==(RETINA_SIZE+AUX_SIZE,) and np.isfinite(x).all()

def test_frame_content_changes_neural_state_and_no_frame_fails():
    c=VisualCircuit()
    try:c.features(np.zeros(47));assert False
    except ValueError:pass
    c.ingest(frame());old=c.state.copy()
    f=frame();f['rgb'][:]=255;c.ingest(f)
    assert c.frames==2 and not np.array_equal(old,c.state)
    assert np.isfinite(c.features(np.zeros(47))).all()


def test_external_initial_scan_without_joint_encoders_is_explicit():
    f=frame();del f['q'];x,history=pack_frame(f)
    assert not history['q_available'] and np.isfinite(x).all()
