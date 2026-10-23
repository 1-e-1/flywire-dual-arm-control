import ast,json
from pathlib import Path
import numpy as np
from connectome import FlyCircuit,INPUTS
LAB=Path(__file__).resolve().parent

def test_full_supplied_graph_and_finite_dynamics():
    c=FlyCircuit();m=json.loads((LAB/'data/graph/manifest.json').read_text())
    assert c.W.shape==(m['neurons'],m['neurons'])
    assert c.W.nnz==m['source_rows']==15091983
    assert np.asarray(abs(c.W).sum(axis=1)).max()<.851
    assert not np.intersect1d(c.inputs,c.outputs).size
    assert np.array_equal(c.features(np.zeros(INPUTS)),np.zeros(2048))
    assert np.isfinite(c.features(np.ones(INPUTS))).all()

def test_no_ik_qp_or_ppo_fallback_in_direct_controller():
    tree=ast.parse((LAB/'mission.py').read_text())
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert 'solve' not in calls and 'propose' not in calls and 'residual_proposal' not in calls
    assert 'segment' in calls and 'certify_visual_path' in calls and 'endpoint_valid' in calls
    methods={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    assert {'plan_step','plan_visual_acquisition','plan_observation_recovery'}<=methods
