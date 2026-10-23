"""Read-only USD asset inspection, using the IsaacLab Python environment."""
import json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(str(__import__('pathlib').Path(__import__('os').environ.get('ZBOT_V63_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'external_assets/zbot-v63')))))
files = [ROOT/'isaacsim_model'/f'zbot_8s_new_{side}.usd' for side in ('left','right')]
files += [ROOT/'zbot_8s_new.usd']
files += [SOURCE/name for name in ('C_structure.usd','cixi.usd','assets/nasa_astronaut/astronaut.usd','finally_astronaut_candidate.usd')]
result = []
for path in files:
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default','render','proxy'])
    record = dict(file=str(path), default=str(stage.GetDefaultPrim().GetPath()), up=str(UsdGeom.GetStageUpAxis(stage)), meters=UsdGeom.GetStageMetersPerUnit(stage), prims=[])
    for prim in stage.Traverse():
        if prim.GetParent().IsPseudoRoot() or prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdGeom.Camera) or prim.GetName().lower() in ('astronaut','c_structure'):
            item = dict(path=str(prim.GetPath()), type=prim.GetTypeName(), schemas=list(prim.GetAppliedSchemas()))
            if prim.IsA(UsdGeom.Xformable):
                item['world'] = [list(row) for row in UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())]
                b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                item['bounds'] = [list(b.GetMin()),list(b.GetMax())]
            item['attributes'] = {a.GetName():str(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(('physics:','xformOp:')) or prim.IsA(UsdGeom.Camera)}
            item['relationships'] = {r.GetName():[str(t) for t in r.GetTargets()] for r in prim.GetRelationships()}
            record['prims'].append(item)
    record['dependencies'] = [l.identifier for l in stage.GetUsedLayers()]
    result.append(record)
out = ROOT/'isaacsim_model'/'asset_inspection.json'
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
for r in result:
    print(r['file'],r['default'],r['up'],r['meters'])
    for p in r['prims']:
        print(p['path'],p['type'],p.get('bounds',''),p['relationships'] if 'Joint' in p['type'] else '')
