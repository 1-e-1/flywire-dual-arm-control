"""Read assembly component transforms without modifying the CAD document."""
import json
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, __import__('os').environ.get('SOLIDWORKS_AUTOMATION_SCRIPTS', str(__import__('pathlib').Path(__file__).resolve().parents[3] / 'optional_tools/solidworks-automation/scripts')))
from sw_session import SolidWorksSession
from sw_connect import get_com_member

root = Path(__file__).resolve().parents[1]
path = root/'solidwork_model'/'7_assembly_2_2_no.SLDASM'
source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
session = SolidWorksSession(visible=False)
doc = session.open(str(path),read_only=True,silent=True)
components = get_com_member(doc,'GetComponents',False) or []
rows=[]
for c in components:
    tr = get_com_member(c,'Transform2')
    rows.append(dict(name=get_com_member(c,'Name2'),path=get_com_member(c,'GetPathName'),transform=list(get_com_member(tr,'ArrayData')) if tr else None))
out = dict(source=str(path),connection=session.connection_info,components=rows)
out['source_sha256']=source_hash
out['missing_reference_paths']=sorted({c['path'] for c in rows if not Path(c['path']).exists()})
out['part_bounds_m'] = {}
for name in ('7_lift_2_2.SLDPRT', '7_mainpart1_2_2.SLDPRT'):
    part = session.open(str(root/'solidwork_model'/name),read_only=True,silent=True)
    out['part_bounds_m'][name] = list(get_com_member(part,'GetPartBox',True))
assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash,'CAD reference changed during read'
out['source_unchanged']=True
(root/'isaacsim_model'/'solidworks_reference.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
print(json.dumps(dict(parts=out['part_bounds_m'],components=len(rows)),indent=2,default=str))
# Leave a pre-existing user session untouched; never save this reference.
