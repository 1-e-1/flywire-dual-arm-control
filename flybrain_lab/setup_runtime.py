"""Freeze private copies of numerical/sensor utilities, never edit the parent."""
import hashlib,json,shutil
from pathlib import Path
LAB=Path(__file__).resolve().parent
def main():
    parent=LAB.parent;dest=LAB/'runtime';dest.mkdir(exist_ok=False)
    manifest={}
    for name in ('astronaut_delivery','classical_fusion','codex_zbot','scripts','configs'):
        for p in (parent/name).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml') and '__pycache__' not in p.parts:
                relative=p.relative_to(parent);out=dest/relative;out.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,out);manifest[str(relative)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (LAB/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Private runtime copied:',len(manifest))
if __name__=='__main__':main()
