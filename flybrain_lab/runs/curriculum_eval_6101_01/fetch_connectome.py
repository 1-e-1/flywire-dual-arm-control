"""Download only the paper authors' public data; never substitute synthetic edges."""
import hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    api='https://api.github.com/repos/philshiu/Drosophila_brain_model/commits/main'
    commit=json.load(urllib.request.urlopen(api,timeout=60))['sha']
    folder=ROOT/'data'/'upstream';folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in ('LICENSE','Readme.md','model.py','Completeness_783.csv','Connectivity_783.parquet'):
        url=f'https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{commit}/{name}'
        path=folder/name
        if path.exists():raise FileExistsError(path)
        print('Downloading '+name,flush=True)
        with urllib.request.urlopen(url,timeout=120) as source,path.open('xb') as out:
            while chunk:=source.read(1024*1024):out.write(chunk)
        if path.read_bytes()[:60].startswith(b'version https://git-lfs'):raise RuntimeError('Git LFS pointer, not dataset')
        rows.append(dict(file=name,url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (folder/'manifest.json').write_text(json.dumps(dict(commit=commit,files=rows,
        source='Paper authors: philshiu/Drosophila_brain_model; public FlyWire v783'),indent=2),encoding='utf-8')
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
