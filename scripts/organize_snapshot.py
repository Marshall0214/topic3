"""Copy third-point assets into topic3; never write into a source tree."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, time
ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git','__pycache__','.pytest_cache','.mypy_cache','.venv'}

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def files(p):
    if p.is_file():yield p;return
    for parent,dirs,names in os.walk(p,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED and not (Path(parent)/d).is_symlink())
        for n in sorted(names):
            q=Path(parent)/n
            if not q.is_symlink():yield q

def mappings():
    project=Path(r'D:\action_compare\keraal_localization')
    for n in ['src','tests','scripts','configs','README.md']:
        yield project/n,ROOT/'legacy/principal_v1'/n
    yield project/'reports',ROOT/'reports/historical/principal_v1'
    for p in sorted((project/'artifacts').iterdir()):
        yield p,(ROOT/'runs/historical/principal_v1' if p.name=='runs' else ROOT/'manifests/historical/principal_v1'/p.name)
    dest=Path(r'D:\AQA\work\keraal_dest')
    for p in sorted(dest.iterdir()):
        if p.name in EXCLUDED:continue
        to=ROOT/'legacy/dest'/p.name
        if p.name=='runs':to=ROOT/'runs/historical/dest'
        elif p.name=='pretrained':to=ROOT/'upstream/dest_pretrained'
        yield p,to
    lit=Path(r'D:\AQA\research\keraal_literature')
    yield lit/'audit',ROOT/'legacy/literature_20261011/audit'
    yield lit/'upstream',ROOT/'upstream'
    for p in sorted(lit.iterdir()):
        if p.is_file():yield p,ROOT/'legacy/literature_20261011/documents'/p.name
    yield Path(r'D:\AQA\KeraalDataset'),ROOT/'upstream/keraal_gmm'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--copy',action='store_true');args=parser.parse_args()
    pairs=[]
    for source,destination in mappings():
        if not source.exists():raise FileNotFoundError(source)
        for f in files(source):
            to=destination if source.is_file() else destination/f.relative_to(source)
            assert to.resolve().is_relative_to(ROOT.resolve())
            pairs.append((f,to))
    assert len({str(d).lower() for _,d in pairs})==len(pairs),'duplicate destination'
    print(json.dumps({'files':len(pairs),'bytes':sum(s.stat().st_size for s,_ in pairs),'copy':args.copy}),flush=True)
    if not args.copy:return
    start=time.time();rows=[]
    for i,(s,d) in enumerate(pairs):
        stat=s.stat();h=digest(s)
        if d.exists():
            if digest(d)!=h:raise RuntimeError('Refusing to replace different destination: '+str(d))
        else:
            d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(s,d)
        if digest(d)!=h:raise RuntimeError('Copy hash mismatch: '+str(d))
        rows.append({'source':str(s),'destination':str(d.relative_to(ROOT)),'bytes':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'sha256':h})
        if (i+1)%250==0:print(f'copied and verified {i+1}/{len(pairs)}',flush=True)
    unchanged=[]
    for row in rows:
        s=Path(row['source']);stat=s.stat()
        if stat.st_mtime_ns!=row['source_mtime_ns'] or digest(s)!=row['sha256']:unchanged.append(str(s))
    pins={}
    for folder in [Path(r'D:\AQA\research\keraal_literature\upstream')/n for n in ['a-mal','SKVP-Python','RF-DTAN','dest']]+[Path(r'D:\AQA\KeraalDataset')]:
        r=subprocess.run(['git','-C',str(folder),'rev-parse','HEAD'],capture_output=True,text=True)
        pins[str(folder)]={'commit':r.stdout.strip() if r.returncode==0 else None,'git_dir_copied':False}
    result={'date':'2026-10-13','mode':'copy only; source content and mtime rechecked','root':str(ROOT),'excluded_directory_names':sorted(EXCLUDED),'raw_dataset_copied':False,'source_changes':unchanged,'files':rows,'upstream_pins':pins,'total_bytes':sum(r['bytes'] for r in rows),'elapsed_seconds':time.time()-start}
    out=ROOT/'evidence/migration';out.mkdir(parents=True,exist_ok=True)
    (out/'COPY_MANIFEST.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert not unchanged,unchanged
    print(json.dumps({'copied_files':len(rows),'source_changes':0,'total_MiB':round(result['total_bytes']/1048576,2),'elapsed_s':round(result['elapsed_seconds'],1)}),flush=True)
if __name__=='__main__':main()