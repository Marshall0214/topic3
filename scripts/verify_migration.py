"""Verify source files and copied snapshots against migration SHA-256; read only."""
import _bootstrap
from _bootstrap import ROOT
import json,hashlib,sys
manifest=json.loads((ROOT/'evidence/migration/COPY_MANIFEST.json').read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
from pathlib import Path
failures=[]
for r in manifest['files']:
    source=Path(r['source']);target=ROOT/r['destination']
    if not source.exists() or digest(source)!=r['sha256'] or source.stat().st_mtime_ns!=r['source_mtime_ns']:failures.append({'source_changed':str(source)})
    if not target.exists() or digest(target)!=r['sha256']:failures.append({'snapshot_changed':str(target)})
print(json.dumps({'copied_files':len(manifest['files']),'source_and_snapshot_checks_passed':not failures,'failures':failures},ensure_ascii=False,indent=2))
sys.exit(bool(failures))
