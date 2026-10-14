from pathlib import Path
import json,hashlib,shutil,difflib
ROOT=Path(r'D:\topic3')
records=[]
def write(rel,text):
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8')
def derive(source,rel,text=None,note='exact working copy'):
 s=ROOT/source;d=ROOT/rel;d.parent.mkdir(parents=True,exist_ok=True)
 if text is None:shutil.copy2(s,d)
 else:d.write_text(text,encoding='utf-8')
 records.append({'snapshot':source,'working_file':rel,'snapshot_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'working_sha256':hashlib.sha256(d.read_bytes()).hexdigest(),'change':note})
 if text is not None:
  diff=''.join(difflib.unified_diff(s.read_text(encoding='utf-8').splitlines(True),text.splitlines(True),fromfile=source,tofile=rel))
  write('evidence/migration/patches/'+rel.replace('/','__')+'.patch',diff)

for d in ['configs/baselines','configs/experiments','src/topic3/data','src/topic3/features','src/topic3/alignment','src/topic3/reference','src/topic3/models','src/topic3/decoding','src/topic3/evaluation','scripts','tests','manifests/historical/principal_v1','runs','reports','evidence/migration']:(ROOT/d).mkdir(parents=True,exist_ok=True)
for p in (ROOT/'legacy/principal_v1/src/keraal_loc').glob('*.py'):derive(str(p.relative_to(ROOT)).replace('\\','/'),'src/keraal_loc/'+p.name)
for p in (ROOT/'legacy/principal_v1/configs').glob('*'):
 if p.is_file():derive(str(p.relative_to(ROOT)).replace('\\','/'),'configs/legacy_principal_v1/'+p.name)
paths={'dataset_root':'D:/AQA/dataset/keraal','historical_manifest_dir':'manifests/historical/principal_v1','upstream_dest':'upstream/dest','historical_results':'runs/historical','default_python':'C:/Users/admin/anaconda3/python.exe','timebase':'annotation15','note':'Manifest labels and splits copied exactly; not a newly frozen protocol.'}
write('configs/paths.json',json.dumps(paths,ensure_ascii=False,indent=2)+'\n')
write('configs/baselines/README.md','# 基线配置状态\n\n旧配置原样保存在 `../legacy_principal_v1/`。新统一基线尚未冻结，当前不生成假装可直接训练的 linear/DTW/GMM 配置。方法状态见根目录 BASELINE_REGISTER.md。\n')
write('configs/experiments/README.md','# 新实验约定\n\n新运行使用独立名称，并登记方法、监督、标注时钟、特征、参考、校准、解码与选模规则。当前整理未启动实验。\n')
write('src/topic3/__init__.py','"""Third-point workspace; historical algorithms are explicitly identified."""\n')
write('src/topic3/paths.py','''from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
DATASET=Path(CONFIG['dataset_root']).resolve()
MANIFESTS=(ROOT/CONFIG['historical_manifest_dir']).resolve()
UPSTREAM_DEST=(ROOT/CONFIG['upstream_dest']).resolve()
RUNS=ROOT/'runs'
def new_run(name):
    if not name or name in {'.','..','historical'} or any(c in name for c in '/\\\\:'):
        raise ValueError('Use a new single-directory run name, excluding historical')
    path=(RUNS/name).resolve()
    if path.parent!=RUNS.resolve():raise ValueError('Run must stay inside topic3/runs')
    if path.exists():raise FileExistsError('Run already exists; refusing to overwrite: '+str(path))
    return path
''')
write('src/topic3/data/__init__.py','''"""Frozen metadata access; does not relabel or read skeleton arrays."""
import json
from topic3.paths import MANIFESTS
from keraal_loc.audit import annotation
def historical_manifest():
    return [json.loads(line) for line in (MANIFESTS/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
def historical_splits():
    return json.loads((MANIFESTS/'splits.json').read_text(encoding='utf-8'))
''')
for rel,fn in [('features','features'),('alignment','align'),('decoding','geometric_decode'),('evaluation','event_iou, metrics'),('reference','reference_ids')]:
 write('src/topic3/'+rel+'/__init__.py','"""Historical Arms/Trunk single-event implementation; not a new validated method."""\nfrom keraal_loc.experiment import '+fn+'\n')
write('src/topic3/models/__init__.py','"""Model adapters; algorithm changes require new experiment registration."""\n')
write('src/topic3/models/legacy_tcn.py','"""Existing principal_v1 TCN, preserved without algorithm changes."""\nfrom keraal_loc.experiment import TCN\n')
s=(ROOT/'legacy/dest/stage1.py').read_text(encoding='utf-8')
s=s.replace("HERE = Path(__file__).resolve().parent\nSOURCE = Path('D:/AQA/research/keraal_literature/upstream/dest')","from topic3.paths import ROOT as HERE, MANIFESTS as ART, DATASET as DATA, UPSTREAM_DEST as SOURCE")
s=s.replace("ART = Path('D:/action_compare/keraal_localization/artifacts')\nDATA = Path('D:/AQA/dataset/keraal')",'# Paths are configured in topic3/configs/paths.json.')
s=s.replace("limitation='existing test results from older experiments have already been viewed; new test remains unopened'","limitation='historical test results have already been viewed; copied splits are not a newly untouched test set'")
derive('legacy/dest/stage1.py','src/topic3/models/dest_stage1.py',s,'Only local paths and test-history wording changed; no model/loss/selection changes. Invoke through scripts/train.py.')
test=(ROOT/'legacy/principal_v1/tests/test_experiment.py').read_text(encoding='utf-8')
test=test.replace("root=Path(__file__).resolve().parents[1]/'artifacts'","root=Path(__file__).resolve().parents[1]/'manifests/historical/principal_v1'")
derive('legacy/principal_v1/tests/test_experiment.py','tests/test_legacy_contract.py',test,'Only manifest directory rebased. Dataset/geometry/matching/reference/padding assertions unchanged.')
write('scripts/_bootstrap.py','''from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'runs/_cache/numba')
os.environ['MPLCONFIGDIR']=str(ROOT/'runs/_cache/matplotlib')
sys.path.insert(0,str(ROOT/'src'))
''')
write('scripts/train.py','''"""Rebased DeST stage1 entry; no training occurs for --help or --selfcheck."""
import _bootstrap
from _bootstrap import ROOT
import sys,runpy
from topic3.paths import new_run
if '--help' not in sys.argv and '-h' not in sys.argv and '--selfcheck' not in sys.argv:
    args=sys.argv[1:]
    name=next((x.split('=',1)[1] for x in args if x.startswith('--run=')),None)
    if '--run' in args:
        index=args.index('--run');name=args[index+1] if index+1<len(args) else None
    if not name:raise SystemExit('Training requires --run NEW_NAME; historical output cannot be overwritten.')
    new_run(name)
    timebase=next((x.split('=',1)[1] for x in args if x.startswith('--timebase=')),'annotation15')
    if '--timebase' in args:timebase=args[args.index('--timebase')+1]
    if timebase!='annotation15':raise SystemExit('Historical encoded clock is archived; current entry requires annotation15.')
runpy.run_module('topic3.models.dest_stage1',run_name='__main__')
''')
write('scripts/summarize_results.py','''"""Summarize already saved development results; no model inference or training."""
import _bootstrap
from _bootstrap import ROOT
import argparse,json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',default='historical/dest/stage1_clock15');a=p.parse_args()
folder=(ROOT/'runs'/a.run).resolve()
if not folder.is_relative_to((ROOT/'runs').resolve()):raise ValueError('Run outside topic3/runs')
files=sorted(folder.glob('fold*_seed*/result.json'))
if not files:raise SystemExit('No DeST fold result.json found: '+str(folder))
rows=[json.loads(x.read_text(encoding='utf-8')) for x in files]
by_seed={}
for r in rows:by_seed.setdefault(r['seed'],[]).append(r)
out={'scope':'saved development validation; fold overlap possible; not independent test','run':a.run,'seeds':{}}
for seed,group in by_seed.items():
    assert len({r['fold'] for r in group})==len(group)
    ms=[r['selected']['validation'] for r in group]
    total={'records':sum(m['records'] for m in ms),'normal_records':sum(m['normal_records'] for m in ms)}
    for key in ['temporal@0.3','temporal@0.5']:
        t={k:sum(m[key][k] for m in ms) for k in ['tp','fp','fn']}
        t['f1']=2*t['tp']/max(2*t['tp']+t['fp']+t['fn'],1);total[key]=t
    total['normal_false_alarm_rate']=sum(m['normal_records']*m['normal_false_alarm_rate'] for m in ms)/max(total['normal_records'],1)
    out['seeds'][str(seed)]=total
print(json.dumps(out,ensure_ascii=False,indent=2))
''')
write('scripts/check_environment.py','''"""Read-only environment inventory; installs nothing."""
import _bootstrap
from _bootstrap import ROOT
import sys,json,importlib.metadata,importlib.util
from topic3.paths import DATASET,MANIFESTS,UPSTREAM_DEST
packages={}
for name,module in [('numpy','numpy'),('scipy','scipy'),('scikit-learn','sklearn'),('torch','torch'),('numba','numba'),('opencv-python','cv2'),('matplotlib','matplotlib'),('PyYAML','yaml')]:
    try:version=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:version=None
    packages[name]={'available':importlib.util.find_spec(module) is not None,'version':version}
print(json.dumps({'python':sys.executable,'root':str(ROOT),'dataset_exists':DATASET.exists(),'manifest_exists':(MANIFESTS/'manifest.jsonl').exists(),'dest_exists':UPSTREAM_DEST.exists(),'packages':packages},ensure_ascii=False,indent=2))
''')
write('scripts/verify_migration.py','''"""Verify source files and copied snapshots against migration SHA-256; read only."""
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
''')
write('scripts/run_checks.ps1','''param([string]$Python = 'C:\\Users\\admin\\anaconda3\\python.exe')
$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NUMBA_CACHE_DIR = Join-Path $TaskRoot 'runs\\_cache\\numba'
$env:MPLCONFIGDIR = Join-Path $TaskRoot 'runs\\_cache\\matplotlib'
Push-Location -LiteralPath $TaskRoot
try {
    & $Python -B scripts/check_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment inventory failed' }
    & $Python -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw 'Contract checks failed' }
    & $Python -B scripts/train.py --selfcheck
    if ($LASTEXITCODE -ne 0) { throw 'DeST synthetic selfcheck failed' }
    & $Python -B scripts/summarize_results.py
    if ($LASTEXITCODE -ne 0) { throw 'Saved-results summary failed' }
    & $Python -B scripts/verify_migration.py
    if ($LASTEXITCODE -ne 0) { throw 'Source/snapshot verification failed' }
} finally { Pop-Location }
''')
write('.gitignore','''# Raw data remains external
__pycache__/
*.py[cod]
.pytest_cache/
*.nbc
*.nbi
.venv/
# Weights, caches, full predictions and archived run products
runs/
*.pt
*.pth
*.ckpt
*.npy
*.npz
legacy/**/details/
legacy/**/train/
legacy/**/validation/
legacy/**/predictions.json
legacy/**/validation_predictions.json
# Local editor/runtime state
.vscode/
.idea/
''')
write('pyproject.toml','''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "topic3-motion-analysis"
version = "0.1.0"
description = "Reference-based motion deviation research workspace"
requires-python = ">=3.9"
dependencies = ["numpy", "scipy", "scikit-learn", "torch", "numba", "matplotlib"]

[tool.setuptools.packages.find]
where = ["src"]
include = ["topic3*", "keraal_loc*"]
''')
write('evidence/migration/DERIVED_FILES.json',json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'working_copies':len(records),'created_entrypoints':['train.py','check_environment.py','summarize_results.py','verify_migration.py','run_checks.ps1']}))