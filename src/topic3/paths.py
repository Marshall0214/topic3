from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
CONFIG=json.loads((ROOT/'configs/paths.json').read_text(encoding='utf-8'))
DATASET=Path(CONFIG['dataset_root']).resolve()
MANIFESTS=(ROOT/CONFIG['historical_manifest_dir']).resolve()
UPSTREAM_DEST=(ROOT/CONFIG['upstream_dest']).resolve()
RUNS=ROOT/'runs'
def new_run(name):
    if not name or name in {'.','..','historical'} or any(c in name for c in '/\\:'):
        raise ValueError('Use a new single-directory run name, excluding historical')
    path=(RUNS/name).resolve()
    if path.parent!=RUNS.resolve():raise ValueError('Run must stay inside topic3/runs')
    if path.exists():raise FileExistsError('Run already exists; refusing to overwrite: '+str(path))
    return path
