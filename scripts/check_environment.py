"""Read-only environment inventory; installs nothing."""
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
