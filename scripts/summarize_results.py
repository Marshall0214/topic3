"""Summarize already saved development results; no model inference or training."""
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
