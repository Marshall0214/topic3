from pathlib import Path
from collections import Counter,defaultdict
import xml.etree.ElementTree as E
import numpy as np,struct,json,hashlib
P=Path(r'D:\AQA\dataset\keraal')
def boxes(d):
 p=0
 while p+8<=len(d):
  s,t=struct.unpack('>I4s',d[p:p+8]);h=8
  if s==1:s=struct.unpack('>Q',d[p+8:p+16])[0];h=16
  if s==0:s=len(d)-p
  if s<h:raise ValueError('invalid box')
  yield t,d[p+h:p+s];p+=s
def child(d,t):return next(v for k,v in boxes(d) if k==t)
def video_meta(p):
 with open(p,'rb') as h:
  while True:
   v=h.read(8)
   if len(v)<8:raise ValueError('no moov')
   s,t=struct.unpack('>I4s',v)
   if t==b'moov':d=h.read(s-8);break
   h.seek(s-8,1)
 for k,tr in boxes(d):
  if k!=b'trak':continue
  md=child(tr,b'mdia')
  if child(md,b'hdlr')[8:12]!=b'vide':continue
  v=child(md,b'mdhd')
  assert v[0]==0,'unsupported version'
  scale,dur=struct.unpack('>II',v[12:20]);sz=child(child(child(md,b'minf'),b'stbl'),b'stsz');frames=struct.unpack('>I',sz[8:12])[0]
  return dict(frames=frames,fps=frames/(dur/scale),duration_s=dur/scale)
def ann(p):
 root=E.parse(p).getroot();tracks={};invalid=[]
 for tr in root.findall('./body/track'):
  name=tr.get('name');tracks[name]=[]
  for e in tr.findall('el'):
   attrs={a.get('name'):a.text for a in e.findall('attribute')}
   try:s=float(e.get('start'));end=float(e.get('end'))
   except:invalid.append(dict(track=name,raw=e.attrib,attrs=attrs));s=end=None
   tracks[name].append(dict(start=s,end=end,attrs=attrs))
 return tracks,invalid
rows=[];summaries=[]
for group,pre in [('group1A','G1A'),('group2A','G2A')]:
 for af in sorted((P/group/'annotatorA').glob(pre+'*.anvil')):
  toks=af.stem.split('-');a,ai=ann(af);bf=P/group/'annotatorB'/af.name;b,bi=ann(bf);sk=P/group/'kinect'/('-'.join([toks[0],'Kinect']+toks[1:])+'.txt');vf=P/group/'videos'/('-'.join([toks[0],'Anon']+toks[1:])+'.mp4');x=np.loadtxt(sk);vm=video_meta(vf);valid_b=[e['end'] for e in b.get('Global evaluation',[]) if e['end'] is not None]
  rows.append(dict(id=af.stem,group=group,subject_key='-'.join(toks[2:-1]),exercise=toks[1],a_path=str(af),b_path=str(bf),skeleton_path=str(sk),video_path=str(vf),skeleton_shape=list(x.shape),skeleton_finite=bool(np.isfinite(x).all()),video=vm,video_skeleton_equal=vm['frames']==len(x),b_clock15_abs_error=abs(max(valid_b)-len(x)/15) if valid_b else None,a=a,b=b,a_invalid=ai,b_invalid=bi))
 for annotator in ['a','b']:
  rr=[r for r in rows if r['group']==group];te=[(r,e) for r in rr for e in r[annotator].get('Temporal error',[])];gg=[e['attrs'].get('evaluation') for r in rr for e in r[annotator].get('Global evaluation',[])]
  summaries.append(dict(group=group,annotator=annotator,recordings=len(rr),subjects=sorted(set(r['subject_key'] for r in rr)),global_evaluations=dict(Counter(gg)),temporal_events=len(te),temporal_recordings=sum(bool(r[annotator].get('Temporal error')) for r in rr),complete_temporal_events=sum(e['start'] is not None and e['end'] is not None and all(k in e['attrs'] for k in ['type','Evaluation','bodyPart']) for _,e in te),temporal_parts=dict(Counter(e['attrs'].get('bodyPart') for _,e in te)),temporal_subjects=dict(Counter(r['subject_key'] for r,e in te)),temporal_exercises=dict(Counter(r['exercise'] for r,e in te)),incorrect_no_temporal=sum(any(e['attrs'].get('evaluation')=='Incorrect' for e in r[annotator].get('Global evaluation',[])) and not r[annotator].get('Temporal error') for r in rr)))
clock=[r['b_clock15_abs_error'] for r in rows if r['b_clock15_abs_error'] is not None];events=[(r,e) for r in rows for e in r['a'].get('Temporal error',[])];both=[r for r in rows if r['a'].get('Temporal error') and r['b'].get('Temporal error')];g3=list((P/'group3'/'kinect').glob('*.txt'))
result=dict(root=str(P),actual_recordings=len(rows),summaries=summaries,all_skeleton_columns=dict(Counter(str(r['skeleton_shape'][1]) for r in rows)),all_skeleton_finite=all(r['skeleton_finite'] for r in rows),video_fps=dict(Counter(str(r['video']['fps']) for r in rows)),video_skeleton_matches=sum(r['video_skeleton_equal'] for r in rows),clock15_valid=len(clock),clock15_max_abs_error=max(clock),anomalies=[r for r in rows if not r['video_skeleton_equal'] or r['a_invalid'] or r['b_invalid']],a_event_durations_quantiles=np.quantile([e['end']-e['start'] for r,e in events],[0,.25,.5,.75,1]).tolist(),a_events_outside15=[r['id'] for r,e in events if not 0<=e['start']<e['end']<=r['skeleton_shape'][0]/15+1/15],a_correct_records=sum(any(e['attrs'].get('evaluation')=='Correct' for e in r['a'].get('Global evaluation',[])) for r in rows),a_incomplete_motionless_records=sum(any(e['attrs'].get('evaluation') in ['Incomplete','Motionless'] for e in r['a'].get('Global evaluation',[])) for r in rows),ab_both_temporal_records=[r['id'] for r in both],ab_exact_nonempty_temporal_matches=sum(r['a']['Temporal error']==r['b']['Temporal error'] for r in both),group3=dict(recordings=len(g3),subjects=dict(Counter(f.stem.split('-')[3].split('T')[0] for f in g3)),labels=dict(Counter(f.stem.split('-')[-2] for f in g3)),exercises=dict(Counter(f.stem.split('-')[2] for f in g3))),records=rows)
print(json.dumps(result,ensure_ascii=False,indent=2))
