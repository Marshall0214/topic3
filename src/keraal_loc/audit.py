"""Audit the sole dataset entry, then freeze participant splits before training."""
import argparse, collections, hashlib, json, re
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np

PARTS = {'LeftArm':'Arms','RightArm':'Arms','BothArms':'Arms','Spine':'Trunk'}

def annotation(path):
    root = ET.fromstring(Path(path).read_bytes())
    label, events = None, []
    for track in root.findall('.//track'):
        for el in track.findall('el'):
            attrs = {a.attrib['name'].lower():a.text for a in el.findall('attribute')}
            if track.attrib['name'] == 'Global evaluation':
                label = attrs.get('evaluation')
            if track.attrib['name'] == 'Temporal error':
                events.append(dict(start_s=float(el.attrib['start']), end_s=float(el.attrib['end']),
                    part=PARTS.get(attrs.get('bodypart')), part_raw=attrs.get('bodypart'),
                    error_type=attrs.get('type'), severity=attrs.get('evaluation')))
    return label, events

def audit(root, out):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for group in ['group1A','group2A']:
        for p in sorted((root/group/'kinect').glob('*.txt')):
            m=re.fullmatch(r'(G[12]A)-Kinect-(CTK|ELK|RTK)-([^-]+)-([^-]+)-(\d+)',p.stem)
            if not m: raise ValueError(p)
            g, exercise, subject, center, number=m.groups()
            sid=f'{g}-{exercise}-{subject}-{center}-{number}'
            ann=root/group/'annotatorA'/f'{sid}.anvil'
            video=root/group/'videos'/f'{g}-Anon-{exercise}-{subject}-{center}-{number}.mp4'
            label,events=annotation(ann)
            xyz=np.loadtxt(p,dtype=np.float32)
            cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS)
            nf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            reasons=[]
            if xyz.ndim!=2 or xyz.shape[1]!=175 or not np.isfinite(xyz).all():reasons.append('invalid_skeleton')
            if nf!=len(xyz) or fps<=0:reasons.append('video_skeleton_length_mismatch')
            if label not in ['Correct','Incorrect']:reasons.append('excluded_global_label')
            if label=='Incorrect' and len(events)!=1:reasons.append('incorrect_without_single_event')
            if label=='Correct' and events:reasons.append('correct_with_event')
            for e in events:
                if e['part'] is None or not 0<=e['start_s']<e['end_s']<=nf/max(fps,1)+1/fps:
                    reasons.append('invalid_event_bounds_or_part')
            # Groups 1A/2A are disjoint patient/healthy cohorts per source documentation.
            # Same S1 token occurs in two centers. Merge it conservatively to prevent
            # potential cross-center leakage; individual identity crosswalk is unavailable.
            uid=f'{"patient" if group=="group1A" else "healthy"}:{subject}'
            rows.append(dict(sample_id=sid,subject_uid=uid,group=group,exercise=exercise,
                skeleton_path=str(p),video_path=str(video),annotation_path=str(ann),annotator='A',
                global_label=label,events=events,skeleton_frames=len(xyz),video_frames=nf,
                video_fps=fps,timing_verified=nf==len(xyz) and fps>0,
                timing_scope='frame-count correspondence; no hardware timestamp available',
                usable_for_localization=not reasons,exclusion_reasons=reasons,
                skeleton_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                annotation_sha256=hashlib.sha256(ann.read_bytes()).hexdigest()))
    hashes=[r['skeleton_sha256'] for r in rows]
    duplicates={h:sum(x==h for x in hashes) for h in set(hashes) if hashes.count(h)>1}
    if duplicates:raise ValueError(f'Duplicate recordings: {duplicates}')
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    valid=[r for r in rows if r['usable_for_localization']]
    subjects=sorted(set(r['subject_uid'] for r in valid))
    split=[]
    for fold in range(3):
        roles={'train':[],'val':[],'test':[]}
        for cohort in ['patient','healthy']:
            ss=[s for s in subjects if s.startswith(cohort+':')]
            if len(ss)!=6:raise ValueError(f'Expected six subjects, found {ss}')
            test=ss[fold*2:fold*2+2]; others=[s for s in ss if s not in test]
            val=[others[fold%len(others)]]
            roles['test']+=test; roles['val']+=val; roles['train'] += [s for s in others if s not in val]
        split.append(dict(fold=fold,subjects=roles,recordings={k:[r['sample_id'] for r in valid if r['subject_uid'] in v] for k,v in roles.items()}))
    summary=dict(dataset_root=str(root),indexed=len(rows),usable=len(valid),
        global_labels=dict(collections.Counter(r['global_label'] for r in rows)),
        usable_labels=dict(collections.Counter(r['global_label'] for r in valid)),
        parts=dict(collections.Counter(e['part'] for r in valid for e in r['events'])),
        subjects={s:dict(collections.Counter(r['global_label'] for r in valid if r['subject_uid']==s)) for s in subjects},
        excluded=[{'sample_id':r['sample_id'],'reasons':r['exclusion_reasons'],'video_frames':r['video_frames'],'skeleton_frames':r['skeleton_frames']} for r in rows if not r['usable_for_localization']],
        groups_available={g:len(list((root/g/'kinect').glob('*.txt'))) for g in ['group1A','group1B','group2A','group2B','group3']},
        fps=sorted(set(r['video_fps'] for r in rows)),
        identity_source='KeraalDataset/doc/readme_files_format.txt; cohort+subject token; S1 centers merged conservatively; no individual-level crosswalk supplied',
        clinical_coverage='annotated principal event only; outside intervals is not claimed clinically normal')
    for name,obj in [('audit.json',summary),('splits.json',split)]:
        (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();audit(a.root,a.out)
