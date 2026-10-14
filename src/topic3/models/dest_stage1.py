"""Stage 1: DeST spatial/temporal encoder -> principal error interval.

No body-part labels are used for training or checkpoint selection in this stage.
No held-out test skeletons are loaded. Encoder source stays unmodified.
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
from numba import njit
from torch import nn
from torch.nn import functional as F

from topic3.paths import ROOT as HERE, MANIFESTS as ART, DATASET as DATA, UPSTREAM_DEST as SOURCE
sys.path.insert(0, str(SOURCE))
from libs.models.SP import MultiScale_GraphConv
from libs.models.DeST_linearformer import STI

# Paths are configured in topic3/configs/paths.json.
GATE = dict(f1_03=.60, f1_05=.45, recall_03=.70, normal_false_alarm_rate=.20)

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_metadata(fold,timebase='encoded'):
    split = json.loads((ART/'splits.json').read_text(encoding='utf8'))[fold]
    allowed = set(split['recordings']['train'] + split['recordings']['val'])
    records = [json.loads(line) for line in (ART/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    records = {r['sample_id']:r for r in records if r['sample_id'] in allowed and r['usable_for_localization']}
    tr, va = [records[s] for s in split['recordings']['train']], [records[s] for s in split['recordings']['val']]
    assert not {r['subject_uid'] for r in tr} & {r['subject_uid'] for r in va}
    assert not set(allowed) & set(split['recordings']['test'])
    for r in tr+va:
        assert len(r['events']) <= 1, 'Principal-event head cannot silently drop extra annotations'
        assert Path(r['skeleton_path']).resolve().is_relative_to(DATA.resolve())
        assert Path(r['annotation_path']).resolve().is_relative_to(DATA.resolve())
        assert digest(r['annotation_path'])==r['annotation_sha256'], 'Annotation changed since frozen manifest'
        assert bool(r['events']) == (r['global_label'] == 'Incorrect')
        if timebase=='annotation15':
            bpath=Path(r['annotation_path']).parent.parent/'annotatorB'/Path(r['annotation_path']).name
            els=ET.parse(bpath).getroot().findall(".//track[@name='Global evaluation']/el")
            duration=max(float(el.attrib['end']) for el in els)
            assert abs(duration-r['skeleton_frames']/15)<=.011, 'Annotation timebase evidence inconsistent'
            r['timebase_fps']=15.
            r['timebase_evidence']=dict(path=str(bpath),sha256=digest(bpath),global_end_s=duration)
            assert all(e['end_s']<=r['skeleton_frames']/15+1/15 for e in r['events']), 'Error outside annotation clock duration'
        else:assert timebase=='encoded'
    return tr, va, split

def features(record, length=256):
    path = Path(record['skeleton_path'])
    assert digest(path) == record['skeleton_sha256'], 'Data changed since manifest'
    raw = np.loadtxt(path, dtype=np.float32)
    assert len(raw) == record['skeleton_frames'] and np.isfinite(raw).all()
    p = median_filter(raw.reshape(len(raw),25,7)[...,:3], size=(3,1,1), mode='nearest')
    scale = np.median(np.linalg.norm(p[:,20]-p[:,0], axis=-1))
    assert scale > .05
    hip = np.median(p[:min(15,len(p)),16]-p[:min(15,len(p)),12],axis=0)
    yaw = np.arctan2(hip[2],hip[0]); c,s = np.cos(yaw),np.sin(yaw)
    rot = np.array([[c,0,-s],[0,1,0],[s,0,c]],dtype=np.float32)
    q = (p-p[:,:1]) @ rot / scale
    clock_fps=record.get('timebase_fps',record['video_fps'])
    velocity = np.gradient(q, 1/clock_fps, axis=0)
    root = (p[:,:1]-p[:1,:1]) @ rot / scale
    root = np.broadcast_to(root, q.shape)
    # Root movement is retained; rotating each frame would erase clinical trunk errors.
    f = np.concatenate([q,velocity,root],axis=-1).reshape(len(p),-1)
    grid = np.linspace(0,len(p)-1,length)
    f = np.stack([np.interp(grid,np.arange(len(p)),col) for col in f.T],axis=-1)
    x = torch.tensor(f.reshape(length,25,9).transpose(2,1,0),dtype=torch.float32)
    end_time = (len(p)-1)/clock_fps
    target = torch.zeros(2)
    if record['events']:
        e = record['events'][0]
        assert 0 <= e['start_s'] < e['end_s'] <= len(p)/clock_fps+1/clock_fps
        target = torch.tensor([e['start_s'],e['end_s']])/end_time
        target = target.clamp(0,1)
    return x, float(bool(record['events'])), target, end_time

@njit(cache=True)
def band_path(cost,band):
    n,m=cost.shape
    dp=np.full((n+1,m+1),np.inf)
    dp[0,0]=0
    for i in range(1,n+1):
        for j in range(max(1,i-band),min(m,i+band)+1):
            dp[i,j]=cost[i-1,j-1]+min(dp[i-1,j-1],dp[i-1,j],dp[i,j-1])
    qi=[];ri=[];i=n;j=m
    while i>0 and j>0:
        qi.append(i-1);ri.append(j-1)
        diagonal=dp[i-1,j-1];up=dp[i-1,j];left=dp[i,j-1]
        if diagonal<=up and diagonal<=left:i-=1;j-=1
        elif up<=left:i-=1
        else:j-=1
    return qi,ri

def reference_cache(train,records,cache):
    refs={};output={}; joints=[1,20,2,4,5,6,8,9,10,12,16]
    for r in records:
        candidates=[c for c in train if c['exercise']==r['exercise'] and not c['events'] and c['subject_uid']!=r['subject_uid']]
        assert candidates, 'No subject-disjoint correct training reference'
        median_duration=np.median([c['skeleton_frames'] for c in candidates])
        candidates.sort(key=lambda c:(not c['subject_uid'].startswith('healthy:'),abs(c['skeleton_frames']-median_duration),c['sample_id']))
        chosen=[];subjects=set()
        for c in candidates:
            if c['subject_uid'] not in subjects:
                chosen.append(c);subjects.add(c['subject_uid'])
            if len(chosen)==3:break
        query=cache[r['sample_id']][0]
        q=query[:3].numpy().transpose(2,1,0)
        aligned=[]
        for c in chosen:
            ref=cache[c['sample_id']][0][:3].numpy().transpose(2,1,0)
            costs=cdist(q[:,joints].reshape(len(q),-1),ref[:,joints].reshape(len(ref),-1),'sqeuclidean')
            qi,ri=band_path(costs,max(1,int(.2*len(q))))
            a=np.zeros_like(q);count=np.zeros(len(q))
            for i,j in zip(qi,ri):a[i]+=ref[j];count[i]+=1
            assert (count>0).all()
            aligned.append(a/count[:,None,None])
        reference=np.median(aligned,axis=0)
        delta=q-reference
        extra=torch.tensor(np.concatenate([reference,delta,np.abs(delta)],axis=-1).transpose(2,1,0),dtype=torch.float32)
        output[r['sample_id']]=(torch.cat([query,extra],dim=0),*cache[r['sample_id']][1:])
        refs[r['sample_id']]=[c['sample_id'] for c in chosen]
    return output,refs

class TemporalModel(nn.Module):
    def __init__(self,width=64,layers=10,in_channels=9,norm='batch',local_error=False):
        super().__init__()
        self.spatial = MultiScale_GraphConv(13,in_channels,width,'PKU-subject')
        if norm=='group':
            for index,layer in enumerate(self.spatial.mlp.layers):
                if isinstance(layer,nn.BatchNorm2d):
                    self.spatial.mlp.layers[index]=nn.GroupNorm(8,width)
        self.temporal = STI(25,width,width,width,layers,list(range(1,layers)))
        self.boundary = nn.Conv1d(width,2,1)
        self.local_error=local_error
        if local_error:
            self.error_score=nn.Conv1d(width,1,1)
        else:
            self.pool_score = nn.Conv1d(width,1,1)
            self.presence = nn.Sequential(nn.Linear(width*2,width),nn.ReLU(),nn.Dropout(.2),nn.Linear(width,1))

    def forward(self,x,return_features=False):
        mask = torch.ones(len(x),1,x.shape[-1],device=x.device,dtype=x.dtype)
        spatial = self.spatial(x)
        temporal = self.temporal(spatial,mask)
        if self.local_error:
            local=self.error_score(temporal).squeeze(1)
            # Smooth max, normalized so a constant frame logit remains unchanged.
            presence=.5*(torch.logsumexp(local/.5,dim=-1)-math.log(local.shape[-1]))
            output=dict(presence=presence,boundary=self.boundary(temporal),local_error=local)
        else:
            weights = self.pool_score(temporal).softmax(-1)
            pooled = torch.cat([(temporal*weights).sum(-1),temporal.amax(-1)],dim=1)
            output = dict(presence=self.presence(pooled).squeeze(-1),boundary=self.boundary(temporal))
        if return_features:
            output.update(spatial=spatial,temporal=temporal)
        return output

def native_representation(cache,paired):
    output={}
    for sid,item in cache.items():
        x=item[0]
        # Native PKU order: person1(displacement XYZ, relative XYZ), person2(...).
        q=x[:3]*.45
        absolute=(x[:3]+x[6:9])*.45
        disp=torch.zeros_like(q);disp[...,:-1]=absolute[...,1:]-absolute[...,:-1]
        if paired:
            r=x[9:12]*.45
            rd=torch.zeros_like(r);rd[...,:-1]=r[...,1:]-r[...,:-1]
        else:r=torch.zeros_like(q);rd=torch.zeros_like(q)
        output[sid]=(torch.cat([disp,q,rd,r],dim=0),*item[1:])
    return output

def initialize_pretrained(model,path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    mapped={}
    for key,value in state.items():
        if key.startswith('SP.'):mapped['spatial.'+key[3:]]=value
        elif key.startswith('STI.'):mapped['temporal.'+key[4:]]=value
    expected={k:v for k,v in model.state_dict().items() if k.startswith(('spatial.','temporal.'))}
    assert set(mapped)==set(expected), 'Pretrained encoder keys must match exactly'
    assert all(mapped[k].shape==expected[k].shape for k in mapped), 'No partial/ambiguous channel transfer'
    missing,unexpected=model.load_state_dict(mapped,strict=False)
    assert not unexpected and all(not k.startswith(('spatial.','temporal.')) for k in missing)
    return dict(path=str(Path(path).resolve()),sha256=digest(path),loaded_encoder_tensors=len(mapped),
                loaded_encoder_parameters=sum(v.numel() for k,v in mapped.items() if k in dict(model.named_parameters())),
                skipped_source_tensors=len(state)-len(mapped),
                source='official DeST PKU-subject linearformer split1 public model zoo')

def known_frame_targets(present,target,length):
    grid=torch.linspace(0,1,length,device=target.device)
    positive=present>.5
    inside=(grid[None,:]>=target[:,0,None]) & (grid[None,:]<=target[:,1,None]) & positive[:,None]
    missing=positive & ~inside.any(-1)
    if missing.any():
        closest=(grid[None,:]-target[missing].mean(-1,keepdim=True)).abs().argmin(-1)
        inside[missing]=F.one_hot(closest,length).bool()
    known=(~positive[:,None]).expand(-1,length) | inside
    labels=positive[:,None].expand(-1,length).float()
    return labels,known


def objective(out,present,target,pos_weight,presence_weight=None,boundary_weight=None):
    per_record=F.binary_cross_entropy_with_logits(out['presence'],present,pos_weight=pos_weight,reduction='none')
    existence=(per_record if presence_weight is None else per_record*presence_weight).mean()
    pos = present > .5
    if pos.any():
        logits = out['boundary'][pos]
        t = target[pos]
        grid = torch.linspace(0,1,logits.shape[-1],device=logits.device)
        soft = torch.exp(-.5*((grid[None,None,:]-t[:,:,None])/.02)**2)
        soft = soft/soft.sum(-1,keepdim=True)
        loc_each = -(soft*logits.log_softmax(-1)).sum(-1).mean(-1)
        expectation = (logits.softmax(-1)*grid).sum(-1)
        regression_each = F.smooth_l1_loss(expectation,t,beta=.05,reduction='none').mean(-1)
        order_each = F.relu(expectation[:,0]-expectation[:,1]+1/logits.shape[-1])
        weights=torch.ones_like(loc_each) if boundary_weight is None else boundary_weight[pos]
        loc=(loc_each*weights).mean();regression=(regression_each*weights).mean();order=(order_each*weights).mean()
    else:
        loc = out['boundary'].sum()*0
        regression = order = loc
    local_loss=existence*0
    if 'local_error' in out:
        labels,known=known_frame_targets(present,target,out['local_error'].shape[-1])
        frame_loss=F.binary_cross_entropy_with_logits(out['local_error'],labels,pos_weight=pos_weight,reduction='none')
        record_loss=(frame_loss*known).sum(-1)/known.sum(-1).clamp_min(1)
        local_loss=(record_loss if presence_weight is None else record_loss*presence_weight).mean()
    loss = existence + loc + regression + .1*order + .5*local_loss
    return loss,dict(presence=float(existence.detach()),boundary=float(loc.detach()),regression=float(regression.detach()),local=float(local_loss.detach()))

def decode_boundaries(logits):
    # Joint constrained MAP: select start < end rather than sort two wrong peaks.
    a,b = logits.log_softmax(-1).unbind(1)
    length = a.shape[-1]
    scores = a[:,:,None]+b[:,None,:]
    valid = torch.triu(torch.ones(length,length,device=a.device,dtype=torch.bool),diagonal=1)
    idx = scores.masked_fill(~valid,-torch.inf).flatten(1).argmax(-1)
    return torch.stack([idx//length,idx%length],dim=-1)/(length-1)

@torch.no_grad()
def infer(model,records,cache,device,batch_size=8):
    model.eval(); raw={}
    for start in range(0,len(records),batch_size):
        group=records[start:start+batch_size]
        x=torch.stack([cache[r['sample_id']][0] for r in group]).to(device)
        out=model(x)
        presence=out['presence'].sigmoid().cpu().tolist()
        intervals=decode_boundaries(out['boundary']).cpu().numpy()
        for r,score,interval in zip(group,presence,intervals):
            endtime=cache[r['sample_id']][3]
            raw[r['sample_id']]=dict(presence=score,start_s=float(interval[0]*endtime),end_s=float(interval[1]*endtime))
        if 'local_error' in out:
            for r,local in zip(group,out['local_error'].sigmoid().cpu().tolist()):
                raw[r['sample_id']]['local_error_scores']=local
    return raw

def iou(a,b):
    inter=max(0.,min(a['end_s'],b['end_s'])-max(a['start_s'],b['start_s']))
    union=(a['end_s']-a['start_s'])+(b['end_s']-b['start_s'])-inter
    return inter/max(union,1e-9)

def metrics(raw,records,threshold):
    normal=[r for r in records if not r['events']]
    positive=[r for r in records if r['events']]
    output=dict(records=len(records),positive_records=len(positive),normal_records=len(normal),threshold=threshold)
    output['normal_false_alarm_rate']=sum(raw[r['sample_id']]['presence']>=threshold for r in normal)/max(len(normal),1)
    oracle_ious=[iou(raw[r['sample_id']],r['events'][0]) for r in positive]
    output['oracle_interval_recall_03']=sum(x>=.3 for x in oracle_ious)/max(len(positive),1)
    output['oracle_interval_recall_05']=sum(x>=.5 for x in oracle_ious)/max(len(positive),1)
    output['oracle_mean_iou']=float(np.mean(oracle_ious)) if oracle_ious else None
    output['boundary_mae_all_errors_s']=[float(np.mean([abs(raw[r['sample_id']][k]-r['events'][0][k]) for r in positive])) for k in ['start_s','end_s']] if positive else None
    for tiou in [.3,.5]:
        tp=fp=fn=0
        for r in records:
            p=raw[r['sample_id']]
            emitted=p['presence']>=threshold
            match=emitted and bool(r['events']) and iou(p,r['events'][0])>=tiou
            tp+=int(match);fp+=int(emitted and not match);fn+=int(bool(r['events']) and not match)
        output[f'temporal@{tiou}']=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1))
    return output

def calibrate(raw,records):
    candidates=[]
    for th in np.arange(.10,.951,.05):
        m=metrics(raw,records,float(round(th,2)))
        candidates.append(m)
    feasible=[m for m in candidates if m['normal_false_alarm_rate']<=GATE['normal_false_alarm_rate']]
    selected=max(feasible or candidates,key=lambda m:(m['temporal@0.3']['f1'],m['temporal@0.5']['f1'],-m['normal_false_alarm_rate'],m['threshold']))
    return selected

def gate(m):
    checks=dict(f1_03=m['temporal@0.3']['f1']>=GATE['f1_03'],
                f1_05=m['temporal@0.5']['f1']>=GATE['f1_05'],
                recall_03=m['temporal@0.3']['recall']>=GATE['recall_03'],
                normal_false_alarm=m['normal_false_alarm_rate']<=GATE['normal_false_alarm_rate'])
    return dict(pass_gate=all(checks.values()),checks=checks)

def median_prior(train,raw,records):
    # Same learned error-presence scores, train-only per-exercise median intervals.
    priors={}
    for ex in ['CTK','ELK','RTK']:
        events=[(r['events'][0]['start_s']/((r['skeleton_frames']-1)/r.get('timebase_fps',r['video_fps'])),r['events'][0]['end_s']/((r['skeleton_frames']-1)/r.get('timebase_fps',r['video_fps']))) for r in train if r['exercise']==ex and r['events']]
        if not events:
            events=[(r['events'][0]['start_s']/((r['skeleton_frames']-1)/r.get('timebase_fps',r['video_fps'])),r['events'][0]['end_s']/((r['skeleton_frames']-1)/r.get('timebase_fps',r['video_fps']))) for r in train if r['events']]
        priors[ex]=np.median(events,axis=0).tolist()
    output={}
    for r in records:
        a,b=priors[r['exercise']]; duration=(r['skeleton_frames']-1)/r.get('timebase_fps',r['video_fps'])
        output[r['sample_id']]=dict(presence=raw[r['sample_id']]['presence'],start_s=a*duration,end_s=b*duration)
    return output,priors

def run(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
    train,val,split=read_metadata(args.fold,args.timebase)
    presence_weights={r['sample_id']:1. for r in train}
    boundary_weights=dict(presence_weights)
    if args.balance=='exercise_label':
        cells={(ex,label):sum(r['exercise']==ex and bool(r['events'])==label for r in train)
               for ex in ['CTK','ELK','RTK'] for label in [False,True]}
        assert min(cells.values())>0, 'All exercise/label cells must exist in train'
        npos_train=sum(bool(r['events']) for r in train)
        for r in train:
            presence_weights[r['sample_id']]=len(train)/(6*cells[r['exercise'],bool(r['events'])])
            boundary_weights[r['sample_id']]=npos_train/(3*cells[r['exercise'],True])
    cache={r['sample_id']:features(r,args.length) for r in train+val}
    reference_ids={}
    if args.reference=='band_dtw':
        cache,reference_ids=reference_cache(train,train+val,cache)
    if args.representation!='extended':
        if args.representation=='pku_pair':assert args.reference=='band_dtw'
        else:assert args.reference=='none'
        cache=native_representation(cache,args.representation=='pku_pair')
    runroot=HERE/'runs'/args.run/f'fold{args.fold}_seed{args.seed}'
    runroot.mkdir(parents=True,exist_ok=False)
    codehash=digest(__file__)
    protocol=dict(stage=1,scope='principal annotated event, not exhaustive multiple-error localization',
                  architecture='official DeST SP + STI encoder; adapted presence and directional boundary heads',
                  source_commit='319e1f10f310e08a198fba3a32a4d82416ecb15f',
                  source_sha256={str(p.relative_to(SOURCE)):digest(p) for p in (SOURCE/'libs/models').rglob('*.py')},
                  args=vars(args),gate=GATE,stage2_locked_until_gate=True,
                  manifest_sha256=digest(ART/'manifest.jsonl'),splits_sha256=digest(ART/'splits.json'),code_sha256=codehash,
                  train_ids=[r['sample_id'] for r in train],val_ids=[r['sample_id'] for r in val],
                  train_subjects=split['subjects']['train'],val_subjects=split['subjects']['val'],
                  data_root=str(DATA),test_loaded=False,torch=torch.__version__,
                  device_name=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
                  length=args.length,features='9 channels: root-relative XYZ, velocity, root displacement; fixed heading',
                  reference=args.reference,reference_ids=reference_ids,input_channels=next(iter(cache.values()))[0].shape[0],
                  normalization=args.norm,balance=args.balance,local_error=args.local_error,
                  training_presence_weights=presence_weights,training_boundary_weights=boundary_weights,
                  representation=args.representation,pretrained=args.pretrained,
                  timebase=args.timebase,timebase_evidence={r['sample_id']:r['timebase_evidence'] for r in train+val if 'timebase_evidence' in r},
                  selection='validation F1@.3 then F1@.5, subject to normal FAR<=.2 when feasible',
                  annotation_policy='boundary/presence supervision only; no dense negative labels outside annotated interval',
                  limitation='historical test results have already been viewed; copied splits are not a newly untouched test set')
    save(runroot/'protocol.json',protocol)
    (runroot/'stage1_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
    model=TemporalModel(args.width,args.layers,next(iter(cache.values()))[0].shape[0],args.norm,args.local_error).to(device)
    if args.pretrained:
        assert args.norm=='batch' and args.representation!='extended'
        loading=initialize_pretrained(model,args.pretrained)
        save(runroot/'pretrained_loading.json',loading)
    backbone=list(model.spatial.parameters())+list(model.temporal.parameters())
    backbone_ids={id(p) for p in backbone}
    head=[p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([dict(params=backbone,lr=args.lr*(.1 if args.pretrained else 1)),dict(params=head,lr=args.lr)],weight_decay=.01)
    npos=sum(bool(r['events']) for r in train)
    pw=torch.tensor(1. if args.balance=='exercise_label' else (len(train)-npos)/max(npos,1),device=device)
    bestkey=None;history=[]; started=time.perf_counter()
    save(runroot/'status.json',dict(status='running',epoch=0))
    for epoch in range(1,args.epochs+1):
        model.train();order=list(train);random.shuffle(order);total=[];components=[]
        if args.pretrained:
            frozen=epoch<=args.freeze_epochs
            for p in backbone:p.requires_grad_(not frozen)
            if frozen:model.spatial.eval();model.temporal.eval()
        for index in range(0,len(order),args.batch_size):
            group=order[index:index+args.batch_size]
            x=torch.stack([cache[r['sample_id']][0] for r in group]).to(device)
            present=torch.tensor([cache[r['sample_id']][1] for r in group],device=device)
            target=torch.stack([cache[r['sample_id']][2] for r in group]).to(device)
            optimizer.zero_grad(set_to_none=True)
            pweights=torch.tensor([presence_weights[r['sample_id']] for r in group],device=device) if args.balance!='none' else None
            bweights=torch.tensor([boundary_weights[r['sample_id']] for r in group],device=device) if args.balance!='none' else None
            out=model(x);loss,details=objective(out,present,target,pw,pweights,bweights)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('nonfinite gradients')
            optimizer.step();total.append(float(loss.detach()));components.append(details)
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
            raw=infer(model,val,cache,device,args.batch_size)
            save(runroot/'validation_epochs'/f'epoch{epoch:03}.json',raw)
            m=calibrate(raw,val)
            item=dict(epoch=epoch,loss=float(np.mean(total)),components={k:float(np.mean([d[k] for d in components])) for k in components[0]},validation=m,elapsed_seconds=time.perf_counter()-started)
            history.append(item);save(runroot/'history.json',history)
            feasible=m['normal_false_alarm_rate']<=.20
            key=(int(feasible),m['temporal@0.3']['f1'],m['temporal@0.5']['f1'],-m['normal_false_alarm_rate'])
            if bestkey is None or key>bestkey:
                bestkey=key
                torch.save(dict(state=model.state_dict(),epoch=epoch,threshold=m['threshold'],args=vars(args),code_sha256=codehash),runroot/'best.pt')
                save(runroot/'validation_predictions.json',raw)
                save(runroot/'selected.json',item)
            save(runroot/'status.json',dict(status='running',epoch=epoch,elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(dict(fold=args.fold,seed=args.seed,epoch=epoch,loss=item['loss'],f1_03=m['temporal@0.3']['f1'],f1_05=m['temporal@0.5']['f1'],recall_03=m['temporal@0.3']['recall'],far=m['normal_false_alarm_rate'],elapsed_seconds=item['elapsed_seconds'])),flush=True)
    # Preserve the trained endpoint even when validation selects the initial model.
    # This is for overfitting/calibration diagnosis, never for replacing selection.
    torch.save(dict(state=model.state_dict(),epoch=args.epochs,threshold=m['threshold'],args=vars(args),code_sha256=codehash),runroot/'final.pt')
    selected=json.loads((runroot/'selected.json').read_text(encoding='utf8'))
    raw=json.loads((runroot/'validation_predictions.json').read_text(encoding='utf8'))
    prior,priors=median_prior(train,raw,val)
    result=dict(fold=args.fold,seed=args.seed,parameters=sum(p.numel() for p in model.parameters()),
                selected=selected,gate=gate(selected['validation']),
                by_exercise={ex:metrics(raw,[r for r in val if r['exercise']==ex],selected['validation']['threshold']) for ex in ['CTK','ELK','RTK']},
                by_cohort={co:metrics(raw,[r for r in val if r['subject_uid'].startswith(co+':')],selected['validation']['threshold']) for co in ['patient','healthy']},
                median_time_prior_same_presence=metrics(prior,val,selected['validation']['threshold']),
                train_time_priors=priors,elapsed_seconds=time.perf_counter()-started,
                test_loaded=False,stage2_started=False)
    save(runroot/'result.json',result)
    save(runroot/'status.json',dict(status='completed',epoch=args.epochs,gate=result['gate'],elapsed_seconds=result['elapsed_seconds']))
    print(json.dumps(dict(completed=str(runroot),result=result),ensure_ascii=False),flush=True)

def selfcheck():
    assert iou(dict(start_s=0,end_s=1),dict(start_s=2,end_s=3))==0
    gt=dict(start_s=2,end_s=4)
    recs=[dict(sample_id='p',events=[gt]),dict(sample_id='n',events=[])]
    p=dict(presence=1,**gt);n=dict(presence=0,start_s=0,end_s=1)
    m=metrics(dict(p=p,n=n),recs,.5)
    assert m['temporal@0.5']['f1']==1 and m['normal_false_alarm_rate']==0
    wrong=dict(presence=1,start_s=5,end_s=6)
    m=metrics(dict(p=wrong,n=n),recs,.5)
    assert m['temporal@0.5']['tp']==0 and m['temporal@0.5']['fp']==m['temporal@0.5']['fn']==1
    torch.manual_seed(1);torch.set_num_threads(4)
    model=TemporalModel(32,4)
    x=torch.randn(2,9,25,64)
    out=model(x,True)
    loss,_=objective(out,torch.zeros(2),torch.zeros(2,2),torch.tensor(1.))
    loss.backward()
    assert torch.isfinite(loss) and out['spatial'].shape==torch.Size([2,32,64,25])
    intervals=decode_boundaries(out['boundary'])
    assert (intervals[:,0]<intervals[:,1]).all()
    print('Selfcheck passed: disjoint IoU, wrong-event FP+FN, all-normal gradient, T x J features, ordered boundaries')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,default=0);ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--run',default='stage1_v1');ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--width',type=int,default=64);ap.add_argument('--layers',type=int,default=10)
    ap.add_argument('--length',type=int,default=256);ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--eval-every',type=int,default=5)
    ap.add_argument('--reference',choices=['none','band_dtw'],default='none')
    ap.add_argument('--norm',choices=['batch','group'],default='batch')
    ap.add_argument('--balance',choices=['none','exercise_label'],default='none')
    ap.add_argument('--local-error',action='store_true')
    ap.add_argument('--timebase',choices=['encoded','annotation15'],default='annotation15')
    ap.add_argument('--representation',choices=['extended','pku_single','pku_pair'],default='extended')
    ap.add_argument('--pretrained',default=None);ap.add_argument('--freeze-epochs',type=int,default=10)
    ap.add_argument('--selfcheck',action='store_true')
    args=ap.parse_args()
    if args.selfcheck:selfcheck()
    else:run(args)
