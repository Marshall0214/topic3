"""Reproducible subject-disjoint principal-event localization benchmark.

The supervised temporal heads predict the two boundaries of the annotated
principal event, rather than treating all unannotated frames as clinical normals.
All model selection and decoding calibration use validation recordings only.
"""
import argparse, collections, hashlib, json, os, random, time, subprocess
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
import torch
from torch import nn
from torch.nn import functional as F
try:
    from numba import njit
except ImportError:
    def njit(*args,**kwargs):return lambda f:f

PARTS=['Arms','Trunk']; EXERCISES=['CTK','ELK','RTK']
METHODS=['B0_no_reference','B2_linear','B2_dtw','A1_band_dtw','P_multi_band_dtw']

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')

def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-6)

def features(xyz,fps):
    """25-joint Kinect coordinates; one fixed heading for the whole recording."""
    p=np.asarray(xyz,dtype=np.float32).copy()
    if p.ndim==2:p=p.reshape(-1,25,7)[...,:3]
    if not np.isfinite(p).all():raise ValueError('Nonfinite skeleton')
    # No framewise orientation cancellation: preserve trunk rotation and side bend.
    p=median_filter(p,size=(3,1,1),mode='nearest')
    scale=np.median(np.linalg.norm(p[:,20]-p[:,0],axis=1))
    if scale<.05:raise ValueError('Degenerate torso length')
    hip=np.median(p[:min(15,len(p)),16]-p[:min(15,len(p)),12],axis=0)
    yaw=np.arctan2(hip[2],hip[0]);c,s=np.cos(yaw),np.sin(yaw)
    rot=np.array([[c,0,-s],[0,1,0],[s,0,c]],dtype=np.float32)
    q=((p-p[:,0:1])@rot)/scale
    arm=q[:,[4,5,6,8,9,10]].reshape(len(p),18)
    angles=[]
    for sh,el,wr in [(4,5,6),(8,9,10)]:
        angles.append(np.sum(unit(q[:,sh]-q[:,el])*unit(q[:,wr]-q[:,el]),axis=1))
    torso=unit(q[:,20]-q[:,0]);shoulder=unit(q[:,8]-q[:,4]);pelvis=unit(q[:,16]-q[:,12])
    twist=np.arctan2(shoulder[:,2],shoulder[:,0])-np.arctan2(pelvis[:,2],pelvis[:,0])
    trunk=np.c_[q[:,1],q[:,20],torso[:,0],torso[:,2],np.sin(twist),np.cos(twist)]
    position=np.c_[arm,np.asarray(angles).T,trunk].astype(np.float32)
    velocity=np.gradient(position,1/fps,axis=0)
    return np.c_[position,velocity].astype(np.float32)

@njit(cache=True)
def _path(cost,band):
    n,m=cost.shape;acc=np.full((n+1,m+1),np.inf);acc[0,0]=0.
    back=np.zeros((n,m),np.int8)
    for i in range(n):
        for j in range(m):
            if band>0 and abs(i/max(n-1,1)-j/max(m-1,1))>band:continue
            a,b,c=acc[i,j],acc[i,j+1],acc[i+1,j]
            k=0
            if b<a:a=b;k=1
            if c<a:a=c;k=2
            acc[i+1,j+1]=cost[i,j]+a;back[i,j]=k
    pi=np.empty(n+m,np.int64);pj=np.empty(n+m,np.int64)
    i=n-1;j=m-1;k=0
    while i>=0 and j>=0:
        pi[k]=i;pj[k]=j;k+=1
        b=back[i,j]
        if b==0:i-=1;j-=1
        elif b==1:i-=1
        else:j-=1
    return pi[:k][::-1],pj[:k][::-1],acc[n,m]

def align(query,reference,mode='dtw'):
    if mode=='linear':
        return np.stack([np.interp(np.linspace(0,1,len(query)),np.linspace(0,1,len(reference)),reference[:,j]) for j in range(query.shape[1])],axis=1).astype(np.float32)
    # Alignment uses position channels; velocity still available to prediction heads.
    width=query.shape[1]//2
    cost=cdist(query[:,:width],reference[:,:width],metric='sqeuclidean')/width
    pi,pj,total=_path(cost,.20 if mode=='band' else 0.)
    if not np.isfinite(total):raise ValueError('Unreachable DTW path')
    mapped=np.zeros_like(query);counts=np.bincount(pi,minlength=len(query))
    np.add.at(mapped,pi,reference[pj])
    if np.any(counts==0):raise ValueError('DTW did not cover query timeline')
    return mapped/counts[:,None]

def event_iou(a,b):
    intersection=max(0.,min(a['end_s'],b['end_s'])-max(a['start_s'],b['start_s']))
    return intersection/max(a['end_s']-a['start_s']+b['end_s']-b['start_s']-intersection,1e-9)

def metrics(predictions,records,cohort=None):
    selected=[r for r in records if cohort is None or r['subject_uid'].startswith(cohort+':')]
    output={'records':len(selected),'positive_records':sum(bool(r['events']) for r in selected)}
    for threshold in [.3,.5]:
        for joint in [False,True]:
            tp=fp=fn=0;errors=[];part_tp=collections.Counter();part_gt=collections.Counter()
            for r in selected:
                gt=r['events'][0] if r['events'] else None;p=predictions[r['sample_id']]
                matched=p is not None and gt is not None and event_iou(p,gt)>=threshold and (not joint or p['part']==gt['part'])
                tp+=int(matched);fp+=int(p is not None and not matched);fn+=int(gt is not None and not matched)
                if gt:part_gt[gt['part']]+=1
                if matched:
                    part_tp[gt['part']]+=1
                    errors.append([abs(p['start_s']-gt['start_s']),abs(p['end_s']-gt['end_s'])])
            prefix=f'{"joint" if joint else "temporal"}@{threshold}'
            output[prefix]=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
                f1=2*tp/max(2*tp+fp+fn,1),boundary_mae_s=np.mean(errors,axis=0).tolist() if errors else None,
                matched_coverage=tp/max(tp+fn,1),part_recall={p:part_tp[p]/part_gt[p] if part_gt[p] else None for p in PARTS})
    normal=[r for r in selected if not r['events']]
    output['normal_records']=len(normal)
    output['normal_false_alarm_rate']=sum(predictions[r['sample_id']] is not None for r in normal)/len(normal) if normal else None
    return output

def calibrate(raw,records,decode):
    best=None
    for threshold in [.2,.3,.4,.5,.6,.7,.8]:
        preds={r['sample_id']:decode(raw[r['sample_id']],r,threshold) for r in records}
        m=metrics(preds,records)
        # Ties favor fewer false alarms, then a stricter threshold.
        key=(m['joint@0.5']['f1'],m['joint@0.3']['f1'],-(m['normal_false_alarm_rate'] or 0),threshold)
        if best is None or key>best[0]:best=(key,threshold,m)
    return best[1],best[2]

def neural_decode(raw,r,threshold):
    if raw['presence']<threshold:return None
    duration=r['skeleton_frames']/r['video_fps']; a,b=sorted(raw['boundaries'])
    a=float(np.clip(a*duration,0,duration-1/r['video_fps']))
    b=float(np.clip(b*duration,a+1/r['video_fps'],duration))
    return dict(start_s=a,end_s=b,part=PARTS[int(np.argmax(raw['part']))],score=raw['presence'])

def geometric_decode(scores,r,threshold):
    # Scores are tail-normalized to [0,1] using training correct recordings.
    scores=gaussian_filter1d(scores,sigma=2,axis=0)
    candidates=[];fps=r['video_fps']
    for part in range(2):
        active=scores[:,part]>=threshold
        # Close gaps no longer than 0.3 seconds, without extending outside recording.
        runs=np.flatnonzero(np.diff(np.r_[False,active,False].astype(int)))
        spans=[]
        for a,b in zip(runs[::2],runs[1::2]):
            if spans and a-spans[-1][1]<=round(.3*fps):spans[-1][1]=b
            else:spans.append([int(a),int(b)])
        for a,b in spans:
            if b-a>=max(2,round(.5*fps)):
                weight=float(np.sum(scores[a:b,part]-threshold))
                candidates.append((weight,dict(start_s=a/fps,end_s=b/fps,part=PARTS[part],score=float(np.mean(scores[a:b,part])))))
    return max(candidates,key=lambda x:x[0])[1] if candidates else None

class TCN(nn.Module):
    def __init__(self,channels,width):
        super().__init__();self.input=nn.Conv1d(channels,width,1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,5,padding=2*d,dilation=d),nn.ReLU(),nn.Dropout(.1),nn.Conv1d(width,width,1)) for d in [1,2,4,8]])
        self.boundary=nn.Conv1d(width,2,1);self.presence=nn.Linear(width*2,1);self.part=nn.Linear(width*2,2)
    def forward(self,x,mask):
        h=F.relu(self.input(x))*mask[:,None]
        for b in self.blocks:h=F.relu(h+b(h))*mask[:,None]
        avg=h.sum(2)/mask.sum(1)[:,None]
        maximum=h.masked_fill(~mask[:,None],-1e4).max(2).values
        pooled=torch.cat([avg,maximum],1)
        logits=self.boundary(h).masked_fill(~mask[:,None],-1e4)
        return self.presence(pooled)[:,0],self.part(pooled),logits

def batch(ids,inputs,record_map,device):
    arrays=[inputs[s] for s in ids];lengths=[len(a) for a in arrays];n=max(lengths)
    x=np.zeros((len(ids),arrays[0].shape[1],n),np.float32);mask=np.zeros((len(ids),n),bool)
    presence=[];parts=[];bounds=[]
    for i,(sid,a) in enumerate(zip(ids,arrays)):
        x[i,:,:len(a)]=a.T;mask[i,:len(a)]=True;r=record_map[sid]
        e=r['events'][0] if r['events'] else None
        presence.append(float(e is not None));parts.append(PARTS.index(e['part']) if e else 0)
        bounds.append([min(len(a)-1,max(0,round(e[k]*r['video_fps']-.5))) for k in ['start_s','end_s']] if e else [0,0])
    return [torch.as_tensor(z,device=device) for z in [x,mask,np.asarray(presence,np.float32),np.asarray(parts,np.int64),np.asarray(bounds,np.int64)]]

def infer(model,ids,inputs,record_map,device):
    model.eval();raw={}
    with torch.no_grad():
        for i in range(0,len(ids),16):
            ss=ids[i:i+16];x,mask,_,_,_=batch(ss,inputs,record_map,device)
            p,part,bound=model(x,mask)
            grid=(torch.arange(x.shape[2],device=device)[None,:]+.5)/mask.sum(1)[:,None]
            boundaries=(bound.softmax(2)*grid[:,None]).sum(2)
            for sid,pp,pa,bo in zip(ss,p.sigmoid().cpu().tolist(),part.softmax(1).cpu().tolist(),boundaries.cpu().tolist()):
                raw[sid]=dict(presence=pp,part=pa,boundaries=bo)
    return raw

def train_model(inputs,records,fold,width,seed,epochs,device,checkpoint):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    rm={r['sample_id']:r for r in records};tr=fold['recordings']['train'];va=fold['recordings']['val']
    model=TCN(next(iter(inputs.values())).shape[1],width).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    train_pos=sum(bool(rm[s]['events']) for s in tr)
    pos_weight=torch.tensor([(len(tr)-train_pos)/max(train_pos,1)],device=device)
    bestkey=None;beststate=None;history=[]
    for epoch in range(1,epochs+1):
        model.train();order=list(tr);random.shuffle(order);losses=[]
        for i in range(0,len(order),16):
            x,mask,presence,parts,bounds=batch(order[i:i+16],inputs,rm,device)
            p,part,logits=model(x,mask)
            loss=F.binary_cross_entropy_with_logits(p,presence,pos_weight=pos_weight)
            positive=presence.bool()
            if positive.any():
                loc=logits[positive];target=bounds[positive]
                ce=(F.cross_entropy(loc[:,0],target[:,0])+F.cross_entropy(loc[:,1],target[:,1]))*.5
                grid=(torch.arange(x.shape[2],device=device)[None,:]+.5)/mask.sum(1)[positive,None]
                expected=(loc.softmax(2)*grid[:,None]).sum(2)
                normtarget=(target+.5)/mask.sum(1)[positive,None]
                # Exact-boundary CE + normalized interval regression, on positive clips only.
                loss=loss+F.cross_entropy(part[positive],parts[positive])+.2*ce+5*F.smooth_l1_loss(expected,normtarget)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(loss.item())
        scheduler.step()
        if epoch%5==0 or epoch==epochs:
            raw=infer(model,va,inputs,rm,device)
            threshold,m=calibrate(raw,[rm[s] for s in va],neural_decode)
            key=(m['joint@0.5']['f1'],m['joint@0.3']['f1'],-(m['normal_false_alarm_rate'] or 0),-epoch)
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),threshold=threshold,validation=m))
            if bestkey is None or key>bestkey:
                bestkey=key;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};chosen=dict(epoch=epoch,threshold=threshold,validation=m)
    model.load_state_dict(beststate)
    torch.save(dict(state_dict=beststate,width=width,channels=next(iter(inputs.values())).shape[1],seed=seed,selection=chosen),checkpoint)
    return model,chosen,history

def reference_ids(r,train_records,count):
    candidates=[v for v in train_records if v['exercise']==r['exercise'] and v['global_label']=='Correct' and v['subject_uid']!=r['subject_uid']]
    if not candidates:raise ValueError(f'No eligible reference for {r["sample_id"]}')
    durations=np.median([c['skeleton_frames']/c['video_fps'] for c in candidates])
    # Prefer healthy correct demonstrators, then duration near train median; one per subject.
    candidates.sort(key=lambda c:(not c['subject_uid'].startswith('healthy:'),abs(c['skeleton_frames']/c['video_fps']-durations),c['sample_id']))
    selected=[];seen=set()
    for c in candidates:
        if c['subject_uid'] not in seen:selected.append(c['sample_id']);seen.add(c['subject_uid'])
        if len(selected)==count:break
    return selected

def prepare(records,fold,raw_features,artifact):
    rm={r['sample_id']:r for r in records};tr=[rm[s] for s in fold['recordings']['train']]
    # Equal recording weighting keeps long recordings from dominating normalization.
    mean=np.mean([raw_features[r['sample_id']].mean(0) for r in tr],0)
    var=np.mean([(raw_features[r['sample_id']]**2).mean(0) for r in tr],0)-mean**2
    std=np.maximum(np.sqrt(np.maximum(var,0)),.05)
    z={s:np.clip((f-mean)/std,-12,12).astype(np.float32) for s,f in raw_features.items()}
    inputs={m:{} for m in METHODS};deviations={};references={};runtime=[]
    for r in records:
        sid=r['sample_id'];q=z[sid];n=len(q)
        ex=np.tile(np.eye(3,dtype=np.float32)[EXERCISES.index(r['exercise'])],(n,1))
        clock=((np.arange(n)+.5)/n)[:,None].astype(np.float32)
        context=np.c_[ex,clock]
        inputs['B0_no_reference'][sid]=np.c_[q,context]
        refs=reference_ids(r,tr,3);references[sid]=refs
        start=time.perf_counter();linear=align(q,z[refs[0]],'linear');dtw=align(q,z[refs[0]],'dtw')
        band=[align(q,z[s],'band') for s in refs]
        runtime.append((time.perf_counter()-start)*1000)
        for name,ref in [('B2_linear',linear),('B2_dtw',dtw),('A1_band_dtw',band[0])]:
            inputs[name][sid]=np.c_[q,ref,q-ref,np.abs(q-ref),context].astype(np.float32)
        median=np.median(band,axis=0);disp=np.median(np.abs(np.asarray(band)-median),axis=0)
        inputs['P_multi_band_dtw'][sid]=np.c_[q,median,q-median,np.abs(q-median),disp,context].astype(np.float32)
        diff=(q-dtw)**2;d=q.shape[1]//2
        deviations[sid]=np.stack([np.mean(diff[:,:20],1),np.mean(diff[:,20:d],1)],1)
    save(artifact/'references.json',references)
    save(artifact/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),fit_ids=fold['recordings']['train']))
    save(artifact/'preparation_runtime.json',dict(all_alignments_per_record_median_ms=float(np.median(runtime)),p95_ms=float(np.percentile(runtime,95)),
        includes='linear + unrestricted DTW + up to three band DTWs; excludes loading and features; first call includes JIT'))
    return inputs,deviations,z

def fit_geometric(records,fold,deviations,z,method):
    rm={r['sample_id']:r for r in records};train=[rm[s] for s in fold['recordings']['train']]
    if method=='B1_dtw_deviation':raw=deviations
    else:
        raw={};d=next(iter(z.values())).shape[1]//2
        for ex in EXERCISES:
            correct=[r for r in train if r['exercise']==ex and r['global_label']=='Correct']
            for part,indices in enumerate([np.arange(20),np.arange(20,d)]):
                data=np.concatenate([z[r['sample_id']][:,indices] for r in correct])
                # Euclidean diagonal GMM adaptation; not a reproduction of manifold GMM.
                gmm=GaussianMixture(n_components=4,covariance_type='diag',reg_covar=.01,random_state=42,max_iter=150).fit(data[::3])
                for r in records:
                    if r['exercise']==ex:
                        sid=r['sample_id']
                        if sid not in raw:raw[sid]=np.zeros((len(z[sid]),2),np.float32)
                        raw[sid][:,part]=-gmm.score_samples(z[sid][:,indices])
    result={};normalization={}
    for ex in EXERCISES:
        normal=np.concatenate([raw[r['sample_id']] for r in train if r['exercise']==ex and not r['events']])
        low=np.quantile(normal,.5,axis=0);high=np.quantile(normal,.99,axis=0)
        normalization[ex]={'median':low.tolist(),'q99':high.tolist()}
        for r in records:
            if r['exercise']==ex:result[r['sample_id']]=np.clip((raw[r['sample_id']]-low)/np.maximum(high-low,.01),0,1)
    return result,normalization

def run(a):
    root=Path(a.project).resolve();art=root/'artifacts';runroot=art/'runs'/a.run
    runroot.mkdir(parents=True,exist_ok=True)
    records=[json.loads(l) for l in (art/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
    records=[r for r in records if r['usable_for_localization']]
    local=json.loads((root/'configs'/'local.json').read_text(encoding='utf8'))
    expected_root=Path(local['dataset_root']).resolve()
    for r in records:
        if expected_root not in Path(r['skeleton_path']).resolve().parents:
            raise ValueError('Manifest path does not belong to configured sole dataset entry')
    folds=json.loads((art/'splits.json').read_text(encoding='utf8'))
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    config=dict(task='annotated_principal_event_localization',methods=METHODS+['B1_dtw_deviation','GMM_euclidean_adaptation'],
        seeds=[42,43,44],widths=[32,64],epochs=a.epochs,selection='validation joint F1@0.5, joint F1@0.3, false alarms; earliest epoch tie',
        thresholds=[.2,.3,.4,.5,.6,.7,.8],band=.2,reference_count=3,normalization='train recording equal weight',
        device=str(device),torch=torch.__version__,hardware=torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',
        dataset_root=str(expected_root),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        manifest_sha256=hashlib.sha256((art/'manifest.jsonl').read_bytes()).hexdigest(),
        splits_sha256=hashlib.sha256((art/'splits.json').read_bytes()).hexdigest(),
        timing_scope='equal video/skeleton frame counts; unmeasured hardware synchronization',
        identity_scope='same subject token across centers grouped conservatively',
        limitations=['single principal event','small cohort','no online or clinical efficacy claim'])
    config_path=runroot/'config.json'
    if config_path.exists():
        previous=json.loads(config_path.read_text(encoding='utf8'))
        for key in ['epochs','seeds','widths','manifest_sha256','splits_sha256','dataset_root']:
            if previous[key]!=config[key]:raise ValueError('Resume configuration mismatch: '+key)
        if previous.get('code_sha256')!=config['code_sha256']:
            # An execution-only parallelization revision is explicitly recorded.
            save(runroot/('config_previous_'+previous['code_sha256'][:12]+'.json'),previous)
    save(config_path,config)
    raw_features={};feature_times=[]
    for r in records:
        start=time.perf_counter();x=np.loadtxt(r['skeleton_path'],dtype=np.float32)
        raw_features[r['sample_id']]=features(x,r['video_fps']);feature_times.append((time.perf_counter()-start)*1000)
    save(runroot/'feature_runtime.json',dict(load_and_features_median_ms=float(np.median(feature_times)),p95_ms=float(np.percentile(feature_times,95))))
    all_results=[]
    for fold in folds:
        if a.fold is not None and fold['fold']!=a.fold:continue
        f=fold['fold'];froot=runroot/f'fold{f}';froot.mkdir(exist_ok=True)
        prepared=froot/'prepared.npz'
        # Python objects are local trusted caches only, never load external pickle data.
        if prepared.exists():
            cache=np.load(prepared,allow_pickle=True);inputs=cache['inputs'].item();deviations=cache['deviations'].item();z=cache['z'].item()
        else:
            print(f'PREPARE fold={f}',flush=True);inputs,deviations,z=prepare(records,fold,raw_features,froot)
            np.savez_compressed(prepared,inputs=inputs,deviations=deviations,z=z)
        rm={r['sample_id']:r for r in records};val=[rm[s] for s in fold['recordings']['val']];test=[rm[s] for s in fold['recordings']['test']]
        for method in ['B1_dtw_deviation','GMM_euclidean_adaptation']:
            dest=froot/f'{method}.json'
            if dest.exists():all_results.append(json.loads(dest.read_text()));continue
            print(f'FIT fold={f} method={method}',flush=True)
            raw,normalization=fit_geometric(records,fold,deviations,z,method)
            threshold,vm=calibrate(raw,val,geometric_decode)
            preds={r['sample_id']:geometric_decode(raw[r['sample_id']],r,threshold) for r in test}
            result=dict(fold=f,method=method,seed=None,threshold=threshold,validation=vm,
                test={c:metrics(preds,test,None if c=='overall' else c) for c in ['overall','patient','healthy']})
            save(dest,result);save(froot/f'{method}_calibration.json',normalization)
            save(froot/f'{method}_predictions.json',preds);all_results.append(result)
        for method in METHODS:
            selection_file=froot/f'{method}_width_selection.json'
            candidates=[];models={}
            if selection_file.exists():chosen_width=json.loads(selection_file.read_text())['chosen_width']
            else:
                for width in [32,64]:
                    checkpoint=froot/f'{method}_w{width}_s42.pt'
                    if checkpoint.exists():
                        print(f'RESUME fold={f} method={method} seed=42 width={width}',flush=True)
                        cp=torch.load(checkpoint,map_location=device)
                        model=TCN(cp['channels'],cp['width']).to(device);model.load_state_dict(cp['state_dict']);chosen=cp['selection']
                    else:
                        print(f'TRAIN fold={f} method={method} seed=42 width={width}',flush=True)
                        model,chosen,history=train_model(inputs[method],records,fold,width,42,a.epochs,device,checkpoint)
                        save(froot/f'{method}_w{width}_s42_history.json',history)
                    m=chosen['validation'];key=(m['joint@0.5']['f1'],m['joint@0.3']['f1'],-(m['normal_false_alarm_rate'] or 0),-width)
                    candidates.append((key,width,chosen));models[width]=(model,chosen)
                winner=max(candidates,key=lambda c:c[0]);chosen_width=winner[1]
                save(selection_file,dict(chosen_width=chosen_width,candidates=[dict(width=w,selection=c) for _,w,c in candidates]))
            for seed in [42,43,44]:
                dest=froot/f'{method}_s{seed}.json'
                if dest.exists():all_results.append(json.loads(dest.read_text()));continue
                if seed==42 and chosen_width in models:model,chosen=models[chosen_width]
                elif seed==42:
                    cp=torch.load(froot/f'{method}_w{chosen_width}_s42.pt',map_location=device)
                    model=TCN(cp['channels'],cp['width']).to(device);model.load_state_dict(cp['state_dict']);chosen=cp['selection']
                else:
                    print(f'TRAIN fold={f} method={method} seed={seed} width={chosen_width}',flush=True)
                    model,chosen,history=train_model(inputs[method],records,fold,chosen_width,seed,a.epochs,device,froot/f'{method}_w{chosen_width}_s{seed}.pt')
                    save(froot/f'{method}_w{chosen_width}_s{seed}_history.json',history)
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter();raw=infer(model,fold['recordings']['test'],inputs[method],rm,device)
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000/max(len(test),1)
                preds={r['sample_id']:neural_decode(raw[r['sample_id']],r,chosen['threshold']) for r in test}
                result=dict(fold=f,method=method,seed=seed,width=chosen_width,selection=chosen,
                    parameters=sum(p.numel() for p in model.parameters()),batched_inference_ms_per_record=elapsed,
                    test={c:metrics(preds,test,None if c=='overall' else c) for c in ['overall','patient','healthy']})
                save(dest,result);save(froot/f'{method}_s{seed}_raw_predictions.json',raw)
                save(froot/f'{method}_s{seed}_predictions.json',preds);all_results.append(result)
                print(f'DONE fold={f} method={method} seed={seed}',flush=True)
        save(froot/'fold_results.json',[r for r in all_results if r['fold']==f])
    if all((runroot/f'fold{f["fold"]}'/'fold_results.json').exists() for f in folds):
        all_results=[]
        for fold in folds:all_results+=json.loads((runroot/f'fold{fold["fold"]}'/'fold_results.json').read_text())
        save(runroot/'results.json',all_results)
        summarize(root,runroot,records,folds,all_results)
    else:print('Worker completed; awaiting other folds before aggregate summary',flush=True)

def summarize(root,runroot,records,folds,results):
    rm={r['sample_id']:r for r in records};summary={};recomputed=[]
    for method in METHODS+['B1_dtw_deviation','GMM_euclidean_adaptation']:
        seed_results=[]
        for seed in ([42,43,44] if method in METHODS else [None]):
            preds={}
            for fold in folds:
                suffix=f'_s{seed}' if seed is not None else ''
                p=runroot/f'fold{fold["fold"]}'/f'{method}{suffix}_predictions.json'
                loaded=json.loads(p.read_text());expected=set(fold['recordings']['test'])
                assert set(loaded)==expected
                assert not (set(preds)&set(loaded));preds.update(loaded)
            assert set(preds)==set(rm)
            m={c:metrics(preds,records,None if c=='overall' else c) for c in ['overall','patient','healthy']}
            seed_results.append(m);recomputed.append(dict(method=method,seed=seed,metrics=m))
        summary[method]={c:{key:dict(mean=float(np.mean([m[c][key]['f1'] for m in seed_results])),std=float(np.std([m[c][key]['f1'] for m in seed_results],ddof=1)) if len(seed_results)>1 else None) for key in ['joint@0.3','joint@0.5','temporal@0.5']} for c in ['overall','patient','healthy']}
        summary[method]['normal_false_alarm_rate']=float(np.mean([m['overall']['normal_false_alarm_rate'] for m in seed_results]))
    save(runroot/'summary.json',summary);save(runroot/'pooled_recomputed.json',recomputed)
    save(root/'reports'/'experiment_summary.json',dict(run=runroot.name,summary=summary,results=results))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--run',default='principal_v1');p.add_argument('--epochs',type=int,default=60);p.add_argument('--cpu',action='store_true');p.add_argument('--fold',type=int,choices=[0,1,2])
    run(p.parse_args())
