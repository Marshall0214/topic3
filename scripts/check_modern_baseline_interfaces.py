"""Synthetic modern-baseline interface checks: zero optimizer steps, no dataset access."""
import _bootstrap
from _bootstrap import ROOT
import argparse,sys,time,json,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
OUT=ROOT/'evidence/modern_baseline_audit_20261014';OUT.mkdir(parents=True,exist_ok=True)
torch.manual_seed(20261014);torch.set_num_threads(4)
p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',choices=['dest','exechecker']);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();device=torch.device(a.device)
start=time.perf_counter()
def source_hashes(folder):
 return {str(x.relative_to(folder)):hashlib.sha256(x.read_bytes()).hexdigest() for x in folder.rglob('*.py')}
report={'date':'2026-10-14','candidate':a.candidate,'scope':'synthetic interface only; no clinical scores','optimizer_steps':0,'dataset_files_read':0,'validation_or_test_inference':False,'torch':torch.__version__,'device':str(device)}
if a.candidate=='dest':
 assert device.type=='cpu','The full DeST audit is fixed to CPU for this comparison'
 source=ROOT/'upstream/dest';sys.path.insert(0,str(source));before=source_hashes(source)
 from libs.models.DeST_linearformer import Model
 from libs.loss_fn import ActionSegmentationLoss,BoundaryRegressionLoss
 config=dict(in_channel=3,n_features=64,n_classes=3,n_stages=4,n_layers=10,n_refine_layers=10,n_stages_asb=2,n_stages_brb=3,SFI_layer=list(range(1,10)),dataset='PKU-subject')
 x=torch.randn(3,3,25,64);mask=torch.ones(3,1,64,dtype=torch.bool)
 y=torch.full((3,64),255,dtype=torch.long);y[0]=0;y[1,16:32]=1;y[2,32:50]=2
 boundary=torch.zeros(3,1,64);known=torch.zeros(3,1,64,dtype=torch.bool);known[0]=True
 for i,begin,end in [(1,16,32),(2,32,50)]:boundary[i,0,begin]=boundary[i,0,end]=1;known[i,0,begin]=known[i,0,end]=True
 model=Model(**config);spatial=[]
 hook=model.SP.register_forward_hook(lambda m,i,o:spatial.append(list(o.shape)))
 model.train();classes,bounds=model(x,mask)
 assert len(classes)==2 and len(bounds)==3
 ce=ActionSegmentationLoss(ce=True,focal=False,tmse=False,gstmse=False,ignore_index=255)
 bce=BoundaryRegressionLoss(bce=True)
 loss=sum(ce(z,y,x) for z in classes)/len(classes)+.1*sum(bce(z,boundary,known) for z in bounds)/len(bounds)
 assert torch.isfinite(loss);loss.backward()
 grad={}
 for prefix in ['SP','STI','conv_cls','conv_bound','asb','brb']:
  gs=[p.grad for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.grad is not None]
  assert gs and all(torch.isfinite(g).all() for g in gs),prefix
  grad[prefix]={'tensors':len(gs),'nonzero_total':bool(sum(float(g.abs().sum()) for g in gs)>0)}
 model.eval()
 with torch.no_grad():c,b=model(x,mask)
 assert list(c.shape)==[3,3,64] and list(b.shape)==[3,1,64]
 hook.remove()
 cfgref=dict(config,in_channel=9);ref_model=Model(**cfgref).eval();ref=torch.randn_like(x)
 paired=torch.cat([x,ref,x-ref],dim=1)
 with torch.no_grad():cr,br=ref_model(paired,mask)
 assert list(cr.shape)==[3,3,64] and list(br.shape)==[3,1,64]
 assert torch.isfinite(cr).all() and torch.isfinite(br).all()
 report.update(config=config,parameters=sum(p.numel() for p in model.parameters()),input_shape=list(x.shape),spatial_shape=spatial[0],class_training_stage_shapes=[list(z.shape) for z in classes],boundary_training_stage_shapes=[list(z.shape) for z in bounds],eval_class_shape=list(c.shape),eval_boundary_shape=list(b.shape),partial_label_loss=float(loss.detach()),finite_gradients=True,gradient_groups=grad,simple_reference_interface={'channels':'query, reference, query-reference; not learned alignment or validated reference selection','config':cfgref,'parameters':sum(p.numel() for p in ref_model.parameters()),'input_shape':list(paired.shape),'output_class_shape':list(cr.shape)},loss_limitations=['CE and BCE use author implementations but task targets/masks are new adaptation','GSTMSE intentionally not used for this interface check; original full recipe not reproduced','known negatives from synthetic normal record; outside synthetic error interval is unknown','random skeleton/reference has no clinical meaning'])
else:
 import yaml
 source=ROOT/'upstream/ExeChecker_snapshot';sys.path.insert(0,str(source));before=source_hashes(source)
 from model.stgat.st_gat import DSTANet_smp
 from model.taskModels import MultiTaskModel
 cfg=yaml.safe_load((source/'config/execheck_Multi_perExe.yaml').read_text(encoding='utf-8'))
 model=MultiTaskModel(DSTANet_smp(num_point=21,skeleton='execheck_skeleton',**cfg['model_param']),embedding_dim=128).train().to(device)
 prepool=[];hook=model.graph_net.graph_layers[-1].register_forward_hook(lambda m,i,o:prepool.append(list(o.shape)))
 x=torch.randn(2,3,128,21,1,device=device)
 with torch.no_grad():features,embeddings=model(x)
 sf=list(model.graph_net.get_stgat_att_w_s().shape);tf=list(model.graph_net.get_stgat_att_w_t().shape)
 with torch.no_grad():f1,e1=model(x[:1])
 hook.remove()
 assert torch.isfinite(features).all() and torch.isfinite(embeddings).all()
 model.eval()
 with torch.no_grad():fe,ee=model(x)
 eval_finite=bool(torch.isfinite(fe).all() and torch.isfinite(ee).all())
 report.update(train_mode_shape_check=True,batchnorm_statistics_updated_without_optimizer=True,eval_after_shape_check_finite=eval_finite,parameters=sum(p.numel() for p in model.parameters()),input_shape=list(x.shape),feature_shape=list(features.shape),embedding_shape=list(embeddings.shape),prepool_feature_shapes=prepool,spatial_attention_shape=sf,temporal_attention_shape=tf,batch1_feature_shape=list(f1.shape),batch1_embedding_shape=list(e1.shape),batch_dimension_collapses=embeddings.ndim==2 and e1.ndim==1,native_error_interval_output=False,limitations=['author21 topology on synthetic input, not Kinect25 adapter','no original trainer import due missing modules','no original JoA localization pipeline reproduced','prepool feature hook proves availability only, not a new trained temporal locator'])
assert before==source_hashes(source),'Author source files changed'
if device.type=='cuda':torch.cuda.synchronize()
report.update(author_python_sources_unchanged=True,source_python_sha256=before,elapsed_seconds=time.perf_counter()-start)
(OUT/(a.candidate+('_cuda' if device.type=='cuda' else '')+'_synthetic_check.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['source_python_sha256','config','loss_limitations','limitations']},ensure_ascii=False,indent=2))