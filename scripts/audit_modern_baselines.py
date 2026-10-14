"""Read-only source audit of modern third-point candidates; never imports trainers."""
import _bootstrap
from _bootstrap import ROOT
from pathlib import Path
import ast,json,hashlib
OUT=ROOT/'evidence/modern_baseline_audit_20261014';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def evidence(rel,needles):
 p=ROOT/rel;s=p.read_text(encoding='utf-8-sig');lines=s.splitlines()
 return {'file':rel,'sha256':sha(p),'anchors':{n:[i+1 for i,l in enumerate(lines) if n in l] for n in needles}}
def module_exists(base,name):
 p=base.joinpath(*name.split('.'));return p.with_suffix('.py').exists() or (p/'__init__.py').exists()
def attributes(rel,cls):
 tree=ast.parse((ROOT/rel).read_text(encoding='utf-8-sig'))
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 init=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
 return sorted({n.attr for n in ast.walk(init) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and isinstance(n.ctx,ast.Store)})
base=ROOT/'upstream/ExeChecker_snapshot'
exe={'name':'ExeChecker','pinned_commit':'1e6df144bb49fd1173b027115d07a332f018c682','local_missing_modules':{n:not module_exists(base,n) for n in ['train_val_test','loss.hierarchical_loss','joints_dtw','utility.video_data']},'entry':'upstream/ExeChecker_snapshot/trainMulti_perExe.py','top_level_training_block':True,'evidence':[
 evidence('upstream/ExeChecker_snapshot/trainMulti_perExe.py',['from train_val_test','hierarchical_loss','.cuda()','TripletLoss_ratio','dist_a2p','with TimerBlock']),
 evidence('upstream/ExeChecker_snapshot/model/taskModels.py',['features, _','return features, embeddings']),
 evidence('upstream/ExeChecker_snapshot/model/stgat/st_gat.py',['feat.mean(1)','feat.squeeze()','return feat, extra','get_stgat_att_w_s']),
 evidence('upstream/ExeChecker_snapshot/model/stgat/AdjMatrixGraph.py',['execheck_skeleton','uiprmd_skeleton','NotImplementedError']),
 evidence('upstream/ExeChecker_snapshot/loss/loss.py',['class TripletLoss_ratio','torch.exp(pos_dist)','ratio.mean()'])]}
dest={'name':'DeST full official architecture','pinned_commit':'319e1f10f310e08a198fba3a32a4d82416ecb15f','full_model_attributes':attributes('upstream/dest/libs/models/DeST_linearformer.py','Model'),'old_stage1_attributes':attributes('legacy/dest/stage1.py','TemporalModel'),'training_assets_exist':{n:(ROOT/'upstream/dest'/n).exists() for n in ['train.py','evaluate.py','libs/dataset.py','libs/helper.py','libs/loss_fn/__init__.py','libs/checkpoint.py','config/DeST_linearformer/PKU-subject/config.yaml']},'evidence':[
 evidence('upstream/dest/libs/models/DeST_linearformer.py',['self.SP =','self.STI =','self.conv_cls =','self.conv_bound =','self.asb =','self.brb =','return (outputs_cls','return (out_cls']),
 evidence('upstream/dest/libs/loss_fn/__init__.py',['ignore_index: int = 255','CrossEntropyLoss','criterion(pred[mask]']),
 evidence('upstream/dest/libs/loss_fn/tmse.py',['torch.where(gt != self.ignore_index)','F.log_softmax(pred[:, 1:], dim=1)']),
 evidence('upstream/dest/train.py',['config.param_search','mode="validation"','best_test_F1','max_epoch']),
 evidence('legacy/dest/stage1.py',['self.spatial =','self.temporal =','self.boundary =','self.presence =','stage2_started=False'])]}
dy={'name':'DyFADet','scope':'partial pinned model snapshot; original task video-feature TAL','pinned_commit':'f645cafd50f9a4414bbb83e4b5e5bcb5eed04ae2','train_entry_present':(ROOT/'upstream/dyfadet/train.py').exists(),'previous_check':json.loads((ROOT/'legacy/literature_20261011/audit/dyfadet/full_model_check.json').read_text(encoding='utf-8')),'note':'Existing check used reduced width and untested NMS shim, not full training reproduction; current selection does not prioritize video-origin models.'}
coach={'name':'CoachMe','scope':'main audited ERROR configuration uses stored error-window fields; not proof all possible localization code is absent','evidence':[evidence('legacy/literature_20261011/audit/coachme/dataloaders_Dataset.py',['setting == \'ERROR\'','error_std_start_frame','error_start_frame','error_seq_len']),evidence('legacy/literature_20261011/audit/coachme/models_CoachMe.py',['T5ForConditionalGeneration','t5-base'])]}
report={'date':'2026-10-14','scope':'fixed local snapshots + primary literature review; no trainer import, no data training or test inference','remote_head':'CLI verification failed on network; no claim fixed snapshots equal latest upstream HEAD','candidates':[exe,dest,dy,coach]}
(OUT/'source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'exechecker_missing_modules':exe['local_missing_modules'],'dest_full_attributes':dest['full_model_attributes'],'old_stage1_attributes':dest['old_stage1_attributes'],'dest_training_assets':dest['training_assets_exist'],'dyfadet_train_entry_present':dy['train_entry_present']},ensure_ascii=False,indent=2))