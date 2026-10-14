import json, sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'keraal_loc'))
from experiment import features,align,event_iou,metrics,geometric_decode,reference_ids,TCN
from audit import annotation

class ExperimentTests(unittest.TestCase):
    def test_temporal_track_attributes(self):
        label,ev=annotation('D:/AQA/dataset/keraal/group1A/annotatorA/G1A-CTK-R1-Brest-022.anvil')
        self.assertEqual(label,'Incorrect');self.assertEqual(ev[0]['start_s'],12.08)
        self.assertEqual(ev[0]['part'],'Arms')

    def test_geometry_scale_translation_and_twist(self):
        p=np.zeros((40,25,3),np.float32)
        p[:,20,1]=1;p[:,1,1]=.5
        p[:,12,0]=-.2;p[:,16,0]=.2
        p[:,4,:]=[-.3,1,0];p[:,8,:]=[.3,1,0]
        p[:,5,:]=[-.6,.7,0];p[:,6,:]=[-.8,.4,0]
        p[:,9,:]=[.6,.7,0];p[:,10,:]=[.8,.4,0]
        original=features(p,10)
        np.testing.assert_allclose(original,features(p*2+np.array([1,2,3]),10),atol=1e-5)
        p[20:,4,2]=-.3;p[20:,8,2]=.3
        twisted=features(p,10)
        self.assertGreater(np.max(np.abs(twisted-original)),.1)

    def test_dtw_preserves_query_grid_and_identity(self):
        t=np.linspace(0,1,35);x=np.c_[np.sin(t*5),np.cos(t*5),t,t**2].astype(np.float32)
        np.testing.assert_allclose(align(x,x),x,atol=1e-6)
        ref=x[::2]
        for mode in ['linear','dtw','band']:
            mapped=align(x,ref,mode);self.assertEqual(mapped.shape,x.shape)
            self.assertTrue(np.isfinite(mapped).all())

    def test_event_matching_and_false_alarms(self):
        e=dict(start_s=2.,end_s=4.,part='Arms')
        r=[dict(sample_id='a',subject_uid='patient:R1',events=[e]),dict(sample_id='b',subject_uid='patient:R2',events=[])]
        m=metrics({'a':dict(e,part='Trunk'),'b':e},r)
        self.assertEqual(m['temporal@0.5']['tp'],1)
        self.assertEqual(m['joint@0.5']['tp'],0)
        self.assertEqual(m['joint@0.5']['fp'],2)
        self.assertEqual(m['joint@0.5']['fn'],1)
        self.assertEqual(m['normal_false_alarm_rate'],1)
        self.assertEqual(event_iou(e,e),1)
        self.assertEqual(metrics({'a':e,'b':None},r)['joint@0.5']['f1'],1)

    def test_geometric_decoder_seconds(self):
        s=np.zeros((100,2));s[20:60,0]=1
        e=geometric_decode(s,dict(video_fps=10),.5)
        self.assertEqual(e['part'],'Arms');self.assertAlmostEqual(e['start_s'],2.,delta=.2)
        self.assertAlmostEqual(e['end_s'],6.,delta=.2)
        self.assertIsNone(geometric_decode(s*0,dict(video_fps=10),.5))

    def test_splits_and_reference_isolation(self):
        root=Path(__file__).resolve().parents[1]/'manifests/historical/principal_v1'
        rows=[json.loads(x) for x in (root/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
        valid=[r for r in rows if r['usable_for_localization']]
        folds=json.loads((root/'splits.json').read_text())
        rm={r['sample_id']:r for r in valid};testsets=[]
        for fold in folds:
            roles=fold['subjects']
            self.assertFalse(set(roles['train'])&set(roles['val']))
            self.assertFalse(set(roles['train'])&set(roles['test']))
            self.assertFalse(set(roles['test'])&set(roles['val']))
            tr=[rm[s] for s in fold['recordings']['train']]
            for r in valid:
                refs=reference_ids(r,tr,3)
                self.assertGreater(len(refs),0)
                for s in refs:
                    self.assertIn(s,fold['recordings']['train'])
                    self.assertNotEqual(rm[s]['subject_uid'],r['subject_uid'])
                    self.assertEqual(rm[s]['exercise'],r['exercise'])
                    self.assertEqual(rm[s]['global_label'],'Correct')
            testsets+=fold['recordings']['test']
        self.assertEqual(len(testsets),len(set(testsets)))
        self.assertEqual(set(testsets),set(rm))

    def test_padding_does_not_change_valid_predictions(self):
        import torch
        torch.manual_seed(10);torch.set_num_threads(2)
        model=TCN(8,16).eval();x=torch.randn(1,8,43)
        with torch.no_grad():
            p,a,b=model(x,torch.ones(1,43,dtype=torch.bool))
            padded=torch.zeros(1,8,71);padded[:,:,:43]=x
            mask=torch.zeros(1,71,dtype=torch.bool);mask[:,:43]=True
            pp,aa,bb=model(padded,mask)
        torch.testing.assert_close(p,pp,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(a,aa,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(b,bb[:,:,:43],rtol=1e-5,atol=1e-5)

if __name__=='__main__':unittest.main()
