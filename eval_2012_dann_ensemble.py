#!/usr/bin/env python3
"""3-seed DANN 概率集成评估（softmax[:,1] 取平均）。"""
import argparse, sys, types, importlib.util
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from sklearn.metrics import classification_report, roc_auc_score

from eval_2012_dann_live import crop_224, res_um, load_gt_points, match
from gen_dataset_2012 import find_csv, label_bboxes, load_gt_regions, n_components

SRC = ("/home/hyw/chendawei/cellpose-DANN-opencode/Biomedical-Data-Design-MIDOG-main/"
       "Phase 2/final_model.py")
DST = Path("/home/hyw/dataset/chendawei/train-classification/ICPR-2012")
NDC = {'Tumor': 7, 'Species': 2, 'Origin': 4, 'Scanner': 4}

def load_mod():
    for n in ['umap','seaborn']:
        if n not in sys.modules: sys.modules[n]=types.ModuleType(n)
    sys.modules['seaborn'].heatmap=lambda *a,**k: None
    sys.modules['umap'].UMAP=object
    spec=importlib.util.spec_from_file_location("afm",SRC); m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--weights', nargs='+', required=True)
    ap.add_argument('--batch', type=int, default=256)
    args=ap.parse_args()
    import torchvision
    o=torchvision.models.resnet50; torchvision.models.resnet50=lambda *a,**k:o(weights=None)
    fm=load_mod(); dev='cuda' if torch.cuda.is_available() else 'cpu'
    models=[]
    for w in args.weights:
        m=fm.DANNModel(2,NDC,0.0); m.load_state_dict(torch.load(w,map_location='cpu')); m.eval().to(dev); models.append(m)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img_dir=DST/"train-val-test"/"test-original-img"; recs=[]; gt_by={}; total_gt=0
    for hi,bmp in enumerate(sorted(img_dir.glob("*.bmp")),1):
        base=bmp.stem; img=cv2.cvtColor(cv2.imread(str(bmp)),cv2.COLOR_BGR2RGB)
        px=res_um(base); csv=find_csv(base); gts=load_gt_points(csv) if csv else []
        gt_by[base]=(gts,5.0/px); total_gt+=len(gts)
        mask=np.load(DST/"cellpose_cache"/"test_masks_cp3"/f"{base}_mask.npy")
        labels=[int(l) for l in np.unique(mask) if l!=0]
        cents=ndimage.center_of_mass(np.ones(mask.shape,np.float32),mask,labels); bboxes=label_bboxes(mask)
        pos=set()
        if csv:
            for xs,ys in load_gt_regions(csv):
                if n_components(xs,ys)!=1: continue
                l,c=np.unique(mask[ys,xs],return_counts=True); nz=l!=0
                if nz.sum()==0: continue
                pos.add(int(l[np.argmax(np.where(nz,c,0))]))
        patches=[]; meta=[]
        for lab,(cy,cx) in zip(labels,cents):
            y0,y1,x0,x1=bboxes[lab]; patches.append(crop_224(img,(x0,y0,x1,y1))); meta.append((float(cx),float(cy),1 if lab in pos else 0))
        probs=[]
        for i in range(0,len(patches),args.batch):
            tens=torch.stack([tf(Image.fromarray(p)) for p in patches[i:i+args.batch]],0).to(dev)
            with torch.no_grad():
                ps=[torch.softmax(m.predict_only(tens),1)[:,1] for m in models]
                probs.extend(torch.stack(ps,0).mean(0).cpu().numpy().tolist())
        for (cx,cy,y),p in zip(meta,probs): recs.append((base,cx,cy,float(p),y))
        print(f"[{hi}] {base} masks={len(labels)}",flush=True)
    P=np.array([r[3] for r in recs]); Y=np.array([r[4] for r in recs])
    print(f"candidates={len(recs)} pos={int(Y.sum())} neg={int((Y==0).sum())}")
    print("[ensemble @0.5]"); print(classification_report(Y,(P>=0.5).astype(int),target_names=['Non-mitotic','Mitotic'],zero_division=0))
    print(f"AUC={roc_auc_score(Y,P):.4f}")
    by={}
    for (b,cx,cy,p,y) in recs: by.setdefault(b,[]).append((cx,cy,p))
    best={'f1':-1}; f05=None
    for thr in np.arange(0.05,0.9995,0.005):
        tp=fp=fn=0
        for b,rs in by.items():
            gts,tpx=gt_by[b]; dets=[(x,y,p) for (x,y,p) in rs if p>=thr]
            t,f,n=match(dets,gts,tpx); tp+=t; fp+=f; fn+=n
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0; f1=2*pr*rc/(pr+rc) if pr+rc else 0
        if f1>best['f1']: best=dict(f1=f1,thr=thr,tp=tp,fp=fp,fn=fn,p=pr,r=rc)
        if abs(thr-0.5)<1e-9: f05=dict(f1=f1,tp=tp,fp=fp,fn=fn,p=pr,r=rc)
    print(f"e2e oracle thr={best['thr']:.3f} F1={best['f1']:.4f} P={best['p']:.4f} R={best['r']:.4f} TP={best['tp']} FP={best['fp']} FN={best['fn']}")
    print(f"e2e @0.5 F1={f05['f1']:.4f} P={f05['p']:.4f} R={f05['r']:.4f} TP={f05['tp']} FP={f05['fp']} FN={f05['fn']}")

if __name__=='__main__': main()
