"""Protocol comparison on frozen ImageNet features (LR + 1-NN)."""
import numpy as np, json, warnings
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
warnings.filterwarnings('ignore')
R='/home/claude/ghadeer/results'
y=np.load(f'{R}/y.npy'); g=np.load(f'{R}/groups.npy'); osp=np.load(f'{R}/orig_split.npy'); N=len(y)
def metrics(yt,p,prob=None):
    d=dict(acc=accuracy_score(yt,p),bacc=balanced_accuracy_score(yt,p),f1=f1_score(yt,p,average='macro'),kappa=cohen_kappa_score(yt,p))
    if prob is not None: d['auc']=roc_auc_score(yt,prob,multi_class='ovr',average='macro',labels=range(5))
    return d
def lr(): return make_pipeline(StandardScaler(),LogisticRegression(C=0.01,class_weight='balanced',max_iter=3000))
def knn(): return KNeighborsClassifier(1,metric='cosine')
def splits(kind,seed):
    return (StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(N),y) if kind=='image'
            else StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(np.zeros(N),y,g))
out={}; oof={}
for net in ['vgg16','densenet121','inception_v3']:
    F=np.load(f'{R}/feat_{net}.npy'); F0=F[0]; V=F.shape[0]
    res={}
    for kind in ['image','subject']:
        for clfname,mk in [('lr',lr),('1nn',knn)]:
            folds=[]; same=[]
            for seed in range(3):
                P=np.zeros((N,5)); pred=np.zeros(N,int)
                for tr,te in splits(kind,seed):
                    m=mk().fit(F0[tr],y[tr]); pred[te]=m.predict(F0[te])
                    if clfname=='lr': P[te]=m.predict_proba(F0[te])
                    else:
                        nb=m.kneighbors(F0[te],1,return_distance=False)[:,0]; same.append((g[tr][nb]==g[te]).mean())
                    folds.append(metrics(y[te],pred[te],P[te] if clfname=='lr' else None))
                if seed==0 and clfname=='lr': oof[f'{net}_{kind}']=P
            res[f'{kind}/{clfname}']=dict(mean={k:float(np.mean([f[k] for f in folds])) for k in folds[0]},sd={k:float(np.std([f[k] for f in folds],ddof=1)) for k in folds[0]},folds=folds,same_subject_nn=float(np.mean(same)) if same else None)
    # distributed split, as given and with overlapping subjects removed from test
    tr=osp=='train'; te=~tr; clean=te & ~np.isin(g,g[tr])
    for nm,mask in [('official',te),('official_clean',clean)]:
        m=lr().fit(F0[tr],y[tr]); res[nm+'/lr']=dict(mean=metrics(y[mask],m.predict(F0[mask]),m.predict_proba(F0[mask])),n_test=int(mask.sum()),n_subj=int(len(set(g[mask]))))
    # offline augmentation BEFORE splitting (all views pooled, image-level) vs AFTER (subject folds, views only in train)
    Fa=F.reshape(V*N,-1); ya=np.tile(y,V); ga=np.tile(g,V); folds=[]
    for tr2,te2 in StratifiedKFold(5,shuffle=True,random_state=0).split(Fa,ya):
        m=lr().fit(Fa[tr2],ya[tr2]); folds.append(metrics(ya[te2],m.predict(Fa[te2]),m.predict_proba(Fa[te2])))
    res['aug_before_split/lr']=dict(mean={k:float(np.mean([f[k] for f in folds])) for k in folds[0]},sd={k:float(np.std([f[k] for f in folds],ddof=1)) for k in folds[0]})
    folds=[]
    for tr,te in splits('subject',0):
        tri=np.concatenate([tr+v*N for v in range(V)])
        m=lr().fit(Fa[tri],ya[tri]); folds.append(metrics(y[te],m.predict(F0[te]),m.predict_proba(F0[te])))
    res['aug_after_split_subject/lr']=dict(mean={k:float(np.mean([f[k] for f in folds])) for k in folds[0]},sd={k:float(np.std([f[k] for f in folds],ddof=1)) for k in folds[0]})
    out[net]=res; print(net,{k:round(v['mean']['f1'],3) for k,v in res.items()},flush=True)
json.dump(out,open(f'{R}/protocols_features.json','w'),indent=1)
np.savez(f'{R}/oof_features.npz',**oof)
