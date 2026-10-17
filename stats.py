"""Subject-bootstrap CIs on pooled OOF predictions, Wilcoxon between backbones, confusion matrices."""
import numpy as np, json
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score, cohen_kappa_score
R='/home/claude/ghadeer/results'
y=np.load(f'{R}/y.npy'); g=np.load(f'{R}/groups.npy'); subs=np.unique(g); idx={s:np.where(g==s)[0] for s in subs}
rng=np.random.default_rng(0); out={}
def boot(P):
    pred=P.argmax(1); vals=[]
    for _ in range(2000):
        ii=np.concatenate([idx[s] for s in rng.choice(subs,len(subs))])
        try: vals.append((balanced_accuracy_score(y[ii],pred[ii]),f1_score(y[ii],pred[ii],average='macro'),roc_auc_score(y[ii],P[ii],multi_class='ovr',labels=range(5))))
        except ValueError: pass
    v=np.array(vals); pt=(balanced_accuracy_score(y,pred),f1_score(y,pred,average='macro'),roc_auc_score(y,P,multi_class='ovr',labels=range(5)))
    return {m:[float(pt[i]),float(np.percentile(v[:,i],2.5)),float(np.percentile(v[:,i],97.5))] for i,m in enumerate(['bacc','f1','auc'])}
O=dict(np.load(f'{R}/oof_features.npz'))
import os
for k in ['cnn_image','cnn_subject']:
    if os.path.exists(f'{R}/{k}_oof.npy'): O[k]=np.load(f'{R}/{k}_oof.npy')
for k,P in O.items():
    out[k]=boot(P); out[k]['cm']=confusion_matrix(y,P.argmax(1)).tolist()
    out[k]['acc']=float(accuracy_score(y,P.argmax(1))); out[k]['kappa']=float(cohen_kappa_score(y,P.argmax(1)))
    out[k]['per_class_f1']=f1_score(y,P.argmax(1),average=None).tolist()
pf=json.load(open(f'{R}/protocols_features.json')); W={}
nets=list(pf)
for i in range(len(nets)):
    for j in range(i+1,len(nets)):
        a=[f['f1'] for f in pf[nets[i]]['subject/lr']['folds']]; b=[f['f1'] for f in pf[nets[j]]['subject/lr']['folds']]
        W[f'{nets[i]} vs {nets[j]}']=dict(median_diff=float(np.median(np.array(a)-np.array(b))),p=float(wilcoxon(a,b).pvalue))
out['wilcoxon_subject_f1']=W
json.dump(out,open(f'{R}/stats.json','w'),indent=1)
for k,v in out.items():
    if k!='wilcoxon_subject_f1': print(k,{m:[round(x,3) for x in v[m]] for m in ['bacc','f1','auc']})
print(W)
