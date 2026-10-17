"""Thesis CNN ('VGG-12' in the thesis) trained from scratch under image-level vs subject-level 5-fold CV.
Usage: python cnn.py <protocol: image|subject> <epochs>"""
import sys, numpy as np, tensorflow as tf, time, json
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from tensorflow.keras import layers as L
tf.config.threading.set_intra_op_parallelism_threads(2)
R='/home/claude/ghadeer/results'; proto=sys.argv[1]; EP=int(sys.argv[2]); SEED=42
X=np.load(f'{R}/X150.npy').astype('float32')[...,None]/255.; y=np.load(f'{R}/y.npy'); g=np.load(f'{R}/groups.npy')
def thesis_cnn():
    m=tf.keras.Sequential([L.Input((150,150,1)),L.RandomFlip('horizontal'),L.RandomRotation(0.03),L.RandomTranslation(0.05,0.05),
      L.Conv2D(32,3,activation='relu'),L.Conv2D(64,3,activation='relu'),L.MaxPooling2D(2),L.Dropout(.3),
      L.Conv2D(64,3,activation='relu'),L.Conv2D(64,3,activation='relu'),L.Dropout(.3),L.MaxPooling2D(2),L.Dropout(.3),
      L.Conv2D(128,3,activation='relu'),L.Conv2D(128,3,activation='relu'),L.Conv2D(128,3,activation='relu'),L.MaxPooling2D(2),L.Dropout(.3),
      L.Conv2D(128,3,activation='relu'),L.Conv2D(256,3,activation='relu'),L.MaxPooling2D(2),L.Dropout(.3),
      L.Flatten(),L.Dense(512,activation='relu'),L.Dense(512,activation='relu'),L.Dropout(.3),L.Dense(5,activation='softmax')])
    m.compile(tf.keras.optimizers.Adam(1e-4),'sparse_categorical_crossentropy',metrics=['accuracy']); return m
cv=StratifiedKFold(5,shuffle=True,random_state=SEED).split(X,y) if proto=='image' else StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,g)
import os
K=int(sys.argv[3]) if len(sys.argv)>3 else 5
if os.path.exists(f'{R}/cnnbal_{proto}_oof.npy'):
    P=np.load(f'{R}/cnnbal_{proto}_oof.npy'); fold=np.load(f'{R}/cnnbal_{proto}_fold.npy')
else: P=np.zeros((len(y),5),'float32'); fold=np.zeros(len(y),int)
hist=[]; ran=0
for k,(tr,te) in enumerate(cv):
    if P[te].sum()>0: continue
    if ran>=K: break
    ran+=1
    tf.keras.utils.set_random_seed(SEED+k); t=time.time()
    m=thesis_cnn(); cw=len(tr)/(5*np.bincount(y[tr],minlength=5)); h=m.fit(X[tr],y[tr],epochs=EP,batch_size=32,verbose=0,class_weight=dict(enumerate(cw)))
    P[te]=m.predict(X[te],verbose=0); fold[te]=k
    acc=(P[te].argmax(1)==y[te]).mean(); hist.append(h.history); print(f'{proto} fold {k} acc={acc:.3f} train_acc={h.history["accuracy"][-1]:.3f} {time.time()-t:.0f}s',flush=True)
    np.save(f'{R}/cnnbal_{proto}_oof.npy',P); np.save(f'{R}/cnnbal_{proto}_fold.npy',fold)
json.dump(hist,open(f'{R}/cnnbal_{proto}_hist_{int(time.time())}.json','w'))
