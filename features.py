"""Extract ImageNet GAP features (original + 4 augmented views) for DenseNet121/ResNet50/VGG16."""
import numpy as np, tensorflow as tf, sys
from tensorflow.keras import applications as A
R='/home/claude/ghadeer/results'; W='/home/claude/ghadeer/weights'
X=np.load(f'{R}/X224.npy')
def views(x):  # deterministic augmentations of the kind used for offline dataset inflation
    out=[x, x[:, :, ::-1]]
    for ang in (-10, 10):
        out.append(np.stack([np.asarray(tf.keras.preprocessing.image.apply_affine_transform(im[...,None].astype('float32'),theta=ang,fill_mode='nearest'))[...,0] for im in x]))
    out.append(np.roll(x, (8, 8), axis=(1, 2)))
    return out
V=views(X)
nets={'densenet121':(A.DenseNet121,A.densenet.preprocess_input,'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'),
      'resnet50':(A.ResNet50,A.resnet50.preprocess_input,'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'),
      'inception_v3':(A.InceptionV3,A.inception_v3.preprocess_input,'inception_v3_notop.h5'),
      'vgg16':(A.VGG16,A.vgg16.preprocess_input,'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')}
for name,(fn,pp,wf) in [(k,v) for k,v in nets.items() if k=='inception_v3']:
    m=fn(include_top=False,weights=f'{W}/{wf}',input_shape=(224,224,3),pooling='avg')
    F=np.stack([m.predict(pp(np.repeat(v[...,None],3,-1).astype('float32')),batch_size=32,verbose=0) for v in V])
    np.save(f'{R}/feat_{name}.npy',F.astype('float32')); print(name,F.shape,flush=True)
