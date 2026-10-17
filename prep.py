"""Load ADNI-1296 manifest, drop exact duplicates, cache grayscale arrays."""
import csv, numpy as np, os, sys
from PIL import Image
ROOT='/home/claude/ghadeer/data/Alzheimers-ADNI'; OUT='/home/claude/ghadeer/results'
rows=[r for r in csv.DictReader(open('/home/claude/ghadeer/out/adni1296_manifest_subject_folds.csv')) if not r['exact_duplicate_of']]
LABELS=['CN','EMCI','MCI','LMCI','AD']
for S in (150,224):
    X=np.stack([np.asarray(Image.open(os.path.join(ROOT,r['path'])).convert('L').resize((S,S),Image.BILINEAR)) for r in rows]).astype(np.uint8)
    np.save(f'{OUT}/X{S}.npy',X)
np.save(f'{OUT}/y.npy',np.array([LABELS.index(r['label']) for r in rows]))
np.save(f'{OUT}/groups.npy',np.array([r['subject'] for r in rows]))
np.save(f'{OUT}/orig_split.npy',np.array([r['orig_split'] for r in rows]))
print(len(rows),X.shape)
