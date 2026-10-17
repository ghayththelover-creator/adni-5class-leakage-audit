ADNI 5-class leakage audit
Code and subject-level split manifest for:
"Subject-Level Evaluation Exposes Data Leakage in Five-Class Alzheimer's Disease Staging from MRI Slices: An Audit and Leakage-Free Benchmark on ADNI" (G. Alkufahi).
Contents
File	Purpose
`prep.py`	Drops exact duplicates, caches grayscale arrays (150 and 224 px)
`features.py`	Frozen ImageNet features (VGG16, DenseNet121, InceptionV3) for the original image and 4 augmented views
`evaluate.py`	Protocol comparison: image-level CV, distributed split (with/without overlapping subjects), augmentation before split, subject-level CV; logistic regression and 1-NN probe
`stats.py`	Subject-bootstrap 95% CIs, Wilcoxon tests, confusion matrices
`cnn.py`, `cnn_bal.py`	Thesis CNN trained from scratch (faithful and class-balanced variants)
`protocols_features.json`, `stats.json`	All reported numbers
Data
The subject-level fold manifest is not included because it contains participant-level ADNI identifiers and diagnoses; it is available from the corresponding author to researchers with approved ADNI access. `prep.py` expects it at `out/adni1296_manifest_subject_folds.csv`.
Images come from ADNI and are not included (ADNI Data Use Agreement). Obtain access at https://adni.loni.usc.edu and place the 5-class slice folders under `data/Alzheimers-ADNI/{train,test}/Final <CLASS> JPEG/`. Paths at the top of each script may need adjusting.
Reproduce
```
pip install tensorflow scikit-learn scipy pillow
python prep.py && python features.py && python evaluate.py && python stats.py
python cnn.py image 30 && python cnn_bal.py subject 30
```
ImageNet weights: Keras legacy releases at https://github.com/fchollet/deep-learning-models/releases
Key finding
Subject-level macro-F1 (best model, DenseNet121): 0.283 [95% CI 0.259–0.308]; chance = 0.20. Augmentation before splitting inflates it to 0.685.
