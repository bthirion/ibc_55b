"""
Experiment on functional fingerprinting: what brain regions 
respond like a target voxel ?


export PYTHONPATH=$PYTHONPATH:/volatile/thirion/mygit/ibc_analysis_pipeline/ibc_main/processing
"""
import os
import numpy as np
import glob
from joblib import Memory, Parallel
from make_results_db import (
    make_db, resample_images, average_anat, LABELS, horizontal_fingerprint)
from nilearn.input_data import NiftiSpheresMasker, NiftiMasker
import nibabel as nib
from nilearn import plotting
import pandas as pd

from nilearn.image import resample_img, math_img, smooth_img, threshold_img, swap_img_hemispheres
from nilearn.regions import connected_regions
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils_group_analysis import one_sample_test
from sklearn import svm
from sklearn.cross_validation import cross_val_score, ShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.dummy import DummyClassifier

def make_roi_mask(seed, ref_affine, fwhm=5):
    """ """
    if len(seed) == 3:
        seed = np.hstack((seed, 1))

    pos = np.dot(np.linalg.inv(ref_affine), seed).astype(np.int)[:3]
    seed_mask = np.zeros(ref_shape)
    seed_mask[pos[0], pos[1], pos[2]] = 1.
    seed_img = nib.Nifti1Image(seed_mask, ref_affine)
    seed_img = smooth_img(seed_img, fwhm)
    seed_img = math_img('img > 1.e-6', img=seed_img)
    return seed_img


# caching
cache = '/neurospin/tmp/bthirion'
mem = Memory(cachedir=cache, verbose=0)

# output directory
write_dir = '/neurospin/tmp/bthirion'
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# Access to the data
data_dir =  '/neurospin/ibc/smooth_derivatives/group'
df = make_db('/neurospin/ibc/smooth_derivatives')
df = df.sort_values(by=['subject', 'task', 'contrast'])
mask_gm = nib.load(os.path.join(data_dir, 'gm_mask.nii.gz'))
mean_anat = average_anat(df)
ref_img = mask_gm
ref_affine, ref_shape = ref_img.affine, ref_img.shape

import json
ordered_contrasts = json.load(open('ordered_contrasts.json', 'r'))['contrasts']
n_contrasts = len(ordered_contrasts)

paths = {}
subjects = list(df.subject.unique())
for subject in subjects:
    paths_ = []
    for contrast in ordered_contrasts:
        contrast_mask = (df.contrast.values == contrast) * (df.subject.values == subject)
        paths_.append(df.path[contrast_mask].values[-1])
    paths[subject] = paths_

roi_name = '55b'
seed = [-50, 0, 47]
if 1:
    # step 1: extract signal from the region
    roi_mask = make_roi_mask(np.array(seed), ref_affine, fwhm=3)
    roi_mask = math_img('im1 * im2', im1=roi_mask, im2=mask_gm)
    roi_mask.to_filename('roi_pos.nii.gz')
    roi_masker = NiftiMasker(mask_img=roi_mask, memory=cache, smoothing_fwhm=None).fit()

    X_pos = np.vstack([(roi_masker.transform(paths[subject]).T) for subject in subjects])
    n_pos = X_pos.shape[0]
    labels = list(np.repeat(subjects, n_pos / 12))

    # step 2: extract signals from other regions
    roi_mask = make_roi_mask(np.array(seed), ref_affine)
    large_roi_mask = make_roi_mask(np.array(seed), ref_affine, fwhm=6)
    roi_mask = math_img('(1 - im1) * im2 * im3', im1=roi_mask,
                        im2=large_roi_mask, im3=mask_gm)
    roi_mask.to_filename('roi_neg.nii.gz')
    roi_masker = NiftiMasker(mask_img=roi_mask, memory=cache, smoothing_fwhm=None).fit()

    X_neg = np.vstack([(roi_masker.transform(paths[subject]).T) for subject in subjects])
    n_neg = X_neg.shape[0]
    labels += list(np.repeat(subjects, n_neg / 12))
else:
    # step 1: extract signal from the region
    roi_mask = resample_img('Area55b/Area55b_Left.nii', ref_affine, ref_shape)
    roi_mask = math_img('(im1 > .5) * im2', im1=roi_mask, im2=mask_gm)
    roi_masker = NiftiMasker(mask_img=roi_mask, memory=cache, smoothing_fwhm=None,
                             standardize=True).fit()
    X_pos = np.vstack([(roi_masker.transform(paths[subject]).T) for subject in subjects])
    n_pos = X_pos.shape[0]
    labels = list(np.repeat(subjects, n_pos / 12))
    # step 2: extract signals from other regions
    conf_mask = resample_img('Area55b/SaccadesMetaAnalysis.nii', ref_affine, ref_shape)
    roi_mask = math_img('(im1 > .5) * im2', im1=conf_mask, im2=mask_gm)
    roi_masker = NiftiMasker(mask_img=roi_mask, memory=cache, smoothing_fwhm=None,
                             standardize=True).fit()

    X_neg = np.vstack([(roi_masker.transform(paths[subject]).T) for subject in subjects])
    n_neg = X_neg.shape[0]
    labels += list(np.repeat(subjects, n_neg / 12))
    
    
# step 3: discriminative analysis
X = np.vstack((X_pos, X_neg))
X = StandardScaler().fit_transform(X)
y = np.hstack((np.ones(n_pos), -np.ones(n_neg)))
# cv = ShuffleSplit(y.size, 5)
cv = LeaveOneGroupOut()

clf = DummyClassifier()
scores = cross_val_score(clf, X, y, cv=cv.split(X, y, labels))
print(roi_name, 'Dummy:', scores.mean(),)

clf = ExtraTreesClassifier(n_estimators=250, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv.split(X, y, labels))
importances = clf.fit(X, y).feature_importances_
print(roi_name, 'RF:', scores.mean(),)

clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=cv.split(X, y, labels))
print(roi_name, 'Linear SVM:', scores.mean())
coefs = np.ravel(clf.fit(X, y).coef_)

labels_bottom = [LABELS[name][0] for name in ordered_contrasts]
labels_top = [LABELS[name][1] for name in ordered_contrasts]
output_file = os.path.join(write_dir, 'cc_%s.png' % roi_name.replace(' ', '_'))
horizontal_fingerprint(importances, seed, roi_name, labels_bottom, labels_top,
                       output_file, wc=True, nonneg=True)
plt.close()
output_file = os.path.join(write_dir, 'svc_%s.png' % roi_name.replace(' ', '_'))
horizontal_fingerprint(coefs, seed, roi_name, labels_bottom, labels_top,
                       output_file, wc=True)
plt.close()
    
plt.show()


# clf = svm.SVC(kernel='rbf', C=1) # achieves better performance !
# need scaling ?

