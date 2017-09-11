"""Script that tries to assess the behavior of the putative 55b are
in hyperalignment procedure

Author Bertrand Thirion 2017

export PYTHONPATH=$PYTHONPATH:/volatile/thirion/mygit/hugo-richard-M2/internship_project:/volatile/thirion/mygit/ibc_analysis_pipeline/ibc_main/data_paper_scripts:/volatile/thirion/mygit/ibc_analysis_pipeline/ibc_main/high_level_analysis_scripts
"""
import os
import numpy as np
from joblib import Memory
from data_utils import data_parser, CONTRASTS
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.image import math_img, resample_img
import nibabel as nib
from piecewise_alignment import PieceWiseAlignment
# from sklearn.linear_model import RidgeCV
from ridge_cv import RidgeCV
from utils_viz import print_fingerprint
from make_results_db import LABELS

if 0:
    ibc = '/neurospin/ibc'
    cache = '/neurospin/tmp/bthirion'
else:
    ibc = '/storage/store/data/ibc'
    cache = '/storage/tompouce/bthirion/'

    
DERIVATIVES = os.path.join(ibc, 'derivatives')
SMOOTH_DERIVATIVES = os.path.join(ibc, 'smooth_derivatives')


# caching

mem = Memory(cachedir=cache, verbose=0)

# output directory
write_dir = cache
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# Access to the data
data_dir =  os.path.join(SMOOTH_DERIVATIVES, 'group')
mask_gm = nib.load(os.path.join(data_dir, 'gm_mask.nii.gz'))
ref_affine, ref_shape = mask_gm.affine, mask_gm.shape

# df = make_db('/neurospin/ibc/smooth_derivatives')
df = data_parser(derivatives=SMOOTH_DERIVATIVES, conditions=CONTRASTS)
conditions = df.contrast[df.modality == 'bold'].unique()
n_conditions = len(conditions)

# Mask of the ROI
# intersect with GM mask
atlas = datasets.fetch_atlas_destrieux_2009()
# 29: left hemisphere
# 104 right hemisphere
roi_index = 29
roi_mask = math_img('im1 == %d' % roi_index, im1=atlas.maps)
roi_mask = resample_img(roi_mask, ref_affine, ref_shape, interpolation='nearest')
masker = NiftiMasker(mask_img=roi_mask, memory=mem).fit()

path_train = []
path_test = []
X_train = []
X_test = []
subjects = df.subject.unique()
for subject in subjects:
    spath = [df[df.acquisition == 'ap'][df.subject == subject][df.contrast == condition].path.values[-1] for condition in conditions]
    path_train.append(spath)
    X_train.append(masker.transform(spath).T)
    spath = [df[df.acquisition == 'pa'][df.subject == subject][df.contrast == condition].path.values[-1]for condition in conditions]
    path_test.append(spath)
    X_test.append(masker.transform(spath).T)

alphas = np.logspace(3., 5., 3)
n_jobs = 1
n_voxels = roi_mask.get_data().sum()
labels = np.zeros(n_voxels)

algo = PieceWiseAlignment(labels, masker=None, method=RidgeCV(alphas=alphas),
                          n_jobs=n_jobs)
algo_name = "ridgeCV"

def hyperalign(X_train, X_test, algo, algo_name, train_subject, test_subjects):
    Y_test = []
    for test_subject in test_subjects:
        delta_X =  X_train[train_subject] - X_train[test_subject]
        algo.fit(X_train[test_subject], delta_X)
        Y_test.append(X_test[test_subject] + algo.transform(X_test[test_subject]))
    return(Y_test)

train_subjects = subjects

train_subject = 3
test_subjects = range(12)
Y_test = hyperalign(X_train, X_test, algo, algo_name, train_subject, test_subjects)


def consistency_map(cube):
    """ return quasi-F map over the first dimension of the cube, averaged over last dimension"""
    return np.mean(one_sample(cube) ** 2, 1)

def one_sample(cube):
    """ return amrginal t maps over the first dimension of the cube"""
    return (np.mean(cube, 0) /  np.std(cube, 0) * np.sqrt(cube.shape[0] - 1))

def extract_blobs(x, masker, size_min=50):
    """ Extract blobs by watershed"""
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    from scipy import ndimage as ndi
    X = masker.inverse_transform(x).get_data()
    threshold = np.percentile(x, 80)
    local_maxi = peak_local_max(X,  indices=False)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-X, markers, mask=(X > threshold))
    ulabels, counts = np.unique(labels, return_counts=True)
    print(counts)
    for count_, label_ in zip(counts, ulabels):
        if count_ < size_min:
            labels[labels == label_] = 0
    return labels


consistency_X =  consistency_map(np.array(X_test))
filename = os.path.join(write_dir, 'consistency_X.nii.gz')
masker.inverse_transform(consistency_X).to_filename(filename)

consistency_Y =  consistency_map(np.array(Y_test))
filename = os.path.join(write_dir, 'consistency_Y.nii.gz')
masker.inverse_transform(consistency_Y).to_filename(filename)

extract_blobs(consistency_X, masker)
blobs = extract_blobs(consistency_Y, masker)
label_img = nib.Nifti1Image(blobs, masker.affine_)
label_img.to_filename(os.path.join(write_dir, 'blobs.nii.gz'))
blob_vector = blobs[(masker.mask_img_).get_data() > 0]

labels_bottom = [LABELS[name][0] for name in conditions]
labels_top = [LABELS[name][1] for name in conditions]
ulabels = np.unique(blobs)
Yt = one_sample(np.array(Y_test))
for label in ulabels:
    if label == 0:
        continue
    fingerprint = np.mean(Yt[blob_vector == label], 0)
    fname = os.path.join(write_dir, 'blob_%03d.png' % label)
    print_fingerprint(fingerprint, fname, labels_bottom, labels_top)

    
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(consistency_X, consistency_Y)
plt.plot(consistency_X, consistency_X, 'k')
plt.show()
