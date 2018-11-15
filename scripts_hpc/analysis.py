"""
Analysis of the Hipocampal signal with IBC dataset
"""
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
import nibabel as nib
from joblib import Memory
from nilearn.image import resample_to_img, math_img
from nilearn.plotting import plot_stat_map, show, plot_roi
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, SUBJECTS, LABELS, CONTRASTS,
    CONDITIONS, THREE_MM, DERIVATIVES)
import ibc_public

SUBJECTS = [subject for subject in SUBJECTS if subject not in
            ['sub-02', 'sub-08', 'sub-15']]

if 1:
    cache = '/neurospin/tmp/bthirion'
else:
    cache = '/storage/tompouce/bthirion/'

# caching
mem = Memory(cachedir=cache, verbose=0, bytes_limit=4.e9)

# output directory
write_dir = cache
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

# mask of the grey  matter across subjects
package_directory = os.path.dirname(
    os.path.abspath(ibc_public.utils_data.__file__))
mask_gm = os.path.join(
    package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
mask_left_hip = 'data/Consensus_L_dil_msk.nii.gz'
mask_right_hip = 'data/Consensus_R_dil_msk.nii.gz'
mask_left = math_img('img > .5', img=resample_to_img(mask_left_hip, mask_gm))
mask_right = math_img('img > .5', img=resample_to_img(mask_right_hip, mask_gm))
mask_total = math_img('im1 + im2', im1=mask_left, im2=mask_right)

side = 'right'
if side == 'left':
    mask_img = mask_left
elif side == 'right':
    mask_img = mask_right
elif side == 'both':
    mask_img = mask_total

# Access to the data
subject_list = SUBJECTS
task_list = ['archi_standard', 'archi_spatial', 'archi_social',
             'archi_emotional', 'hcp_language', 'hcp_social', 'hcp_gambling',
             'hcp_motor', 'hcp_emotion', 'hcp_relational', 'hcp_wm',
             'rsvp_language']
df = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=SUBJECTS,
                 conditions=CONTRASTS, task_list=task_list)
df = df[df.acquisition == 'ffx']
conditions = df[df.modality == 'bold'].contrast.unique()
n_conditions = len(conditions)

masker = NiftiMasker(mask_img=mask_img, memory=mem).fit()
data = []
for subject in subject_list:
    imgs = df[df.subject == subject].path.values
    X = masker.transform(imgs).T
    data.append(X)

#############################################################################
# energy analysis
for i, X in enumerate(data):
    if i == 0:
        energy = np.mean(X ** 2, 1)
    else:
        energy += np.mean(X ** 2, 1)

energy /= len(subject_list)
energy_img = masker.inverse_transform(energy)
plot_stat_map(energy_img)

#############################################################################
# dimensionality analysis
n_components = range(0, 20, 3)


def compute_scores(X):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()
    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages}, cv=5)
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X, cv=5))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X, cv=5))

n_clusters = 3
mean_data = np.hstack(data)
xyz = np.rollaxis(np.indices(mask_img.shape), 0, 4)
xyz_image = nib.Nifti1Image(xyz, mask_img.affine)

"""
from nilearn.regions import Parcellations

ward = Parcellations(method='ward', n_parcels=n_clusters, mask=mask_img,
                     standardize=False, memory='nilearn_cache', memory_level=1,
                     smoothing_fwhm=5, verbose=1)
ward.fit(xyz_image)  # df.path.values)
"""
"""
ward.fit(df.path.values[np.random.rand(len(df.path.values)) > .95])
cluster_img = ward.labels_img_
clusters = cluster_img.get_data()[mask_img.get_data() > 0] - 1
"""
rng = check_random_state(42)
mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters,
                      batch_size=100,
                      n_init=10, max_no_improvement=10, verbose=0,
                      random_state=0).fit(xyz[mask_img.get_data() > 0])
clusters = mbk.labels_
cluster_img = masker.inverse_transform(clusters + 1)

_, counts = np.unique(clusters, return_counts=True)
print(counts)

plot_roi(cluster_img, cmap=matplotlib.cm.cool, threshold=0, vmax=3)
plot_roi(cluster_img, cmap=matplotlib.cm.cool, threshold=0, vmax=3,
         output_file=os.path.join(write_dir, 'clusters_%s.png' % side))


for X, subject in zip(data, subject_list):
    plt.figure(figsize=(8, 3))
    for label, color in zip(np.unique(clusters), ['c', 'purple', 'pink']):
        plt.subplot(1, 3, label + 1)
        pca_scores, fa_scores = compute_scores(X[clusters == label])
        print(np.sum(clusters == label), pca_scores, fa_scores)
        plt.plot(n_components, pca_scores, color, label='PCA scores')
        plt.plot(n_components, fa_scores, color, linestyle=':',
                 label='FA scores')
    plt.legend()
    plt.savefig(os.path.join(write_dir, 'dimension_%s_%s.png' %
                (subject, side)))

"""
for X in data[:1]:

    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa,
                linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')

plt.show(block=False)
"""

# resting-state connectivity analysis
from ibc_public.utils_data import get_subject_session


rs_masker = NiftiMasker(
    mask_img=mask_gm, memory=mem, standardize=True,
    detrend=True, low_pass=.1, high_pass=.01, t_r=.76).fit()
roi_masker = NiftiLabelsMasker(
    labels_img=cluster_img, memory=mem, standardize=True,
    detrend=True, low_pass=.1, high_pass=.01, t_r=.76).fit()


def compute_connectivity(subject, session, acq):
    rs_fmri = os.path.join(
        DERIVATIVES, subject, session, 'func',
        'wrdc%s_%s_task-RestingState_acq-%s_bold.nii.gz' %
        (subject, session, acq))
    X = rs_masker.transform(rs_fmri)
    seeds = roi_masker.transform(rs_fmri)
    normalization = np.sqrt(X.shape[0])
    X = (X.T - X.mean(1)).T
    X = (X.T / X.std(1)).T / normalization
    seeds = (seeds.T - seeds.mean(1)).T
    seeds = (seeds.T / seeds.std(1)).T / normalization
    correlation = np.dot(seeds.T, X)
    return correlation


q = 0
for subject in SUBJECTS:
    for task in ['mtt1', 'mtt2']:
        subject_session = np.array(get_subject_session(task))
        session = subject_session[subject_session[:, 0] == subject][0][1]
        for acq in ['ap', 'pa']:
            print(subject, session, acq)
            correlation = compute_connectivity(subject, session, acq)
        if q == 0:
            mean_correlation = correlation
        else:
            mean_correlation += correlation
        q += 1
mean_correlation /= q

cut_coords =  [3, -26, 2 ]
for i in range(3):
    correlation_img = rs_masker.inverse_transform(mean_correlation[i])
    plot_stat_map(correlation_img, cut_coords=cut_coords,
                  output_file=os.path.join(
                      write_dir, 'correlation_%s_%d.png' % (side, i)))
    correlation_img.to_filename(
        os.path.join(write_dir, 'correlation_%s_%d.nii.gz' % (side, i)))

plt.show(block=False)
