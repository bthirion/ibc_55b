"""
Analysis of the Hipocampal signal with IBC dataset
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from joblib import Memory
from nilearn.image import resample_to_img, math_img
from nilearn.plotting import plot_stat_map, show
from nilearn.input_data import NiftiMasker
from ibc_public.utils_data import (
    data_parser, SMOOTH_DERIVATIVES, SUBJECTS, LABELS, CONTRASTS,
    CONDITIONS, THREE_MM)
import ibc_public

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

# Access to the data
subject_list = SUBJECTS
task_list = ['archi_standard', 'archi_spatial', 'archi_social',
             'archi_emotional', 'hcp_language', 'hcp_social', 'hcp_gambling',
             'hcp_motor', 'hcp_emotion', 'hcp_relational', 'hcp_wm',
             'rsvp_language']
df = data_parser(derivatives=SMOOTH_DERIVATIVES, subject_list=SUBJECTS,
                 conditions=CONDITIONS, task_list=task_list)
df = df[df.acquisition == 'ffx']
conditions = df[df.modality == 'bold'].contrast.unique()
n_conditions = len(conditions)

masker = NiftiMasker(mask_img=mask_total, memory=mem).fit()
data = []
for subject in subject_list:
    imgs = df[df.subject == subject].path.values
    X = masker.transform(imgs)
    data.append(X)

for i, X in enumerate(data):
    if i == 0:
        energy = np.mean(X ** 2, 0)
    else:
        energy += np.mean(X ** 2, 0)

energy /= len(subject_list)
energy_img = masker.inverse_transform(energy)
plot_stat_map(energy_img)
plt.show(block=False)
