import numpy as np
import cv2
import os
from . import transform
import matplotlib.pyplot as plt

def filter_points(all_tracking, method=None):
  '''
  tracking_f = remove row with np.nan (Tracking filtered)
  tracking_nf = Tracking no filtered
  Masking = 1 = visible, np.nan = no tracking
  '''
  # plt.plot(all_tracking[:, 0], all_tracking[:, 1])
  # plt.show()
  # plt.plot(all_tracking[:, 2], all_tracking[:, 3])
  # plt.show()
  # Add index reference for further match
  matching_idx = np.arange(all_tracking.shape[0]).reshape(-1, 1)
  all_tracking = np.concatenate((all_tracking, matching_idx), axis=-1)
  if method is None:
    # Masking "False" to all rows and replace np.nan element
    # All Camera
    masking_all_cam = (~np.isnan(all_tracking)).all(axis=1)
    # Replace "False" with np.nan
    masking_all_cam = np.where(masking_all_cam == False, np.nan, masking_all_cam).reshape(-1, 1)
    # broadcast np.nan masking to all row for filter it out
    tracking_all_cam = all_tracking * np.repeat(masking_all_cam, all_tracking.shape[-1], axis=1)
    # Choose only point that not np.nan
    tracking_all_cam = all_tracking[~np.isnan(tracking_all_cam).any(axis=1)]

    # Camera 1
    all_tracking_cam1 = all_tracking[..., [0, 1]]
    masking_cam1 = (~np.isnan(all_tracking_cam1)).all(axis=1)
    # Replace "False" with np.nan
    masking_cam1 = np.where(masking_cam1 == False, np.nan, masking_cam1).reshape(-1, 1)
    # broadcast np.nan masking to all row for filter it out
    tracking_cam1 = all_tracking_cam1 * np.repeat(masking_cam1, 2, axis=1)
    # Choose only point that not np.nan
    tracking_cam1 = all_tracking_cam1[~np.isnan(tracking_cam1).any(axis=1)]

    # Camera 2
    all_tracking_cam2 = all_tracking[..., [2, 3]]
    masking_cam2 = (~np.isnan(all_tracking_cam2)).all(axis=1)
    # Replace "False" with np.nan
    masking_cam2 = np.where(masking_cam2 == False, np.nan, masking_cam2).reshape(-1, 1)
    # broadcast np.nan masking to all row for filter it out
    tracking_cam2 = all_tracking_cam2 * np.repeat(masking_cam2, 2, axis=1)
    # Choose only point that not np.nan
    tracking_cam2 = all_tracking_cam2[~np.isnan(tracking_cam2).any(axis=1)]


  tracking_dict = {'all':{'mask':masking_all_cam, 'tracking_f':tracking_all_cam, 'tracking_nf':all_tracking},
                   'cam1':{'mask':masking_cam1, 'tracking_f':tracking_cam1, 'tracking_nf':all_tracking[..., [0, 1, -1]]},
                   'cam2':{'mask':masking_cam2, 'tracking_f':tracking_cam2, 'tracking_nf':all_tracking[..., [2, 3, -1]]},}

  return tracking_dict
