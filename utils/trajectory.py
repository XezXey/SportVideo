import numpy as np
import cv2
import os
from . import transform

def interpolation(all_tracking, method=None):
  if method is None:
    # Masking "False" to all rows with np.nan element
    masking = (~np.isnan(all_tracking)).all(axis=1)
    masking = np.where(masking == False, np.nan, masking).reshape(-1, 1)
    all_tracking = all_tracking * np.repeat(masking, all_tracking.shape[-1], axis=1)
    tracking = all_tracking[~np.isnan(all_tracking).any(axis=1)]

  return tracking
