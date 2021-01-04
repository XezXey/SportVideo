import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd

def visualize_video(vid_list, track_list=None):
  cap_list = []
  ncol = 3
  dsize = (600, 400, 3)
  frame_counter = 0
  if track_list is not None:
    tracking = read_tracking(track_list)
  for v in vid_list:
    cap_list.append(cv2.VideoCapture(v))

  while(is_cap_open(cap_list)):
    # Display frame-by-frame
    frames = []
    for idx, cap in enumerate(cap_list):
      ret, frame = cap.read()
      if ret == False:
        frame = np.zeros(dsize)
      else:
        src_size = frame.shape
        # frame = cv2.flip(frame, 1)
        if track_list is not None:
          center = tracking[idx].loc[frame_counter, [' x', ' y']].values
          if not any('-' in c for c in center):
            center = tuple(map(int, center))
            center = np.array(center)
            # center[0] = src_size[1] - center[0]
            center = tuple(center)
            cv2.circle(frame, center, 4, (0, 255, 0), 3)
        frame = cv2.resize(src=frame, dsize=dsize[:2])
      frames.append(frame)

    if len(frames) % ncol != 0:
      n_pad = ncol - (len(frame) % ncol)
      for i in range(n_pad):
        frames.append(np.zeros(shape=frames[0].shape))

    frame_ptr = 0
    n_frame = len(frames)
    disp_frame = []
    while(n_frame > 0):
      disp_frame.append(np.hstack(frames[frame_ptr:frame_ptr+ncol]))
      frame_ptr += 3
      n_frame -= 3

    disp_frame = np.vstack(disp_frame)
    cv2.imshow('Frame', disp_frame)
    frame_counter += 1

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

def is_cap_open(cap_list):
  cap_status = []
  for cap in cap_list:
    cap_status.append(cap.isOpened())

  return False not in cap_status


def read_tracking(track_list):
  tracking = []
  for track in track_list:
    print(track)
    if '.csv' in track:
      tracking.append(pd.read_csv(track))

  return tracking


