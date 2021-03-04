import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import glob

# Utils
from visualize import visualize

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', type=str, required=True)
parser.add_argument('--video_ext', type=str, required=True)
parser.add_argument('--tracking_path', type=str, default=None, required=True)
parser.add_argument('--tracking_ext', type=str, default=None, required=True)
args = parser.parse_args()

if __name__ == '__main__':
  vid_list = sorted(glob.glob('{}/*.{}'.format(args.video_path, args.video_ext)))
  if args.tracking_path is not None:
    track_list = sorted(glob.glob('{}/*.{}'.format(args.tracking_path, args.tracking_ext)))
  else:
    track_list = None

  visualize.visualize_video(vid_list, track_list)
