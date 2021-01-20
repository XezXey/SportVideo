# Libs
import numpy as np
import os
from os import listdir
from os.path import isfile, join, exists
import glob
import cv2
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib
import argparse
# SportVideo class
import SportVideo

parser = argparse.ArgumentParser("Calibration & Stereo Calibration any 2 input videos together")
parser.add_argument('--first_vid', type=str, required=True)
parser.add_argument('--second_vid', type=str, required=True)
parser.add_argument('--field_template', type=str, required=True)
parser.add_argument('--out_name', type=str, required=True)
parser.add_argument('--ransac', type=int, default=0)
parser.add_argument('--ext', type=str, default='csv')
parser.add_argument('--use_tracking', action='store_true', default=False)
args = parser.parse_args()


if __name__ == '__main__':
  sv = SportVideo.SportVideo(path_to_video=[args.first_vid, args.second_vid], out_name=args.out_name, field_img_path=args.field_template, args=args)
  calib = join(sv.path_to_calibration, 'calib', 'calib_manual.npy')
  tracking_1 = args.first_vid[:-4] + '.{}'.format(args.ext)
  tracking_2 = args.second_vid[:-4] + '.{}'.format(args.ext)
  trajectory_3d, tracking = sv.extract_3d_tracking(tracking_1, tracking_2)
  # sv.visualize_calibration(trajectory_3d=trajectory_3d)
  trajectory = sv.split_trajectory(trajectory_3d=trajectory_3d, tracking=tracking)
  sv.preprocess_save_to_npy(trajectory)
