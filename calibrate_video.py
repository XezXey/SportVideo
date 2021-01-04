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
args = parser.parse_args()


if __name__ == '__main__':
  sv = SportVideo.SportVideo(path_to_video=[args.first_vid, args.second_vid], out_name=args.out_name, field_img_path=args.field_template, args=args)
  calib = join(sv.path_to_calibration, 'calib', 'calib_manual.npy')
  if exists(calib):
    sv.visualize_calibration()
  else:
    sv.calibrate_camera()
