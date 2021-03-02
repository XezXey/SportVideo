# Libs
import cv2
from tqdm import tqdm
import os
from os import listdir, makedirs
from os.path import isfile, join, exists
import json
import numpy as np
import matplotlib.pyplot as plt
# import calibration
from SportVideo import calibration

# import utils
import utils.io as io
import utils.camera as cam_utils
import utils.trajectory as traj_utils
import utils.transform as transf_utils


class SportVideo:
  def __init__(self, path_to_video, out_name, field_img_path, args):
    image_extension = ['jpg', 'png']

    # Intialize Path
    self.path_to_video = path_to_video
    self.video_name = [vid_path.split('/')[-1].split('.')[0] for vid_path in self.path_to_video]
    self.out_name = out_name
    self.path_to_calibration = join(os.path.dirname(os.path.dirname(__file__)), 'CalibrationFile', out_name)
    self.extract_log = "{}/extract_log.txt".format(self.path_to_calibration)
    self.images_path = []
    self.field_img_path = field_img_path
    # Intialize the folder
    for sub_folder in self.video_name:
      self.images_path.append("{}/{}".format(self.path_to_calibration, sub_folder))
      self.intiailize_folder(path="{}/{}".format(self.path_to_calibration, sub_folder))

    # Initialize variable
    self.args = args

    # Extract frame
    if isfile(self.extract_log):
      print("[#] Found extracted frames.")
    else:
      self.extract_frame()

    # Load images
    self.frame_basenames = {self.video_name[0]:[], self.video_name[1]:[]}
    self.frame_fullnames = {self.video_name[0]:[], self.video_name[1]:[]}
    self.mask_fullnames = {self.video_name[0]:[], self.video_name[1]:[]}
    for idx, vid_name in enumerate(self.video_name):
      # Path of images
      self.frame_basenames[vid_name] = [f for f in listdir(self.images_path[idx])]
      self.frame_fullnames[vid_name] = [join(self.path_to_calibration, vid_name, f) for f in self.frame_basenames[vid_name]]
      self.frame_basenames[vid_name] = [f[:-4] for f in self.frame_basenames[vid_name]]
      self.mask_fullnames[vid_name] = [join(self.path_to_calibration, '{}_detectron'.format(vid_name), '{}.png'.format(f)) for f in self.frame_basenames[vid_name]]
      # Sorting image indices
      self.frame_basenames[vid_name].sort()
      self.frame_fullnames[vid_name].sort()
      self.mask_fullnames[vid_name].sort()

    frame = self.get_frame(self.video_name[0], 0)
    self.h, self.w = frame.shape[:2]
    self.n_frame = len(self.frame_basenames[vid_name])

  def extract_frame(self,):
    for idx, vid_path in enumerate(self.path_to_video):
      print("Video {} - {} : Processing...".format(idx, self.video_name[idx]))
      cap = cv2.VideoCapture(vid_path)
      counter = 0
      while(cap.isOpened()):
        # Display frame-by-frame
        ret, frame = cap.read()
        if ret == False:
          break
        else:
          cv2.imwrite("{}/{}/{}.jpg".format(self.path_to_calibration, self.video_name[idx], str(counter).zfill(5)), frame)
          counter += 1
      print("===> Done with {} frames".format(counter))
    with open(self.extract_log, 'w') as file:
      file.write('Success')

  def intiailize_folder(self, path):
    if not exists(path):
      makedirs(path)

  def calibrate_camera(self):
    self.intiailize_folder(join(self.path_to_calibration, 'calib'))
    calib_file = join(self.path_to_calibration, 'calib', 'calib_manual.npy*')
    if exists(calib_file):
      print("[#] Already Calibrated")
      exit()
    else:
      # Get mask and frame for calibrate
      img1 = self.get_frame(self.video_name[0], 0)
      img2 = self.get_frame(self.video_name[1], 0, flip=True)
      mask1 = self.get_mask_from_detectron(self.video_name[0], 0)
      mask2 = self.get_mask_from_detectron(self.video_name[1], 0, flip=True)
      A, R, T, label_points = calibration.calibrate_by_click(img1=img1, img2=img2, mask1=mask1, mask2=mask2, field_img_path=self.field_img_path)

      if (len(A) != 2) or (len(R) != 2) or (len(T) != 2):
        print("[#] Calibration Error : [None] in calibration")
        exit()
      else:
        self.write_calibs(A=A, R=R, T=T, points=label_points, suffix='manual')

    if (len(A) != 2) or (len(R) != 2) or (len(T) != 2):
      print("[#] Calibration Error")
      exit()

    elif self.args.ransac > 0:
      # Get finale A, R, T from ransac
      for i in range(self.args.ransac):
        print("[#] Refinement A, R, T")
        idx = np.random.randint(low=1, high=self.n_frame)
        img = self.get_frame(self.video_name[1], 0)
        mask = self.get_mask_from_detectron(self.video_name[0], 0)
        # A, R, T, __ = calibration.calibrate_from_initialization(img=, mask=, A_init=A, R_init=R, T_init=T, vis)
        print("===> A:", A)
        print("===> R:", R)
        print("===> T:", T)

      self.write_calibs(A=A, R=R, T=T, points=label_points, suffix='ransac')

  def get_frame(self, vid_name, frame_number, dtype=np.float32, image_type='rgb', sfactor=1.0, flip=False):
    print("[#] Gather Frame : ", vid_name)
    return io.imread(self.frame_fullnames[vid_name][frame_number], dtype=dtype, sfactor=sfactor, image_type=image_type, flip=flip)

  def get_mask_from_detectron(self, vid_name, frame_number, dtype=np.float32, image_type='rgb', sfactor=1.0, flip=False):
    print("[#] Gather Detectron : ", vid_name)
    return io.imread(self.mask_fullnames[vid_name][frame_number], dtype=dtype, sfactor=sfactor, image_type=image_type, flip=flip)[..., 0]

  def get_frame_index(self, frame_name):
    print("[#] Gather Frame Index : ", frame_name)
    return self.frame_basenames.index(frame_name)

  def write_calibs(self, A, R, T, points, suffix):
    print("[#] Saving calibration dict : ", suffix)
    calibs_dict_npy = {self.video_name[0]:{'A':None, 'R':None, 'T':None, 'P2d':None, 'P3d':None, 'H':None, 'W':None},
                   self.video_name[1]:{'A':None, 'R':None, 'T':None, 'P2d':None, 'P3d':None, 'H':None, 'W':None}}
    for i, vid_name in enumerate(self.video_name):
      calibs_dict[vid_name]['A'] = A[i].tolist()
      calibs_dict[vid_name]['R'] = R[i].tolist()
      calibs_dict[vid_name]['T'] = T[i].tolist()
      calibs_dict[vid_name]['P2d'] = points[i].tolist()
      calibs_dict[vid_name]['P3d'] = points[-1].tolist()
      calibs_dict[vid_name]['H'] = self.h
      calibs_dict[vid_name]['W'] = self.w

    # .npy format (For extracting 3d trajectory)
    calib_npy = join(self.path_to_calibration, 'calib', 'calib_{}'.format(suffix))
    np.save(calib_npy, calibs_dict)

  def load_calibs(self, suffix):
    print("[#] Loading calibration dict : ", suffix)
    calib = join(self.path_to_calibration, 'calib', 'calib_{}.npy'.format(suffix))
    calibration = np.load(calib, allow_pickle=True).item()
    return calibration

  def visualize_calibration(self, trajectory_3d=None):
    print("[#] Visualizing calibration")
    calibs = self.load_calibs(suffix='manual')
    cam1 = calibs[self.video_name[0]]
    cam2 = calibs[self.video_name[1]]
    calibration.visualize_calibration(cam1=cam1, cam2=cam2, trajectory_3d=trajectory_3d)

  def extract_3d_tracking(self, tracking_1, tracking_2, flip=True):
    print("[#] Extracting 3D tracking by triangulation")
    calibs = self.load_calibs(suffix='manual')
    cam1 = calibs[self.video_name[0]]
    cam1_obj = cam_utils.Camera(name='cam1', A=np.array(cam1['A']), R=np.array(cam1['R']), T=np.array(cam1['T']), h=cam1['H'], w=cam1['W'])
    cam2 = calibs[self.video_name[1]]
    cam2_obj = cam_utils.Camera(name='cam2', A=np.array(cam2['A']), R=np.array(cam2['R']), T=np.array(cam2['T']), h=cam2['H'], w=cam2['W'])

    tracking_1 = io.read_tracking(full_name=tracking_1, ext=tracking_1[-3:])
    tracking_2 = io.read_tracking(full_name=tracking_2, ext=tracking_2[-3:])
    all_tracking = np.concatenate((tracking_1, tracking_2), axis=1)
    # self.visualize_2d_trajectory(tracking_1, tracking_2, all_tracking)
    tracking_dict = traj_utils.filter_points(all_tracking)
    tracking = tracking_dict['all']['tracking_f']
    if flip:
      tracking[:, 2] = self.w - tracking[:, 2]    # Inverse the tracking for 2nd video

    trajectory_3d = cam_utils.triangulate_points(cam1=cam1_obj, cam2=cam2_obj, tracking=tracking)
    # Output 3D trajectory is in Camera 1 coordinates, so we move it to world coordinates
    trajectory_3d[:, [1, 2]] *= -1 # Invert y-z axis
    trajectory_3d = transf_utils.transform_3d(m=cam1_obj.to_opengl()[0], pts=trajectory_3d[:, :-1], inv=True)
    return trajectory_3d, tracking_dict

  def convert_calibration_to_unity(self):
    print("[#] Convert the calibration to Unity (Json format)")
    calibs = self.load_calibs(suffix='manual')
    cam1 = calibs[self.video_name[0]]
    cam1_obj = cam_utils.Camera(name='cam1', A=np.array(cam1['A']), R=np.array(cam1['R']), T=np.array(cam1['T']), h=cam1['H'], w=cam1['W'])
    cam2 = calibs[self.video_name[1]]
    cam2_obj = cam_utils.Camera(name='cam2', A=np.array(cam2['A']), R=np.array(cam2['R']), T=np.array(cam2['T']), h=cam2['H'], w=cam2['W'])

    calibs_json = {"firstCam":{"width":None, "height":None, "projectionMatrix":None, "worldToCameraMatrix":None},
                   "secondCam":{"width":None, "height":None, "projectionMatrix":None, "worldToCameraMatrix":None}}

    cam_obj = [cam1_obj, cam2_obj]
    cam_list = ["firstCam", "secondCam"]
    for i, vid_name in enumerate(cam_list):
      calibs_json[vid_name]['width'] = cam_obj[i].width
      calibs_json[vid_name]['height'] = cam_obj[i].height
      calibs_json[vid_name]['projectionMatrix'] = cam_obj[i].to_opencv()[1].flatten().tolist() # Unity need fx, fy, cy, cz from opencv component
      calibs_json[vid_name]['worldToCameraMatrix'] = cam_obj[i].to_opengl()[0].flatten().tolist()  # Unity need Extrinsic(worldtocameramatrix) in OpenGL convention

    calib_unity_out = join(self.path_to_calibration, 'calib', 'calib_manual_unity.json')
    with open(calib_unity_out, 'w') as outfile:
      json.dump(calibs_json, outfile)

  def visualize_2d_trajectory(self, tracking_1, tracking_2, all_tracking):
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(tracking_1[:, 0], self.h - tracking_1[:, 1], '-o')
    axes[0, 0].set_title('cam_1')
    axes[0, 0].set_xlim(-100, self.w + 100)
    axes[0, 0].set_ylim(-100, self.h + 100)
    axes[0, 1].plot(tracking_2[:, 0], self.h - tracking_2[:, 1], '-o')
    axes[0, 1].set_title('cam_2')
    axes[0, 1].set_xlim(-100, self.w + 100)
    axes[0, 1].set_ylim(-100, self.h + 100)
    all_tracking = traj_utils.interpolation(all_tracking)
    axes[1, 0].plot(all_tracking[:, 0], self.h - all_tracking[:, 1], '-o')
    axes[1, 0].set_title('cam_1 & cam_2')
    axes[1, 0].set_xlim(-100, self.w + 100)
    axes[1, 0].set_ylim(-100, self.h + 100)
    axes[1, 1].plot(all_tracking[:, 2], self.h - all_tracking[:, 3], '-o')
    axes[1, 1].set_title('cam_1 & cam_2')
    axes[1, 1].set_xlim(-100, self.w + 100)
    axes[1, 1].set_ylim(-100, self.h + 100)
    plt.show()

  def split_trajectory(self, trajectory_3d, tracking_dict):
    '''
    TBD : Split trajectory function later
    Input : Trajectory_3d with tracking pair
    Output : Splitted trajectory_3d with tracking pair
      - Output shape : (n_trajectory, seq_len, features) ===> e.g. (1, )
    '''
    # This only the mock up version = Use every points w/o a segmentation.
    idx_trajectory_3d = tracking_dict['all']['tracking_f'][..., [-1]]

    x = trajectory_3d[:, 0]
    y = trajectory_3d[:, 1]
    z = trajectory_3d[:, 2]
    diff_x = np.diff(x, axis=0)
    diff_y = np.diff(y, axis=0)
    diff_z = np.diff(z, axis=0)
    threshold = 5
    split_x = diff_x > threshold
    split_y = diff_y > threshold
    split_z = diff_z > threshold

    split_idx = np.logical_or(np.logical_or(split_x, split_y), split_z)
    split_idx = np.where(split_idx == True)
    split_idx = np.concatenate((np.zeros(1), split_idx[0] + 1, np.array([trajectory_3d.shape[0]])))

    trajectory_split = []
    tracking_split = []
    for idx in range(len(split_idx)-1):
      start = int(split_idx[idx])
      stop = int(split_idx[idx+1])
      trajectory_split.append(trajectory_3d[start:stop, :])
      tracking_split.append(tracking[start:stop, :])
      print(idx_trajectory_3d[start:stop])

    trajectory_split = np.array(trajectory_split)
    tracking_split = np.array(tracking_split)
    traj_and_track = [np.concatenate((trajectory_split[i], tracking_split[i]), axis=-1) for i in range(len(trajectory_split))]
    traj_and_track = np.array(traj_and_track)
    print(traj_and_track)
    exit()
    return traj_and_track

  def preprocess_save_to_npy(self, trajectory):
    '''
    Input shape = (n_trajectory, seq_len, features) ===> e.g. (1, )
    Features cols = (x, y, z, u_c1, v_c1, u_c2, v_c2)
    '''

    print("[#] Saving 3D tracking by triangulation")
    calibs = self.load_calibs(suffix='manual')
    cam1 = calibs[self.video_name[0]]
    cam1_obj = cam_utils.Camera(name='cam1', A=np.array(cam1['A']), R=np.array(cam1['R']), T=np.array(cam1['T']), h=cam1['H'], w=cam1['W'])
    cam2 = calibs[self.video_name[1]]
    cam2_obj = cam_utils.Camera(name='cam2', A=np.array(cam2['A']), R=np.array(cam2['R']), T=np.array(cam2['T']), h=cam2['H'], w=cam2['W'])

    cam_list = [cam1_obj, cam2_obj]
    trajectory_list = [[], []]
    for i in range(trajectory.shape[0]):
      # Unity convention
      trajectory[i][:, [2]] *= -1
      x, y, z = trajectory[i][:, [0]], trajectory[i][:, [1]], trajectory[i][:, [2]]
      for j, cam in enumerate(cam_list):
        if not self.args.use_tracking:
          # Do a projection if tracking is not used as an input
          trajectory_npy = []
          u, v, d = transf_utils.project(cam=cam, pts=trajectory[i][..., [0, 1, 2]])
        else:
          trajectory_cam = transf_utils.transform_3d(m=cam.to_unity()[0], pts=trajectory[i][..., [0, 1, 2]], inv=False)
          d = -trajectory_cam[:, [2]]   # Depth direction should be positive for unity
          if j == 0:
            u, v = trajectory[i][:, [3]], trajectory[i][:, [4]]
          if j == 1:
            u, v = trajectory[i][:, [5]], trajectory[i][:, [6]]

          # For raw tracking, unity coordinates start from (0, 0) at the bottom left, so I need to remove with the height
          v = cam.height - v

        dummy_eot = np.zeros(shape=(x.shape))
        saving_trajectory = np.concatenate((x, y, z, u, v, d, dummy_eot), axis=1)
        saving_trajectory = np.vstack((saving_trajectory[0, :], np.diff(saving_trajectory, axis=0)))
        trajectory_list[j].append(saving_trajectory)

    for i, trajectory_cam in enumerate(trajectory_list):
      trajectory_cam = np.array(trajectory_cam)
      print("Camera {} : {}".format(i+1, trajectory_cam.shape))
      # np.save(trajectory_cam)
      # .npy format (For extracting 3d trajectory)
      self.intiailize_folder(path="{}/{}".format(self.path_to_calibration, 'preprocessed_npy'))
      preprocessed_npy = join(self.path_to_calibration, 'preprocessed_npy', 'trajectory_{}'.format(self.video_name[i]))
      preprocessed_npy = join(self.path_to_calibration, 'preprocessed_npy', 'MixedTrajectory_Trial{}'.format(self.video_name[i][-1]))
      np.save(preprocessed_npy, trajectory_cam)
