from __future__ import print_function
# Import libs
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import json
import re
import tqdm
import cv2
import pandas as pd
import math
pd.options.mode.chained_assignment = None  # default='warn'

parser = argparse.ArgumentParser(description='Preprocess and visualize the trajectory in real world')
parser.add_argument('--dataset_path', type=str, help='Specify path to dataset')
parser.add_argument('--process_trial_index', dest='process_trial_index', help='Process trial at given idx only', default=None)
parser.add_argument('--output_path', type=str, help='Specify output path to save dataset')
parser.add_argument('--unit', dest='unit', help='Scaling unit', required=True, type=str)
args = parser.parse_args()

def get_savepath(output_path, dataset_folder):
  if output_path == None:
    output_path = dataset_folder
  else:
    output_path = args.output_path
    if not os.path.exists(output_path):
      os.makedirs(output_path)
  return output_path

def get_depth(traj_df, cam_dict):
  '''
  This function will return the points in camera space and screen space
  - This projectionMatrix use in unity screen convention : (0, 0) is in the bottom-left
  '''
  worldToCameraMatrix = np.array(cam_dict['E_unity'])
  projectionMatrix = np.array(cam_dict['P_unity'])
  # Plane space
  world_space = np.hstack((traj_df[['ball_plane_x', 'ball_plane_y', 'ball_plane_z']].values, np.ones(traj_df.shape[0]).reshape(-1, 1)))
  # Uncomment this to test the particular points
  # WorldToCameraMatrix : Plane space -> Camera space (u*depth, v*depth, depth, 1)
  # In the unity depth is in the +z direction, so we need to negate the z direction that came from multiplied with extrinsic
  camera_space = world_space @ worldToCameraMatrix.T
  traj_df['ball_camera_x'] = camera_space[:, 0]
  traj_df['ball_camera_y'] = camera_space[:, 1]
  # projectionMatrix : Camera space -> NDC Space
  # In the unity, projection matrix didn't give the output as the screen space but it will in the NDC space(We'll see the world in range(-1, 1))
  # Then we need to unnormalized it to ge the screen space
  ndc_space = camera_space @ projectionMatrix.T
  traj_df['ball_camera_depth'] = ndc_space[:, 2]
  # Get the screen coordinates
  u = ((ndc_space[:, 0]/ndc_space[:, 2]) + 1) * (cam_dict['w']/2)
  v = ((ndc_space[:, 1]/ndc_space[:, 2]) + 1) * (cam_dict['h']/2)
  traj_df['ball_screen_unity_u_project'] = u
  traj_df['ball_screen_unity_v_project'] = v
  return traj_df

# RayCasting
def cast_ray(uv, I, E):
  '''
  Casting a ray given UV-coordinates, intrinisc, extrinsic
  Input : 
      - UV : 2d screen points in shape=(batch, seq_len, 2)
      - I : Intrinsic in shape=(batch, seq_len, 4, 4)
      - E : Extrinsic in shape=(batch, seq_len, 4, 4)
  Output :
      - ray : Ray direction (batch, seq_len, 3)
      (Can check with ray.x, ray.y, ray.z for debugging)
  '''
  #print("UV: : ", uv.shape)
  #print("I : ", I.shape)
  #print("E : ", E.shape)

  w = 1664.0
  h = 1088.0
  
  transf = I @ E
  transf_inv = np.linalg.inv(transf)

  u = ((uv[..., [0]] / w) * 2) - 1
  v = ((uv[..., [1]] / h) * 2) - 1

  ones = np.ones(u.shape)

  ndc = np.concatenate((u, v, ones, ones), axis=-1)
  ndc = np.expand_dims(ndc, axis=-1)

  ndc = transf_inv @ ndc

  cam_pos = np.linalg.inv(E)[..., 0:3, -1]
  ray = ndc[..., [0, 1, 2], -1] - cam_pos

  return ray

def ray_to_plane(E, ray):
  '''
  Find the intersection points on the plane given ray-vector and camera position
  Input :
      - cam_pos : camera position in shape=(batch, seq_len, 3)
      - ray : ray vector in shape=(batch, seq_len, 3)
  Output : 
      - intr_pts : ray to plane intersection points from camera through the ball tracking
  '''
  cam_pos = np.linalg.inv(E)[..., 0:3, -1]
  normal = np.array([0, 1, 0])
  p0 = np.array([0, 0, 0])
  ray_norm = ray / np.linalg.norm(ray, axis=-1, keepdims=True)

  t = np.dot((p0-cam_pos), normal) / np.dot(ray_norm, normal)
  t = np.expand_dims(t, axis=-1)

  intr_pos = cam_pos + (ray_norm * t)

  return intr_pos

def get_bound(I, E):
  w = 1664.0
  h = 1088.0
  d = 1
  
  screen_bound = np.array([[[0, 0, d], [0, h, d], [w, h, d], [w, 0, d]]])
  I = np.expand_dims(np.array([I[i] for i in range(screen_bound.shape[1])]), axis=0)
  E = np.expand_dims(np.array([E[i] for i in range(screen_bound.shape[1])]), axis=0)
  ray = cast_ray(uv=screen_bound[..., :2], I=I, E=E)
  world_bound = ray_to_plane(E, ray)
  #print("Intrinsic : \n", I[0][0])
  #print("Extrinsic : \n", E[0][0])
  #print("Extrinsic_inv : \n", np.linalg.inv(E[0][0]))
  #print("World bound : \n", world_bound)
  #print("Ray : \n", ray)
  return world_bound, ray

def visualize_trajectory(trajectory_df, fig, group):
  #print('=' * 100)
  #print('[#] Visualization')
  #print('Shape : ', trajectory_df.shape)
  #print('Example of data : ')
  #print(trajectory_df[['ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'ball_plane_x', 'ball_plane_y', 'ball_plane_z']].tail(5))
  #print('=' * 100)
  seq_len = trajectory_df.shape[0]
  # marker properties of uv, xyz
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=3)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=3)
  marker_dict_cam = dict(color='rgba(128, 0, 128, 0.7)', size=3)
  marker_dict_ray = dict(color='rgba(128, 0, 128, 0.3)', size=3)
  marker_dict_bound = dict(color='rgba(0, 255, 0, 0.7)', size=3)
  # marker properties of displacement
  marker_dict_u = dict(color='rgba(200, 0, 0, 0.4)', size=7)
  marker_dict_v = dict(color='rgba(0, 125, 0, 0.4)', size=7)
  marker_dict_depth = dict(color='rgba(0, 0, 255, 0.4)', size=7)
  # Screen
  fig.add_trace(go.Scatter(x=trajectory_df['ball_screen_unity_u_project'], y=trajectory_df['ball_screen_unity_v_project'], mode='markers+lines', marker=marker_dict_pred, legendgroup=group, legendgrouptitle_text=group, name="Trajectory (Screen coordinates proj_unproj_verify)"), row=1, col=1)
  # Displacement
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_screen_unity_u_project']), mode='lines', marker=marker_dict_u, legendgroup=group, legendgrouptitle_text=group, name="Displacement - u"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_screen_unity_v_project']), mode='lines', marker=marker_dict_v, legendgroup=group, legendgrouptitle_text=group, name="Displacement - v"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_camera_depth']), mode='lines', marker=marker_dict_depth, legendgroup=group, legendgrouptitle_text=group, name="Displacement - Depth"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_camera_depth']), mode='lines', marker=marker_dict_depth, legendgroup=group, legendgrouptitle_text=group, name="Depth"), row=2, col=1)
  # World
  fig.add_trace(go.Scatter3d(x=trajectory_df['ball_plane_x'], y=trajectory_df['ball_plane_y'], z=trajectory_df['ball_plane_z'], mode='markers+lines', marker=marker_dict_gt, legendgroup=group, legendgrouptitle_text=group, name="Motion capture (World coordinates)"), row=1, col=2)
  fig.add_trace(go.Scatter3d(x=trajectory_df['ball_plane_x_unity'], y=trajectory_df['ball_plane_y_unity'], z=trajectory_df['ball_plane_z_unity'], mode='markers+lines', marker=marker_dict_pred, legendgroup=group, legendgrouptitle_text=group, name="proj_unproj_verify trajectory (World coordinates)"), row=1, col=2)
  #fig['layout']['scene1'].update(xaxis=dict(nticks=5, range=[-4, 4]), yaxis=dict(nticks=5, range=[-1, 2]), zaxis=dict(nticks=5, range=[-4, 4]))
  # Camera
  E = np.stack([trajectory_df['Extrinsic_unity'][i] for i in range(len(trajectory_df['Extrinsic_unity']))], axis=0)
  E_inv = np.linalg.inv(E)
  cam = np.unique(E_inv[:, 0:3, 3], axis=0)
  fig.add_trace(go.Scatter3d(x=cam[:, 0].reshape(-1), y=cam[:, 1].reshape(-1), z=cam[:, 2].reshape(-1), mode='markers', marker=marker_dict_cam, legendgroup=group, legendgrouptitle_text=group, name=group), row=1, col=2)
  # Bound
  bound, ray = get_bound(E=trajectory_df['Extrinsic_unity'].values, I=trajectory_df['projectionMatrix_unity'].values)
  bound = np.concatenate((bound[0], bound[0][0].reshape(1, 3)), axis=0)
  fig.add_trace(go.Scatter3d(x=bound[:, 0].reshape(-1), y=bound[:, 1].reshape(-1), z=bound[:, 2].reshape(-1), mode='markers+lines', marker=marker_dict_bound, legendgroup=group, legendgrouptitle_text=group, name="Bound"), row=1, col=2)
  # Ray
  ray = np.squeeze(ray, axis=0)
  l = bound[:-1] - cam
  l = np.sqrt(np.sum(l**2, axis=1))
  for i in range(ray.shape[0]):
    r = (ray[i] / np.linalg.norm(ray[i], axis=0)) * l[i]
    r_x, r_y, r_z = [cam[:, 0], cam[:, 0] + r[0]], [cam[:, 1], cam[:, 1] + r[1]], [cam[:, 2], cam[:, 2] + r[2]]
    r_x, r_y, r_z = np.array(r_x), np.array(r_y), np.array(r_z)
    fig.add_trace(go.Scatter3d(x=r_x.reshape(-1), y=r_y.reshape(-1), z=r_z.reshape(-1), mode='markers+lines', marker=marker_dict_ray, legendgroup=group, legendgrouptitle_text=group, name="Ray"), row=1, col=2)

  #fig.show()
  return fig

def load_config_file(folder_name, idx, cam):
  config_file_E_inv = folder_name + "/motioncapture_camParams_Trial_{}_{}.yml".format(idx, args.unit)
  E_inv = cv2.FileStorage(config_file_E_inv, cv2.FILE_STORAGE_READ)
  config_file_intrinsic = folder_name + "/motioncapture_camIntrinsic_{}.yml".format(args.unit)
  intrinsic = cv2.FileStorage(config_file_intrinsic, cv2.FILE_STORAGE_READ)
  nodes_extrinsic = [cam, "w", "h"]
  cam_dict = {}
  change_basis_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

  if args.unit == 'cm':
    scaling = 1e-3
  elif args.unit == 'm':
    scaling = 1

  # Load an extrinsic
  for each_node in nodes_extrinsic:
    # print("Translation : ", extrinsic.getNode(each_node).mat()[:-1, -1].reshape(-1, 1))
    if each_node == "w" or each_node == "h":
      cam_dict[each_node] = E_inv.getNode(each_node).real()
    elif 'K' in each_node:
      '''
       MOCAP
       - ROTATION Matrix: World-space rotation from camera extrinsic
       - Translation Vector : camera position in world-space
      '''

      #rotation = extrinsic.getNode(each_node).mat()[:-1, :-1]
      #rotation_inv = np.linalg.inv(rotation)
      #rotation_inv = np.linalg.inv(extrinsic.getNode(each_node).mat()[:-1, :-1])

      rotation_inv = E_inv.getNode(each_node).mat()[:-1, :-1]
      rotation = np.linalg.inv(rotation_inv)


      translation = E_inv.getNode(each_node).mat()[:-1, -1].reshape(-1, 1) / scaling

      cam_dict['Inversed_Extrinsic'] = np.concatenate((np.concatenate((rotation_inv, translation), axis=1), np.array([[0, 0, 0, 1]])))
      cam_dict['Extrinsic'] = np.linalg.inv(cam_dict['Inversed_Extrinsic'].copy())
      print("#" * 100)
      print(each_node)
      print("E")
      for i in range(cam_dict['Extrinsic'].shape[0]):
        tmp = cam_dict['Extrinsic'][i]
        print("E_inv.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))

      print("Einv")
      for i in range(cam_dict['Inversed_Extrinsic'].shape[0]):
        tmp = cam_dict['Inversed_Extrinsic'][i]
        print("E_inv.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))



  # ========== Make it similar to unity ==========
  # [#] Note@13 Aug 2020 : Using Extrinsic from Mocap -> Set to unity get the Inversed of rotation Then i need to inversed it.
  # [#] Note@15 Aug 2020 : Unity accept the inversed extrinsic of rotation and translation part.
  # Change of basis to make the extrinsic in mocap be the same as unity
  # worldToCamera = [[1, 0, 0], [0, 1, 0], [0, 0, -1]] @ Rotation_unity
  # Then rotation_unity = change_basis_matrix @ change_basis_matrix @ rotation_optitrack @ change_basis_matrix = I @ rotation_optitrack @ change_basis_matrix
  rotation_unity_inv = np.linalg.inv(rotation @ change_basis_matrix)
  translation_unity = np.array([translation[0], translation[1], -translation[2]])
  cam_dict['Inversed_Extrinsic_unity'] = np.concatenate((np.concatenate((rotation_unity_inv, translation_unity), axis=1), np.array([[0, 0, 0, 1]])))
  cam_dict['Extrinsic_unity'] = np.linalg.inv(cam_dict['Inversed_Extrinsic_unity'].copy())

  # Load an intrinsic and convert to projectionMatrix
  cam_dict["Intrinsic"] = intrinsic.getNode("K").mat()

  # Set the projectionMatrix to be the same as unity
  far = 1000
  near = 0.1
  fx = cam_dict['Intrinsic'][0, 0]
  fy = cam_dict['Intrinsic'][1, 1]
  cx = cam_dict['Intrinsic'][0, 2]
  cy = cam_dict['Intrinsic'][1, 2]

  # Convert the projectionMatrix of real world to unity
  cam_dict["projectionMatrix"] = np.array([[fx/cx, 0, 0, 0],
                                                         [0, fy/cy, 0, 0],
                                                         [0, 0, -(far+near)/(far-near), -2*(far*near)/(far-near)],
                                                         [0, 0, -1, 0]])

  I_unity = cam_dict['projectionMatrix'].copy()
  I_unity[2, :] = I_unity[3, :]
  I_unity[3, :] = np.array([0, 0, 0, 1])
  cam_dict['projectionMatrix_unity'] = I_unity

  #print("#" * 100)
  #print("[###] Motion Capture")
  #print("Extrinsic (WorldToCameraMatrix) : \n", cam_dict['Extrinsic'])
  #print("Inversed Extrinsic (CameraToWorldMatrix) : \n", cam_dict['Inversed_Extrinsic'])
  #print("#" * 100)
  #print("[###] Unity")
  #print("Unity Extrinsic : \n", cam_dict['Extrinsic_unity'])
  #print("Unity Inversed_Extrinsic : \n", cam_dict['Inversed_Extrinsic_unity'])
  #print("#" * 100)
  #print("Intrinsic : \n", cam_dict['Intrinsic'])
  #print("#" * 100)
  #print("Projection Matrix : \n", cam_dict['projectionMatrix'])
  #print("#" * 100)


  print("E_unity")
  for i in range(cam_dict['Extrinsic_unity'].shape[0]):
    tmp = cam_dict['Extrinsic_unity'][i]
    print("E.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))

  print("Einv_unity")
  for i in range(cam_dict['Inversed_Extrinsic_unity'].shape[0]):
    tmp = cam_dict['Inversed_Extrinsic_unity'][i]
    print("E.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))

  print("#" * 100)
  return cam_dict

def split_trajectory(t_df, cam_dict):
    '''
    Seperate the trajectory (ground = 0.025 in y-axis), Remove a short/incomplete trajectory
    Input :
        1. t_df : Dataframe of trajectory sequences
        2. cam_dict : Dictionary contains camera parameters
    Output : 
        1. ts_prep : List of preprocessed trajectories
    '''
    # Maximum possible size of pitch in unity
    if args.unit == 'cm':
        scaling = 100
    elif args.unit == 'm':
        scaling = 1
    ground_threshold = 0.025
    len_threshold = 150
    t_split = np.split(t_df, np.where(np.isnan(t_df['ball_plane_x']))[0]) # Split by nan
    t_split = [t[~np.isnan(t['ball_plane_x'])] for t in t_split if not isinstance(t, np.ndarray)] # Remove NaN entries
    t_split = [t for t in t_split if not t.empty] # Remove empty DataFrames
    ground = [np.where((t_split[i]['ball_plane_y'].values < ground_threshold) == True)[0] for i in range(len(t_split))] # Ground position

    ts_prep = []
    for i, t in enumerate(t_split):
        if len(ground[i]) <= 2:
            continue
        else:
            t_split[i][['ball_opencv_u', 'ball_opencv_v']] = t_split[i][['ball_opencv_u', 'ball_opencv_v']].astype(float)
            t_split[i][['ball_plane_x', 'ball_plane_y', 'ball_plane_z']] = t_split[i][['ball_plane_x', 'ball_plane_y', 'ball_plane_z']] * scaling
            t_split[i] = t_split[i].iloc[ground[i][0]:, :]
            t_split[i].reset_index(drop=True, inplace=True)
            # Remove outside/short trajectory
            if (np.all(t_split[i][['ball_opencv_u', 'ball_opencv_v']] > 1e-16) and 
                np.all(t_split[i][['ball_opencv_u']] < cam_dict['w']) and 
                np.all(t_split[i][['ball_opencv_v']] < cam_dict['h']) and 
                t_split[i].shape[0] >= len_threshold):
                    ts_prep.append(t_split[i])

    print("Number of all trajectory : ", len(t_split))
    return ts_prep

def to_npy(trajectory_df, trajectory_type, cam_params):
  '''
  [#] Remove unused columns and save to .npy
  -> Before : ['ball_screen_opencv_u(0)', 'ball_screen_opencv_v(1)', 'ball_plane_x(2)', 'ball_plane_y(3)', 'ball_plane_z(4)',
      'ball_camera_x(5)', 'ball_camera_y(6)', 'ball_camera_depth(7)', 'ball_screen_unity_u_project(8)', 'ball_screen_unity_v_project(9)']
  -> After : ['ball_screen_unity_u_project(8)', 'ball_screen_unity_v_project(9)'
      'ball_plane_x(2)', 'ball_plane_y(3)', 'ball_plane_z(4)']
  '''
  drop_cols = ["frame", "camera_index", "camera_serial", "Time"]
  f_cols = ['ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'ball_camera_x', 'ball_camera_y', 'ball_camera_depth', 'ball_plane_x', 'ball_plane_y', 'ball_plane_z']
  trajectory_npy = trajectory_df.copy()
  for traj_type in trajectory_type:
    temp = []
    for i in range(len(trajectory_df[traj_type])):
      # Trajectory
      t = trajectory_df[traj_type][i].drop(drop_cols, axis=1).loc[:, f_cols]
      # Add I, E cols
      E = cam_params['Extrinsic_unity']
      I = cam_params['projectionMatrix_unity']
      mat = pd.DataFrame({'E':[E]*t.shape[0], 'I':[I]*t.shape[0]})
      t['Extrinsic_unity'] = mat['E']
      t['projectionMatrix_unity'] = mat['I']
      temp.append(t)

    trajectory_npy[traj_type] = temp

  return trajectory_npy

if __name__ == '__main__':
  # List trial in directory
  dataset_folder = sorted(glob.glob(args.dataset_path + "/*/"))
  pattern = r'(Trial_[0-9]+)+'
  print("Dataset : ", [re.findall(pattern, dataset_folder[i]) for i in range(len(dataset_folder))])
  if args.process_trial_index is not None:
    # Use only interesting trial : Can be edit the trial index in process_trial_index.txt
    with open(args.process_trial_index) as f:
      # Split the text input of interested trial index into list of trial index
      trial_index = f.readlines()[-1].split()
      # Create the pattern for regex following this : (10)|(11)|(12) ===> match the any trial from 10, 11 or 12
      pattern_trial_index= ['({})'.format(trial_index[i]) for i in range(len(trial_index))]
      # Add it into full pattern of regex : r'(Trial_((10)|(11)|(12))+)+'
      pattern_trial_index = r'(Trial_({})+)+'.format('|'.join(pattern_trial_index))
      # filter the dataset folder which is not in the trial_index
      filter_trial_index = [re.search(pattern_trial_index, dataset_folder[i]) for i in range(len(dataset_folder))]
      dataset_folder = [dataset_folder[i] for i in range(len(filter_trial_index)) if filter_trial_index[i] is not None]
  else :
    # Use all trial
    trial_index = [re.findall(r'[0-9]+', re.findall(pattern, dataset_folder[i])[0])[0] for i in range(len(dataset_folder))]
  print("Trial index : ", trial_index)

  # Define columns names following the MocapCameraParameters.cpp
  col_names = ["frame", "camera_index", "camera_serial", "Time", "ball_opencv_u", "ball_opencv_v", "ball_plane_x", "ball_plane_y", "ball_plane_z"]
  cam_list = ['K{}'.format(i) for i in range(1, 10)]
  cam_list.remove('K4')
  # Trajectory type
  trajectory_type = ["Rolling", "Projectile", "MagnusProjectile", "Mixed"]
  for i in tqdm.tqdm(range(len(dataset_folder)), desc="Loading dataset"):
    output_path = get_savepath(args.output_path, dataset_folder[i])
    # Read json for column names
    trajectory_df = {}
    fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter3d'}], [{'type':'scatter'}, {'type':'scatter'}]])
    for c_ in cam_list:
      cam_dict = load_config_file(folder_name=dataset_folder[i], idx=trial_index[i], cam=c_)
      # Check and read .csv file in the given directory
      for traj_type in trajectory_type:
        if os.path.isfile(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i])):
          trajectory_df[traj_type] = pd.read_csv(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i]), names=col_names, delimiter=',')
          # Preprocess by splitting, adding EOT and remove the first incomplete trajectory
          trajectory_df[traj_type] = split_trajectory(trajectory_df[traj_type], cam_dict)
          # Get the depth, screen position in unity
          print("#"*100)
          print("[###] Get depth from XYZ motion capture using Extrinsic (Move to camera space)")
          trajectory_df[traj_type] = [get_depth(trajectory_df[traj_type][i], cam_dict) for i in range(len(trajectory_df[traj_type]))]

      print("Trajectory type in Trial{} : {}".format(trial_index[i], trajectory_df.keys()))

      # Cast to npy format
      trajectory_npy = to_npy(trajectory_df, trajectory_type=trajectory_df.keys(), cam_params=cam_dict)
      sav_col = ['ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'projectionMatrix_unity', 'Extrinsic_unity', 'ball_plane_x', 'ball_plane_y', 'ball_plane_z']
      # Save the .npy files in shape and formation that ready to use in train/test
      for traj_type in trajectory_df.keys():
        print("Preprocessed trajectory shape : ", np.array(trajectory_npy[traj_type]).shape)
        temp_arr = np.array([trajectory_npy[traj_type][i].loc[:, sav_col] for i in range(len(trajectory_npy[traj_type]))])
        np.save(file=output_path + "/{}Trajectory_Trial{}.npy".format(traj_type, trial_index[i]), arr=temp_arr)

      # Visualize the trajectory
      vis_idx = np.random.randint(0, len(trajectory_npy['Mixed']))
      vis_idx = 1
      trajectory_plot = pd.DataFrame(data=trajectory_npy['Mixed'][vis_idx].copy(),
                                    columns=['ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'ball_camera_x', 'ball_camera_y', 'ball_camera_depth', 'ball_plane_x', 'ball_plane_y', 'ball_plane_z', 'Extrinsic_unity', 'projectionMatrix_unity'])

      trajectory_plot = proj_unproj_verify(trajectory_df=trajectory_plot, cam_params=cam_dict)
      fig = visualize_trajectory(trajectory_df=trajectory_plot, fig=fig, group=c_)
    fig.show()
    exit()