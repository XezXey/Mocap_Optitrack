from __future__ import print_function
# Import libs
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse
from tqdm import tqdm
import PIL.Image
import io
import plotly
from scipy.spatial.transform import Rotation
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import json
import re
import tqdm
import pandas as pd
import cv2
import math
pd.options.mode.chained_assignment = None  # default='warn'
from scipy import interpolate
from scipy import signal

parser = argparse.ArgumentParser(description='Preprocess and visualize the trajectory in real world')
parser.add_argument('--dataset_path', type=str, help='Specify path to dataset')
parser.add_argument('--process_trial_index', dest='process_trial_index', help='Process trial at given idx only', default=None)
parser.add_argument('--output_path', type=str, help='Specify output path to save dataset')
parser.add_argument('--vis', dest='vis', help='Visualize the trajectory', action='store_true')
parser.add_argument('--no_vis', dest='vis', help='Visualize the trajectory', action='store_false')
parser.add_argument('--scale', dest='scale', help='Scale the uv-coordinates', action='store_true')
parser.add_argument('--no_scale', dest='scale', help='Scale the uv-coordinates', action='store_false')
parser.add_argument('--label', dest='label', help='Manually Label UV -> EOT', action='store_true')
parser.add_argument('--no_label', dest='label', help='No Label UV -> EOT', action='store_false')
parser.add_argument('--load_label', dest='load_label', help='Loading the label and concatenate to the preprocessed trajectory', action='store_true')
parser.add_argument('--unit', dest='unit', help='Scaling unit', required=True, type=str)
parser.add_argument('--clip_threshold', dest='clip_threshold', help='applying clipping that point out', type=int, default=None)
args = parser.parse_args()

def get_savepath(output_path, dataset_folder):
  if output_path == None:
    output_path = dataset_folder
  else:
    output_path = args.output_path
    if not os.path.exists(output_path):
      os.makedirs(output_path)
  return output_path

def get_depth(trajectory_df, cam_params):
  '''
  This function will return the points in camera space and screen space
  - This projectionMatrix use in unity screen convention : (0, 0) is in the bottom-left
  '''
  worldToCameraMatrix = np.array(cam_params['Extrinsic_unity'])
  projectionMatrix = np.array(cam_params['projectionMatrix'])
  # Remove the clip space that we don't use in proj_unproj_verifyion of real cameras
  projectionMatrix[2, :] = projectionMatrix[3, :]
  projectionMatrix[3, :] = np.array([0, 0, 0, 1])
  # Plane space
  world_space = np.hstack((trajectory_df[['ball_plane_x', 'ball_plane_y', 'ball_plane_z']].values, np.ones(trajectory_df.shape[0]).reshape(-1, 1)))
  # Uncomment this to test the particular points
  # world_space = np.array([[0, 0, 0, 1],
                          # [-21.2, 0.1, 0, 1],
                          # [-21.2, 0.1, 33, 1],
                          # [2, 0, 33, 1],
                          # [23.5, 0, 33, 1],
                          # [-20.5, 0, -11.1, 1],
                          # [-22.9, 0, 39.6, 1]])
  # WorldToCameraMatrix : Plane space -> Camera space (u*depth, v*depth, depth, 1)
  # In the unity depth is in the +z direction, so we need to negate the z direction that came from multiplied with extrinsic
  camera_space = world_space @ worldToCameraMatrix.T
  trajectory_df['ball_camera_x'] = camera_space[:, 0]
  trajectory_df['ball_camera_y'] = camera_space[:, 1]
  # trajectory_df['ball_camera_depth'] = -camera_space[:, 2]
  # projectionMatrix : Camera space -> NDC Space
  # In the unity, projection matrix didn't give the output as the screen space but it will in the NDC space(We'll see the world in range(-1, 1))
  # Then we need to unnormalized it to ge the screen space
  ndc_space = camera_space @ projectionMatrix.T
  trajectory_df['ball_camera_depth'] = ndc_space[:, 2]
  # Get the screen coordinates
  u = ((ndc_space[:, 0]/ndc_space[:, 2]) + 1) * (camera_properties_dict['w']/2)
  v = ((ndc_space[:, 1]/ndc_space[:, 2]) + 1) * (camera_properties_dict['h']/2)
  trajectory_df['ball_screen_unity_u_project'] = u
  trajectory_df['ball_screen_unity_v_project'] = v
  return trajectory_df

def proj_unproj_verify(trajectory_df, cam_params):
  '''
  Trying to proj_unproj_verify the point by given the (u, v and depth) -> world coordinates
  - This projectionMatrix use in unity screen convention : (0, 0) is in the bottom-left
  '''
  print("#" * 100)
  print("[###] proj_unproj_verifyion to verify the projection-proj_unproj_verifyion, transformation matrix")
  eps = np.finfo(float).eps
  # Get the projectionMatrix
  projectionMatrix = np.array(cam_params['projectionMatrix'])
  # Remove the clip space that we don't use in proj_unproj_verifyion of real cameras
  projectionMatrix[2, :] = projectionMatrix[3, :]
  projectionMatrix[3, :] = np.array([0, 0, 0, 1])
  # Get the camera Extrinsic
  worldToCameraMatrix = np.array(cam_params['Extrinsic_unity'])
  cameraToWorldMatrix = np.linalg.inv(worldToCameraMatrix)
  # The (u, v and depth) from the get_depth() function
  depth = trajectory_df['ball_camera_depth'].values.reshape(-1, 1)
  u_unity = trajectory_df['ball_screen_unity_u_project'].values.reshape(-1, 1)
  v_unity = trajectory_df['ball_screen_unity_v_project'].values.reshape(-1, 1)

  # The screen space will be : (u*depth, v*depth, depth, 1)
  screen_space = np.hstack((np.multiply(u_unity, depth), np.multiply(v_unity, depth), depth, np.ones(u_unity.shape)))
  print("PARAMS SHAPE : ", projectionMatrix.shape, worldToCameraMatrix.shape)
  print("DATA SHAPE : ", screen_space.shape, depth.shape, u_unity.shape, v_unity.shape)

  '''
  PROJECTION
  X, Y, Z (MOCAP) ===> U, V (UNITY)
  [#####] Note : Multiply the X, Y, Z point by scaling factor that use in opengl visualization because the obtained u, v came from the scaled => Scaling = 10
  [#####] Note : Don't multiply if use the perspective projection matrix from intrinsic
  '''

  print('='*40 + 'Projection checking...' + '='*40)
  # Get the world coordinates : (X, Y, Z, 1)
  world_space = np.hstack((trajectory_df[['ball_plane_x', 'ball_plane_y', 'ball_plane_z']], np.ones(trajectory_df.shape[0]).reshape(-1, 1)))
  # worldToCameraMatrix : World space -> Camera space (u*depth, v*depth, depth, 1)
  camera_space = world_space @ worldToCameraMatrix.T
  # projectionMatrix : Camera space -> NDC space (u and v) in range(-1, 1)
  ndc_space = camera_space @ projectionMatrix.T
  ndc_space[:, :-1] /= ndc_space[:, 2].reshape(-1, 1)
  # Get the Screen space from unnormalized the NDC space
  screen_space = ndc_space.copy()
  # We need to subtract the camera_properties_dict['w'] out of the ndc_space since the Extrinisc is inverse from the unity. So this will make x-axis swapped and need to subtract it to get the same 3D reconstructed points : 
  screen_space[:, 0] = ((ndc_space[:, 0]) + 1) * (camera_properties_dict['w']/2)
  screen_space[:, 1] = ((ndc_space[:, 1]) + 1) * (camera_properties_dict['h']/2)
  # The answer should be : True, True
  print("[#] Equality check of screen space (u, v) with projection : ", np.all(np.isclose(screen_space[:, 0].reshape(-1, 1), u_unity)), ", ", np.all(np.isclose(screen_space[:, 1].reshape(-1, 1), v_unity)))

  '''
  UNPROJECTION
  U, V, Depth (UNITY) ===> X, Y, Z (UNITY)
  '''
  # Target value
  target_camera_space = np.hstack((trajectory_df['ball_camera_x'].values.reshape(-1, 1), trajectory_df['ball_camera_y'].values.reshape(-1, 1), depth))
  target_world_space = np.hstack((trajectory_df['ball_plane_x'].values.reshape(-1, 1), trajectory_df['ball_plane_y'].values.reshape(-1, 1),
                                  trajectory_df['ball_plane_z'].values.reshape(-1, 1)))

  # [Screen, Depth] -> Plane
  # Following the proj_unproj_verifyion by inverse the projectionMatrix and inverse of arucoPlaneToCameraMatrix(here's cameraToWorldMatrix)
  # Normalizae the Screen space to NDC space
  # SCREEN -> NDC
  # screen_space = np.array([[888.1517, 428.5164, 54.5191, 1]])
  # screen_space = np.array([[888.1517, 428.5164, 1, 1]])
  screen_space[:, 0] /= camera_properties_dict['w']
  screen_space[:, 1] /= camera_properties_dict['h']
  ndc_space = screen_space.copy()
  ndc_space = (ndc_space * 2) - 1

  # NDC space -> CAMERA space (u*depth, v*depth, depth, 1)
  ndc_space = np.hstack((np.multiply(ndc_space[:, 0].reshape(-1, 1), depth), np.multiply(ndc_space[:, 1].reshape(-1, 1), depth), depth, np.ones(ndc_space.shape[0]).reshape(-1, 1)))
  projectionMatrix_inv = np.linalg.inv(projectionMatrix)
  camera_space =  ndc_space @ projectionMatrix_inv.T
  # CAMERA space -> Plane space
  world_space =  camera_space @ cameraToWorldMatrix.T
  # Store the plane proj_unproj_verify to dataframe
  trajectory_df.loc[:, 'ball_plane_x_unity'] = world_space[:, 0]
  trajectory_df.loc[:, 'ball_plane_y_unity'] = world_space[:, 1]
  trajectory_df.loc[:, 'ball_plane_z_unity'] = world_space[:, 2]
  print('='*40 + 'proj_unproj_verifyion checking...' + '='*40)
  print('Screen : {}\nCamera : {}\nPlane : {}'.format(screen_space[0].reshape(-1), target_camera_space[0].reshape(-1), target_world_space[0].reshape(-1)))
  print('\n[#]===> Target Camera proj_unproj_verifyion : ', target_camera_space[0].reshape(-1))
  print('Screen -> Camera (By projectionMatrix) : ', camera_space[0].reshape(-1))
  print('[Screen, depth] : ', camera_space[0].reshape(-1))
  print('\n[#]===> Target Plane proj_unproj_verifyion : ', target_world_space[3])
  print('Screen -> Plane (By inverse the projectionMatrix) : ', world_space[3])
  print("[#] Equality check of world space (x, y, z) : ", np.all(np.isclose(world_space[:, 0].reshape(-1, 1), trajectory_df['ball_plane_x'].values.reshape(-1, 1))), ", ", np.all(np.isclose(world_space[:, 1].reshape(-1, 1), trajectory_df['ball_plane_y'].values.reshape(-1, 1))), ", ", np.all(np.isclose(world_space[:, 2].reshape(-1, 1), trajectory_df['ball_plane_z'].values.reshape(-1, 1))))
  print('='*89)
  # exit()
  return trajectory_df

def visualize_trajectory(trajectory_df):
  print('=' * 100)
  print('[#] Visualization')
  print('Shape : ', trajectory_df.shape)
  print('Example of data : ')
  print(trajectory_df[['ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'EOT']].tail(5))
  print(np.diff(trajectory_df[['ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'EOT']].tail(5).values, axis=0))
  print('=' * 100)
  seq_len = trajectory_df.shape[0]
  # marker properties of uv, xyz
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=7)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=7)
  # marker properties of displacement
  marker_dict_u = dict(color='rgba(200, 0, 0, 0.4)', size=7)
  marker_dict_v = dict(color='rgba(0, 125, 0, 0.4)', size=7)
  marker_dict_depth = dict(color='rgba(0, 0, 255, 0.4)', size=7)
  marker_dict_eot = dict(color='rgba(0, 0, 255, 0.7)', size=7)
  fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter3d'}], [{'type':'scatter'}, {'type':'scatter'}]])
  # Screen
  fig.add_trace(go.Scatter(x=trajectory_df['ball_screen_unity_u_project'], y=trajectory_df['ball_screen_unity_v_project'], mode='markers+lines', marker=marker_dict_pred, name="Trajectory (Screen coordinates proj_unproj_verify)"), row=1, col=1)
  # Displacement
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_screen_unity_u_project']), mode='lines', marker=marker_dict_u, name="Displacement - u"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_screen_unity_v_project']), mode='lines', marker=marker_dict_v, name="Displacement - v"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_camera_depth']), mode='lines', marker=marker_dict_depth, name="Displacement - Depth"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=np.diff(trajectory_df['ball_camera_depth']), mode='lines', marker=marker_dict_depth, name="Depth"), row=2, col=1)
  # EOT
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=trajectory_df.loc[1:, 'EOT'], mode='lines', marker=marker_dict_eot, name="EOT"), row=2, col=2)
  fig.add_trace(go.Scatter(x=np.arange(seq_len-1), y=trajectory_df.loc[1:, 'EOT'], mode='lines', marker=marker_dict_eot, name="EOT"), row=2, col=1)
  # World
  fig.add_trace(go.Scatter3d(x=trajectory_df['ball_plane_x'], y=trajectory_df['ball_plane_y'], z=trajectory_df['ball_plane_z'], mode='markers+lines', marker=marker_dict_gt, name="Motion capture (World coordinates)"), row=1, col=2)
  if not args.label or not args.load_label:
    fig.add_trace(go.Scatter3d(x=trajectory_df['ball_plane_x_unity'], y=trajectory_df['ball_plane_y_unity'], z=trajectory_df['ball_plane_z_unity'], mode='markers+lines', marker=marker_dict_pred, name="proj_unproj_verify trajectory (World coordinates)"), row=1, col=2)
  return fig

def load_config_file(folder_name, idx):
  config_file_extrinsic = folder_name + "/motioncapture_camParams_Trial_{}_{}.yml".format(idx, args.unit)
  extrinsic = cv2.FileStorage(config_file_extrinsic, cv2.FILE_STORAGE_READ)
  config_file_intrinsic = folder_name + "/motioncapture_camIntrinsic_{}.yml".format(args.unit)
  intrinsic = cv2.FileStorage(config_file_intrinsic, cv2.FILE_STORAGE_READ)
  nodes_extrinsic = ["K2", "w", "h"]
  nodes_intrinsic = ["K"]
  camera_properties_dict = {}
  change_basis_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

  if args.unit == 'cm':
    scaling = 1e-3
  elif args.unit == 'm':
    scaling = 1

  # Load an extrinsic
  for each_node in nodes_extrinsic:
    # print("Translation : ", extrinsic.getNode(each_node).mat()[:-1, -1].reshape(-1, 1))
    if each_node == "w" or each_node == "h":
      camera_properties_dict[each_node] = extrinsic.getNode(each_node).real()
    elif each_node == "K2":
      '''
       MOCAP
       - ROTATION Matrix: World-space rotation from camera extrinsic
       - Translation vector : camera position in world-space
      '''

      rotation_inv = np.linalg.inv(extrinsic.getNode(each_node).mat()[:-1, :-1])
      rotation = extrinsic.getNode(each_node).mat()[:-1, :-1]
      translation = extrinsic.getNode(each_node).mat()[:-1, -1].reshape(-1, 1) / scaling

      camera_properties_dict['Inversed_Extrinsic'] = np.concatenate((np.concatenate((rotation_inv, translation), axis=1), np.array([[0, 0, 0, 1]])))
      camera_properties_dict['Extrinsic'] = np.linalg.inv(camera_properties_dict['Inversed_Extrinsic'])
      # camera_properties_dict['Inversed_Extrinsic'] = np.concatenate((np.concatenate((rotation, translation), axis=1), np.array([[0, 0, 0, 1]])))
      # camera_properties_dict['Extrinsic'] = np.linalg.inv(camera_properties_dict['Inversed_Extrinsic'])

      # Divide the translation by 10 because data acquisition I've scaled it up by 10.

  # ========== Make it similar to unity ==========
  # [#] Note@13 Aug 2020 : Using Extrinsic from Mocap -> Set to unity get the Inversed of rotation Then i need to inversed it.
  # [#] Note@15 Aug 2020 : Unity accept the inversed extrinsic of rotation and translation part.
  # Change of basis to make the extrinsic in mocap be the same as unity
  # worldToCamera = [[1, 0, 0], [0, 1, 0], [0, 0, -1]] @ Rotation_unity
  # Then rotation_unity = change_basis_matrix @ change_basis_matrix @ rotation_optitrack @ change_basis_matrix = I @ rotation_optitrack @ change_basis_matrix
  rotation_unity_inv = np.linalg.inv(rotation @ change_basis_matrix)
  translation_unity = np.array([translation[0], translation[1], -translation[2]])
  camera_properties_dict['Inversed_Extrinsic_unity'] = np.concatenate((np.concatenate((rotation_unity_inv, translation_unity), axis=1), np.array([[0, 0, 0, 1]])))
  camera_properties_dict['Extrinsic_unity'] = np.linalg.inv(camera_properties_dict['Inversed_Extrinsic_unity'])

  # Load an intrinsic and convert to projectionMatrix
  camera_properties_dict["Intrinsic"] = intrinsic.getNode("K").mat()

  # Set the projectionMatrix to be the same as unity
  far = 1000
  near = 0.1
  fx = camera_properties_dict['Intrinsic'][0, 0]
  fy = camera_properties_dict['Intrinsic'][1, 1]
  cx = camera_properties_dict['Intrinsic'][0, 2]
  cy = camera_properties_dict['Intrinsic'][1, 2]

  # Convert the projectionMatrix of real world to unity
  camera_properties_dict["projectionMatrix"] = np.array([[fx/cx, 0, 0, 0],
                                                         [0, fy/cy, 0, 0],
                                                         [0, 0, -(far+near)/(far-near), -2*(far*near)/(far-near)],
                                                         [0, 0, -1, 0]])

  print("#" * 100)
  print("[###] Motion Capture")
  print("Extrinsic (WorldToCameraMatrix) : \n", camera_properties_dict['Extrinsic'])
  print("Inversed Extrinsic (CameraToWorldMatrix) : \n", camera_properties_dict['Inversed_Extrinsic'])
  print("#" * 100)
  print("[###] Unity")
  print("Unity Extrinsic : \n", camera_properties_dict['Extrinsic_unity'])
  print("Unity Inversed_Extrinsic : \n", camera_properties_dict['Inversed_Extrinsic_unity'])
  print("#" * 100)
  print("Intrinsic : \n", camera_properties_dict['Intrinsic'])
  print("#" * 100)
  print("Projection Matrix : \n", camera_properties_dict['projectionMatrix'])
  print("#" * 100)

  return camera_properties_dict

def preprocess_split_eot(trajectory_df, camera_properties_dict):
  '''
  This function will clean the data by split the trajectory, remove some trajectory that too short(maybe an artifact and
  make the first and last point start on ground (ground_threshold = 0.025 in y-axis)
  '''
  # Maximum possible size of pitch in unity
  if args.unit == 'cm':
    scaling = 100
  elif args.unit == 'm':
    scaling = 1
  ground_threshold = 0.025
  length_threshold = 150
  # Split by nan
  trajectory_split = np.split(trajectory_df, np.where(np.isnan(trajectory_df['ball_plane_x']))[0])
  # removing NaN entries
  trajectory_split = [each_traj[~np.isnan(each_traj['ball_plane_x'])] for each_traj in trajectory_split if not isinstance(each_traj, np.ndarray)]
  # removing empty DataFrames
  trajectory_split = [each_traj for each_traj in trajectory_split if not each_traj.empty]
  # Remove some dataset that have an start artifact and an end artifact
  ground_annotated = [np.where((trajectory_split[i]['ball_plane_y'].values < ground_threshold) == True)[0] for i in range(len(trajectory_split))]

  trajectory_split_clean = []
  # Add the EOT flag at the last rows
  print("[###] Preprocessing...")
  print("Filter the uncompleteness projectile (Remove : start from air, too short, and outside the screen)")
  for idx in range(len(trajectory_split)):
    # print("IDX : ", idx, " => length ground_annotated : ", len(ground_annotated[idx]), ", length trajectory : ", len(trajectory_split[idx]))
    if len(ground_annotated[idx]) <= 2:
      # Not completely projectile
      print("At {} : Not completely projectile ===> Continue...".format(idx))
      continue
    else :
      trajectory_split[idx]['ball_screen_opencv_u'] = trajectory_split[idx]['ball_screen_opencv_u'].astype(float)
      trajectory_split[idx]['ball_screen_opencv_v'] = trajectory_split[idx]['ball_screen_opencv_v'].astype(float)
      # Invert the world coordinates to make the same convention
      trajectory_split[idx]['ball_plane_z'] *= -1
      # Scaling the trajectory in real world to be close to unity scale
      if args.scale:
        # These scale came from possible range in unity divided by possible range at mocap
        trajectory_split[idx]['ball_plane_x'] = (trajectory_split[idx]['ball_plane_x'] * scaling)
        trajectory_split[idx]['ball_plane_y'] = (trajectory_split[idx]['ball_plane_y'] * scaling)
        trajectory_split[idx]['ball_plane_z'] = (trajectory_split[idx]['ball_plane_z'] * scaling)
      # trajectory_split[idx] = trajectory_split[idx].iloc[ground_annotated[idx][0]:ground_annotated[idx][-1], :]
      trajectory_split[idx] = trajectory_split[idx].iloc[ground_annotated[idx][0]:, :]
      trajectory_split[idx].reset_index(drop=True, inplace=True)

      # Initialize the EOT flag to 0
      trajectory_split[idx]['EOT'] = np.zeros(trajectory_split[idx].shape[0], dtype=bool)
      trajectory_split_clean.append(trajectory_split[idx])

  print("Number of all trajectory : ", len(trajectory_split))

  # Remove the point that stay outside the screen and too short(blinking point).
  trajectory_split_clean = [each_traj for each_traj in trajectory_split_clean if (np.all(each_traj[['ball_screen_opencv_u', 'ball_screen_opencv_v']] > np.finfo(float).eps) and np.all(each_traj[['ball_screen_opencv_u']] < camera_properties_dict['w']) and np.all(each_traj[['ball_screen_opencv_v']] < camera_properties_dict['h']) and each_traj.shape[0] >= length_threshold)]

  print("Number of all trajectory(Inside the screen) : ", len(trajectory_split_clean))
  print("#" * 100)
  return trajectory_split_clean

def manually_label_eot(trajectory):
  fig = visualize_trajectory(trajectory)
  fig.show()
  unsuccessful_label = True
  while unsuccessful_label:
    eot_annotated = np.array([int(x) for x in input("Input the index of EOT (Available indexing = {}) : ".format(trajectory.index)).split()])
    if np.all(eot_annotated < trajectory.shape[0]):
      unsuccessful_label = False
    else: print("[#]Annotation index is out-of-bounds, Re-annotate...")

  trajectory['EOT'][eot_annotated] = True
  return trajectory['EOT'].values

def computeDisplacement(trajectory_df, trajectory_type):
  # Compute the displacement
  # print(trajectory_df['Mixed'][0].columns)
  drop_cols = ["frame", "camera_index", "camera_serial", "Time", "EOT"]
  trajectory_npy = trajectory_df.copy()
  for traj_type in trajectory_type:
    trajectory_npy[traj_type] = [np.hstack((np.vstack((trajectory_df[traj_type][i].drop(drop_cols, axis=1).iloc[0, :].values,
                                                       np.diff(trajectory_df[traj_type][i].drop(drop_cols, axis=1).values, axis=0))),
                                            trajectory_df[traj_type][i]['EOT'].values.reshape(-1, 1))) for i in range(len(trajectory_df[traj_type]))]

    if args.label or args.load_label:
      # Use the eot_filtered_index to cut the trajectory
      eot_filtered_index = [np.where(trajectory_npy[traj_type][idx][:, -1] == True)[0][-1] for idx in range(len(trajectory_df[traj_type]))]
      trajectory_npy[traj_type] = [trajectory_npy[traj_type][idx][:eot_filtered_index[idx]+1, :] for idx in range(len(trajectory_df[traj_type]))]

    # Before reindex columns would be : ['ball_screen_opencv_u(0)', 'ball_screen_opencv_v(1)', 'ball_plane_x(2)', 'ball_plane_y(3)', 'ball_plane_z(4)',
       # 'ball_camera_x(5)', 'ball_camera_y(6)', 'ball_camera_depth(7)', 'ball_screen_unity_u_project(8)', 'ball_screen_unity_v_project(9)', 'EOT(10)']

  # Cast to ndarray (Bunch of trajectory)
  # rearrange_index = [2, 3, 4, 0, 1, 7, 6, 5, 0, 1, 10]  # x=0, y=1, z=2, u=3, v=4, depth=5, eot=-1
  rearrange_index = [2, 3, 4, 8, 9, 7, 6, 5, 0, 1, 10]  # x=0, y=1, z=2, u=3, v=4, depth=5, eot=-1
  # Finally the columns would be : ['ball_plane_x(2)', 'ball_plane_y(3)', 'ball_plane_z(4)', 'ball_screen_unity_u_project(8)', 
  # 'ball_screen_unity_v_project(9)', 'ball_camera_depth(7)', 'ball_camera_y(6)', 'ball_camera_x(5)', 'ball_screen_opencv_u(0)', 'ball_screen_opencv_v(1)', 'EOT(10)']

  # Reindex following : x=0, y=1, z=2, u=3, v=4, depth=5, eot=-1
  trajectory_npy[traj_type] = np.array([trajectory_npy[traj_type][i][:, rearrange_index] for i in range(len(trajectory_npy[traj_type]))])
  # Too far point
  if args.clip_threshold is not None:
    for idx, each_traj in enumerate(trajectory_npy[traj_type]):
      clip_mask = np.logical_and(np.abs(each_traj[..., [3]]) < args.clip_threshold, np.abs(each_traj[..., [4]]) < args.clip_threshold)
      clip_mask[0, :] = True
      trajectory_npy[traj_type][idx] = each_traj[clip_mask.reshape(-1), :]

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
  col_names = ["frame", "camera_index", "camera_serial", "Time", "ball_screen_opencv_u", "ball_screen_opencv_v", "ball_plane_x", "ball_plane_y", "ball_plane_z"]
  # Trajectory type
  trajectory_type = ["Rolling", "Projectile", "MagnusProjectile", "Mixed"]
  for i in tqdm.tqdm(range(len(dataset_folder)), desc="Loading dataset"):
    output_path = get_savepath(args.output_path, dataset_folder[i])
    # Read json for column names
    trajectory_df = {}
    camera_properties_dict = load_config_file(folder_name=dataset_folder[i], idx=trial_index[i])
    # Check and read .csv file in the given directory
    for traj_type in trajectory_type:
      if os.path.isfile(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i])):
        trajectory_df[traj_type] = pd.read_csv(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i]), names=col_names, delimiter=',')
        # Preprocess by splitting, adding EOT and remove the first incomplete trajectory(throwing)
        # The return from split_by_nan() is the list of splitted dataframe
        trajectory_df[traj_type] = preprocess_split_eot(trajectory_df[traj_type], camera_properties_dict)
        # Get the depth, screen position in unity
        print("#"*100)
        print("[###] Get depth from XYZ motion capture using Extrinsic (Move to camera space)")
        trajectory_df[traj_type] = [get_depth(trajectory_df[traj_type][i], camera_properties_dict) for i in range(len(trajectory_df[traj_type]))]

        # Manually label if the flag is specified
        if args.label:
          print("Manually label trajectory file...")
          for j in range(len(trajectory_df[traj_type])):
            trajectory_df[traj_type][j]['EOT'] = manually_label_eot(trajectory_df[traj_type][j])
            end_points = np.where(trajectory_df[traj_type][j]['EOT'] == True)[-1][-1] + 1
            trajectory_df[traj_type][j] = trajectory_df[traj_type][j][:end_points]
        elif args.load_label:
          print("Loading label file...")
          labeled_eot = np.load(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}_EOT_Labeled.npy".format(traj_type, trial_index[i]), allow_pickle=True)
          print("#N trajectory : ", len(trajectory_df[traj_type]))
          print("#N labeled EOT : ", labeled_eot.shape[0])
          for idx in range(labeled_eot.shape[0]):
            print("Trajectory : ", trajectory_df[traj_type][idx].shape)
            print("Loaded labeled : ", labeled_eot[idx].shape)
            trajectory_df[traj_type][idx]['EOT'][:labeled_eot[idx].shape[0]] = labeled_eot[idx].reshape(-1).astype(bool)
            end_points = np.where(trajectory_df[traj_type][idx]['EOT'] == True)[-1][-1] + 1
            trajectory_df[traj_type][idx] = trajectory_df[traj_type][idx][:end_points]

    print("Trajectory type in Trial{} : {}".format(trial_index[i], trajectory_df.keys()))

    # Cast to npy format
    # This will compute the displacement and rearrange_index into the order following in unity
    trajectory_npy = computeDisplacement(trajectory_df, trajectory_type=trajectory_df.keys())
    # Save the .npy files in shape and formation that ready to use in train/test
    for traj_type in trajectory_df.keys():
      print("Preprocessed trajectory shape : ", trajectory_npy[traj_type].shape)
      np.save(file=output_path + "/{}Trajectory_Trial{}.npy".format(traj_type, trial_index[i]), arr=trajectory_npy[traj_type])

      # Save the .npy of labeled EOT
      if args.label:
        print("EOT labeled shape : ", trajectory_npy[traj_type].shape)
        eot_labeled = np.array([trajectory_npy[traj_type][idx][:, [-1]] for idx in range(trajectory_npy[traj_type].shape[0])])
        np.save(file=dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}_EOT_Labeled".format(traj_type, trial_index[i]), arr=eot_labeled)


    # Visualize the trajectory
    vis_idx = np.random.randint(0, len(trajectory_npy['Mixed']))
    trajectory_plot = pd.DataFrame(data=np.cumsum(trajectory_npy['Mixed'][vis_idx].copy(), axis=0),
                                   columns=['ball_plane_x', 'ball_plane_y', 'ball_plane_z', 'ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'ball_camera_depth', 'ball_camera_y', 'ball_camera_x', 'ball_screen_opencv_u', 'ball_screen_opencv_v', 'EOT'])

    trajectory_plot = proj_unproj_verify(trajectory_df=trajectory_plot, cam_params=camera_properties_dict)
    fig = visualize_trajectory(trajectory_df=trajectory_plot)
    fig.show()
