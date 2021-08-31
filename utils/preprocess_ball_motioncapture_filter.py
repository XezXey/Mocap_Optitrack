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

def get_savepath(output_path, dataset_folder):
  if output_path == None:
    output_path = dataset_folder
  else:
    output_path = args.output_path
    if not os.path.exists(output_path):
      os.makedirs(output_path)
  return output_path

def get_unity_parameters(cam_params):
  print('='*37 + 'Parameters to set in Unity' + '='*37)
  print('Input inversed extrinsic (cameraToWorldMatrix) : \n', cam_params['Extrinsic'])
  print('Extrinsic (worldToCameraMatrix) : \n', np.linalg.inv(cam_params['Extrinsic']))
  T = cam_params['Extrinsic'][:3, -1] # Translation
  R = cam_params['Extrinsic'][:3, :3].copy()
  # print("Before inverse : \n", R)
  R[:, 0] *= -1
  # R[:, 1] *= -1
  R[:, 2] *= -1
  print("After inverse : \n", R)
  R = Rotation.from_matrix(R) # Rotation
  euler = R.as_euler('zxy', degrees=True)
  # Focal length
  fx = cam_params['Intrinsic'][0, 0]
  fy = cam_params['Intrinsic'][1, 1]
  fov_x = 2 * math.degrees(math.atan((cam_params['w']/(2*fx))))
  fov_y = 2 * math.degrees(math.atan((cam_params['h']/(2*fy))))
  print('Euler angles (zxy) : ', euler)
  print('Translation : ', T)
  print('Fov_x : ', fov_x)
  print('Fov_y : ', fov_y)
  print('='*100)

def get_depth(trajectory_df, cam_params):
  '''
  This function will return the points in camera space and screen space
  - This projectionMatrix use in unity screen convention : (0, 0) is in the bottom-left
  '''
  worldToCameraMatrix = np.array(cam_params['Extrinsic_unity'])
  cameraToWorldMatrix = np.linalg.inv(worldToCameraMatrix)
  projectionMatrix = np.array(cam_params['projectionMatrix'])
  # Remove the clip space that we don't use in unprojection of real cameras
  projectionMatrix[2, :] = projectionMatrix[3, :]
  projectionMatrix[3, :] = np.array([0, 0, 0, 1])
  # Plane space
  plane_space = np.hstack((trajectory_df[['ball_plane_x', 'ball_plane_y', 'ball_plane_z']].values, np.ones(trajectory_df.shape[0]).reshape(-1, 1)))
  # Uncomment this to test the particular points
  # plane_space = np.array([[0, 0, 0, 1],
                          # [-21.2, 0.1, 0, 1],
                          # [-21.2, 0.1, 33, 1],
                          # [2, 0, 33, 1],
                          # [23.5, 0, 33, 1],
                          # [-20.5, 0, -11.1, 1],
                          # [-22.9, 0, 39.6, 1]])
  # WorldToCameraMatrix : Plane space -> Camera space (u*depth, v*depth, depth, 1)
  camera_space = plane_space @ worldToCameraMatrix.T
  trajectory_df['ball_camera_x'] = camera_space[:, 0]
  trajectory_df['ball_camera_y'] = camera_space[:, 1]
  trajectory_df['ball_camera_depth'] = camera_space[:, 2]
  # projectionMatrix : Camera space -> NDC Space
  # print("DEPTH : ", camera_space)
  # print("DEPTH DISPLACEMENT : ", np.diff(camera_space, axis=0))
  # print("PLANE : ", plane_space)
  # In the unity, projection matrix didn't give the output as the screen space but it will in the NDC space(We'll see the world in range(-1, 1))
  # Then we need to unnormalized it to ge the screen space
  ndc_space = camera_space @ projectionMatrix.T
  # Get the screen coordinates
  u = ((ndc_space[:, 0]/ndc_space[:, 2]) + 1) * (camera_properties_dict['w']/2)
  v = ((ndc_space[:, 1]/ndc_space[:, 2]) + 1) * (camera_properties_dict['h']/2)
  # print("U :", u)
  # print("V :", v)
  # print("Depth : ", camera_space[:, -2])
  trajectory_df['ball_screen_unity_u_project'] = u
  trajectory_df['ball_screen_unity_v_project'] = camera_properties_dict['h'] - v

  return trajectory_df

def unproject(trajectory_df, cam_params):
  '''
  Trying to unproject the point by given the (u, v and depth) -> world coordinates
  - This projectionMatrix use in unity screen convention : (0, 0) is in the bottom-left
  '''
  eps = np.finfo(float).eps
  # Get the projectionMatrix
  projectionMatrix = np.array(cam_params['projectionMatrix'])
  # Remove the clip space that we don't use in unprojection of real cameras
  projectionMatrix[2, :] = projectionMatrix[3, :]
  projectionMatrix[3, :] = np.array([0, 0, 0, 1])
  # Get the camera Extrinsic
  worldToCameraMatrix = np.array(cam_params['Extrinsic_unity'])
  cameraToWorldMatrix = np.linalg.inv(worldToCameraMatrix)
  # The (u, v and depth) from the get_depth() function
  depth = trajectory_df['ball_camera_depth'].values.reshape(-1, 1)
  u = trajectory_df['ball_screen_unity_u_project'].values.reshape(-1, 1)
  v = trajectory_df['ball_screen_unity_v_project'].values.reshape(-1, 1)

  # The screen space will be : (u*depth, v*depth, depth, 1)
  screen_space = np.hstack((np.multiply(u, depth), np.multiply(v, depth), depth, np.ones(u.shape)))
  print("PARAMS SHAPE : ", projectionMatrix.shape, worldToCameraMatrix.shape)
  print("DATA SHAPE : ", screen_space.shape, depth.shape, u.shape, v.shape)

  '''
  PROJECTION
  X, Y, Z ===> U, V
  [#####] Note : Multiply the X, Y, Z point by scaling factor that use in opengl visualization because the obtained u, v came from the scaled => Scaling = 10
  [#####] Note : Don't multiply if use the perspective projection matrix from intrinsic
  '''

  # Get the world coordinates : (X, Y, Z, 1)
  plane_space = np.hstack((trajectory_df[['ball_plane_x', 'ball_plane_y', 'ball_plane_z']], np.ones(trajectory_df.shape[0]).reshape(-1, 1)))
  # worldToCameraMatrix : Plane space -> Camera space (u*depth, v*depth, depth, 1)
  camera_space = plane_space @ worldToCameraMatrix.T
  # projectionMatrix : Camera space -> NDC space (u and v) in range(-1, 1)
  ndc_space = camera_space @ projectionMatrix.T
  ndc_space[:, :-1] /= ndc_space[:, 2].reshape(-1, 1)
  # Get the Screen space from unnormalized the NDC space
  screen_space = ndc_space.copy()
  # We need to subtract the camera_properties_dict['w'] out of the ndc_space since the Extrinisc is inverse from the unity. So this will make x-axis swapped and need to subtract it to get the same 3D reconstructed points : 
  screen_space[:, 0] = camera_properties_dict['w'] - ((ndc_space[:, 0]) + 1) * (camera_properties_dict['w']/2)
  screen_space[:, 1] = ((ndc_space[:, 1]) + 1) * (camera_properties_dict['h']/2)
  print("[#] Equality check of screen space (u, v) : ", np.all((screen_space[:, 0].reshape(-1, 1) - u) <= eps), ", ", np.all((screen_space[:, 1].reshape(-1, 1) - v) <= eps))

  '''
  U, V, Depth ===> X, Y, Z
  '''
  # Target value
  target_camera_space = np.hstack((trajectory_df['ball_camera_x'].values.reshape(-1, 1), trajectory_df['ball_camera_y'].values.reshape(-1, 1), depth))
  target_plane_space = np.hstack((trajectory_df['ball_plane_x'].values.reshape(-1, 1), trajectory_df['ball_plane_y'].values.reshape(-1, 1),
                                  trajectory_df['ball_plane_z'].values.reshape(-1, 1)))

  # [Screen, Depth] -> Plane
  # Following the unprojection by inverse the projectionMatrix and inverse of arucoPlaneToCameraMatrix(here's cameraToWorldMatrix)
  # Normalizae the Screen space to NDC space
  # SCREEN -> NDC
  # screen_space = np.array([[888.1517, 428.5164, 54.5191, 1]])
  screen_space[:, 0] /= camera_properties_dict['w']
  screen_space[:, 1] /= camera_properties_dict['h']
  ndc_space = screen_space.copy()
  ndc_space = (ndc_space * 2) - 1

  # NDC space -> CAMERA space (u*depth, v*depth, depth, 1)
  # ndc_space = np.hstack((np.multiply(ndc_space[:, 0].reshape(-1, 1), screen_space[:, 2].reshape(-1, 1)), np.multiply(ndc_space[:, 1].reshape(-1, 1), screen_space[:, 2].reshape(-1, 1)), screen_space[:, 2].reshape(-1, 1), np.ones(ndc_space.shape[0]).reshape(-1, 1)))
  ndc_space = np.hstack((np.multiply(ndc_space[:, 0].reshape(-1, 1), depth), np.multiply(ndc_space[:, 1].reshape(-1, 1), depth), depth, np.ones(ndc_space.shape[0]).reshape(-1, 1)))
  camera_space =  ndc_space @ (np.linalg.inv(projectionMatrix)).T
  # CAMERA space -> Plane space
  # cameraToWorldMatrix[:, 0] *= -1
  # cameraToWorldMatrix[:, 2] *= -1
  plane_space =  camera_space @ cameraToWorldMatrix.T
  # Store the plane unproject to dataframe
  trajectory_df.loc[:, 'ball_plane_x_unproject'] = plane_space[:, 0]
  trajectory_df.loc[:, 'ball_plane_y_unproject'] = plane_space[:, 1]
  trajectory_df.loc[:, 'ball_plane_z_unproject'] = plane_space[:, 2]
  print('='*40 + 'Unproject' + '='*40)
  print('Screen : {}\nCamera : {}\nPlane : {}'.format(screen_space[0].reshape(-1), target_camera_space[0].reshape(-1), target_plane_space[0].reshape(-1)))
  print('\n[#]===> Target Camera unprojection : ', target_camera_space[0].reshape(-1))
  print('Screen -> Camera (By projectionMatrix) : ', camera_space[0].reshape(-1))
  print('[Screen, depth] : ', camera_space[0].reshape(-1))
  print('\n[#]===> Target Plane unprojection : ', target_plane_space[3])
  print('Screen -> Plane (By inverse the projectionMatrix) : ', plane_space[3])
  print('='*89)
  return trajectory_df

def visualize_trajectory(trajectory_df):
  window_size = 77
  order = 2
  # marker properties of uv, xyz
  marker_dict_gt = dict(color='rgba(0, 0, 255, 0.4)', size=7)
  marker_dict_pred = dict(color='rgba(255, 0, 0, 0.4)', size=7)
  # marker properties of displacement
  marker_dict_u = dict(color='rgba(200, 0, 0, 0.4)', size=7)
  marker_dict_v = dict(color='rgba(0, 125, 0, 0.4)', size=7)
  marker_dict_depth = dict(color='rgba(0, 0, 255, 0.4)', size=7)
  marker_dict_u_sav = dict(color='rgba(200, 100, 128, 0.7)', size=7)
  marker_dict_v_sav = dict(color='rgba(34, 100, 39, 0.7)', size=7)
  marker_dict_depth_sav = dict(color='rgba(128, 100, 255, 0.7)', size=7)
  marker_dict_uv_filtered = dict(color='rgba(0, 0, 255, 0.4)', size=7)
  marker_dict_eot = dict(color='rgba(0, 0, 255, 0.7)', size=7)
  fig = make_subplots(rows=2, cols=2, specs=[[{'type':'scatter'}, {'type':'scatter3d'}], [{'type':'scatter'}, {'type':'scatter'}]])
  # Screen
  fig.add_trace(go.Scatter(x=trajectory_df['ball_screen_unity_u_project'], y=trajectory_df['ball_screen_unity_v_project'], mode='markers+lines', marker=marker_dict_pred, name="Trajectory (Screen coordinates unproject)"), row=1, col=1)
  # Screen - Convolved
  # fig.add_trace(go.Scatter(x=np.convolve(trajectory_df['ball_screen_unity_u_project'], np.ones(window_size)/window_size, 'same'), y=np.convolve(trajectory_df['ball_screen_unity_v_project'], np.ones(window_size)/window_size, 'same'), mode='markers+lines', marker=marker_dict_uv_filtered, name="Trajectory screen convolved"), row=1, col=1)
  # Screen - Savgol
  fig.add_trace(go.Scatter(x=signal.savgol_filter(trajectory_df['ball_screen_unity_u_project'], window_size, order), y=signal.savgol_filter(trajectory_df['ball_screen_unity_v_project'], window_size, order), mode='markers+lines', marker=marker_dict_uv_filtered, name="Trajectory screen savgol"), row=1, col=1)
  # Displacement before filter
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_screen_unity_u_project'])-1), y=np.diff(trajectory_df['ball_screen_unity_u_project']), mode='lines', marker=marker_dict_u, name="Displacement - u"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_screen_unity_v_project'])-1), y=np.diff(trajectory_df['ball_screen_unity_v_project']), mode='lines', marker=marker_dict_v, name="Displacement - v"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_camera_depth'])-1), y=np.diff(trajectory_df['ball_camera_depth']), mode='lines', marker=marker_dict_depth, name="Displacement - Depth"), row=2, col=1)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_camera_depth'])-1), y=trajectory_df['ball_camera_depth'], mode='lines', marker=marker_dict_depth, name="Depth"), row=2, col=1)
  # Displacement after filter by convolution
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_screen_unity_u_project'])-1), y=np.convolve(np.diff(trajectory_df['ball_screen_unity_u_project']), np.ones(window_size)/window_size, 'same'), mode='lines', marker=marker_dict_u, name="Smoothed Displacement - u"), row=2, col=2)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_screen_unity_v_project'])-1), y=np.convolve(np.diff(trajectory_df['ball_screen_unity_v_project']), np.ones(window_size)/window_size, 'same'), mode='lines', marker=marker_dict_v, name="Smoothed Displacement - v"), row=2, col=2)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_camera_depth'])-1), y=np.convolve(np.diff(trajectory_df['ball_camera_depth']), np.ones(window_size)/window_size, 'same'), mode='lines', marker=marker_dict_depth, name="Smoothed Displacement - depth"), row=2, col=2)
  # Displacement after filter by savgol
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_screen_unity_u_project'])-1), y=signal.savgol_filter(np.diff(trajectory_df['ball_screen_unity_u_project']), window_size, order), mode='lines', marker=marker_dict_u_sav, name="SAV : Displacement - u"), row=2, col=2)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_screen_unity_v_project'])-1), y=signal.savgol_filter(np.diff(trajectory_df['ball_screen_unity_v_project']), window_size, order), mode='lines', marker=marker_dict_v_sav, name="SAV : Displacement - v"), row=2, col=2)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['ball_camera_depth'])-1), y=signal.savgol_filter(np.diff(trajectory_df['ball_camera_depth']), window_size, order), mode='lines', marker=marker_dict_depth_sav, name="SAV : Displacement - depth"), row=2, col=2)
  # EOT
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['EOT'])), y=trajectory_df['EOT'], mode='lines', marker=marker_dict_eot, name="EOT"), row=2, col=2)
  fig.add_trace(go.Scatter(x=np.arange(len(trajectory_df['EOT'])), y=trajectory_df['EOT'], mode='lines', marker=marker_dict_eot, name="EOT"), row=2, col=1)
  # World
  fig.add_trace(go.Scatter3d(x=trajectory_df['ball_plane_x'], y=trajectory_df['ball_plane_y'], z=trajectory_df['ball_plane_z'], mode='markers+lines', marker=marker_dict_gt, name="Motion capture (World coordinates)"), row=1, col=2)
  if not args.label:
    fig.add_trace(go.Scatter3d(x=trajectory_df['ball_plane_x_unproject'], y=trajectory_df['ball_plane_y_unproject'], z=trajectory_df['ball_plane_z_unproject'], mode='markers+lines', marker=marker_dict_pred, name="Unproject trajectory (World coordinates)"), row=1, col=2)
  return fig

def load_config_file(folder_name, idx):
  config_file_extrinsic = folder_name + "/motioncapture_camParams_Trial_{}.yml".format(idx)
  extrinsic = cv2.FileStorage(config_file_extrinsic, cv2.FILE_STORAGE_READ)
  config_file_intrinsic = folder_name + "/motioncapture_camIntrinsic.yml".format(idx)
  intrinsic = cv2.FileStorage(config_file_intrinsic, cv2.FILE_STORAGE_READ)
  nodes_extrinsic = ["K2", "w", "h"]
  nodes_intrinsic = ["K"]
  camera_properties_dict = {}

  # Load an extrinsic
  for each_node in nodes_extrinsic:
    if each_node == "w" or each_node == "h":
      camera_properties_dict[each_node] = extrinsic.getNode(each_node).real()
    elif each_node == "K2":
      camera_properties_dict["Inversed_Extrinsic"] = extrinsic.getNode(each_node).mat()
      camera_properties_dict["Inversed_Extrinsic"][:-1, :-1] = np.linalg.inv(camera_properties_dict["Inversed_Extrinsic"][:-1, :-1])
      camera_properties_dict["Inversed_Extrinsic"][:-1, -1] /= 10   # Scale the scale
      camera_properties_dict["Extrinsic"] = np.linalg.inv(camera_properties_dict["Inversed_Extrinsic"])
      # Divide the translation by 10 because data acquisition I've scaled it up by 10.

  # ========== Make it similar to unity ==========
  # Change of basis to make the extrinsic in mocap be the same as unity
  camera_properties_dict['Extrinsic_unity'] = camera_properties_dict['Extrinsic']
  camera_properties_dict['Extrinsic_unity'][2, 3] *= -1 # Inverse the camera position (Rearrange the basis direction)
  # Inverse the extrinsic
  camera_properties_dict['Extrinsic_unity'][0, 2] *= -1
  camera_properties_dict['Extrinsic_unity'][1, 2] *= -1
  camera_properties_dict['Extrinsic_unity'][2, 0] *= -1
  camera_properties_dict['Extrinsic_unity'][2, 1] *= -1
  camera_properties_dict['Extrinsic_unity'][2, :] *= -1

  print("\nInversed Extrinsic : ")
  print(camera_properties_dict['Inversed_Extrinsic'])
  print("Extrinsic : ")
  print(camera_properties_dict['Extrinsic'])
  print("Unity Extrinsic : \n", camera_properties_dict['Extrinsic_unity'])
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
  return camera_properties_dict

def preprocess_split_eot(trajectory_df, camera_properties_dict):
  '''
  This function will clean the data by split the trajectory, remove some trajectory that too short(maybe an artifact and
  make the first and last point start on ground (ground_threshold = 0.025 in y-axis)
  '''
  ground_threshold = 0.025
  smooth_length = 200
  # Split by nan
  trajectory_split = np.split(trajectory_df, np.where(np.isnan(trajectory_df['ball_plane_x']))[0])
  # removing NaN entries
  trajectory_split = [each_traj[~np.isnan(each_traj['ball_plane_x'])] for each_traj in trajectory_split if not isinstance(each_traj, np.ndarray)]
  # removing empty DataFrames
  trajectory_split = [each_traj for each_traj in trajectory_split if not each_traj.empty]
  # removing too short(blinking point) in DataFrames
  trajectory_split = [each_traj for each_traj in trajectory_split if (len(each_traj) > 300)]
  # Remove some dataset that have an start artifact and aend artifact
  ground_annotated = [np.where((trajectory_split[i]['ball_plane_y'].values < ground_threshold) == True)[0] for i in range(len(trajectory_split))]

  trajectory_split_clean = []
  # Add the EOT flag at the last rows
  print("length : ", len(trajectory_split))
  for idx in range(len(trajectory_split)):
    # print("IDX : ", idx, " => length ground_annotated : ", len(ground_annotated[idx]), ", length trajectory : ", len(trajectory_split[idx]))
    if len(ground_annotated[idx]) <= 2:
      # Not completely projectile
      print("Continue...")
      continue
    else :
      trajectory_split[idx]['ball_screen_opencv_u'] = trajectory_split[idx]['ball_screen_opencv_u'].astype(float)
      trajectory_split[idx]['ball_screen_opencv_v'] = trajectory_split[idx]['ball_screen_opencv_v'].astype(float)
      # Scaling the trajectory in real world to be close to unity scale
      if args.scale:
        # These scale came from possible range in unity divided by possible range at mocap
        trajectory_split[idx]['ball_plane_x'] *= 13.5
        trajectory_split[idx]['ball_plane_y'] *= 15
        trajectory_split[idx]['ball_plane_z'] *= 12
      trajectory_split[idx] = trajectory_split[idx].iloc[ground_annotated[idx][0]:ground_annotated[idx][-1], :]
      trajectory_split[idx].reset_index(drop=True, inplace=True)
      if args.smooth :
        trajectory_split[idx] = trajectory_split[idx].iloc[:len(trajectory_split[idx])-smooth_length, :]
      trajectory_split[idx]['EOT'] = np.zeros(trajectory_split[idx].shape[0], dtype=bool)
      trajectory_split_clean.append(trajectory_split[idx])

  print("Number of all trajectory : ", len(trajectory_split))
  trajectory_split_clean = [each_traj for each_traj in trajectory_split_clean if np.all(each_traj[['ball_screen_opencv_u', 'ball_screen_opencv_v']] > np.finfo(float).eps)]
  print("Number of all trajectory(In the screen) : ", len(trajectory_split_clean))
  # for e in trajectory_split_clean:
    # print(e[['ball_screen_opencv_u', 'ball_screen_opencv_v']])

  return trajectory_split_clean

def manually_label_eot(trajectory):
  fig = visualize_trajectory(trajectory)
  fig.show()
  eot_annotated = [int(x) for x in input("Input the index of EOT (Available indexing = {}) : ".format(trajectory.index)).split()]
  trajectory['EOT'][eot_annotated] = True
  return trajectory['EOT'].values

def computeDisplacement(trajectory_df, trajectory_type):
  # Compute the displacement
  window_size = 77
  order = 2
  drop_cols = ["frame", "camera_index", "camera_serial", "Time", "EOT"]
  trajectory_npy = trajectory_df.copy()
  for traj_type in trajectory_type:
    trajectory_npy[traj_type] = [np.hstack((np.vstack((trajectory_df[traj_type][i].drop(drop_cols, axis=1).iloc[0, :].values,
                                                       np.diff(trajectory_df[traj_type][i].drop(drop_cols, axis=1).values, axis=0))),
                                            trajectory_df[traj_type][i]['EOT'].values.reshape(-1, 1))) for i in range(len(trajectory_df[traj_type]))]

    eot_filtered_index = [np.where(trajectory_npy[traj_type][idx][:, -1] == True)[0][-1] for idx in range(len(trajectory_df[traj_type]))]
    trajectory_npy[traj_type] = [trajectory_npy[traj_type][idx][:eot_filtered_index[idx]+1, :] for idx in range(len(trajectory_df[traj_type]))]

    # Before reindex columns would be : ['ball_screen_opencv_u(0)', 'ball_screen_opencv_v(1)', 'ball_plane_x(2)', 'ball_plane_y(3)', 'ball_plane_z(4)',
       # 'ball_camera_x(5)', 'ball_camera_y(6)', 'ball_camera_depth(7)', 'ball_screen_unity_u_project(8)', 'ball_screen_unity_v_project(9)', 'EOT(10)']

  # Cast to ndarray (Bunch of trajectory)
  rearrange_index = [2, 3, 4, 8, 9, 7, 6, 5, 0, 1, 10]  # x=0, y=1, z=2, u=3, v=4, depth=5, eot=-1
  # Finally the columns would be : ['ball_plane_x(2)', 'ball_plane_y(3)', 'ball_plane_z(4)', 'ball_screen_unity_u_project(8)', 
  # 'ball_screen_unity_v_project(9)', 'ball_camera_depth(7)', 'ball_camera_y(6)', 'ball_camera_x(5)', 'ball_screen_opencv_u(0)', 'ball_screen_opencv_v(1)', 'EOT(10)']

  # Reindex following : x=0, y=1, z=2, u=3, v=4, depth=5, eot=-1
  trajectory_npy[traj_type] = np.array([trajectory_npy[traj_type][i][:, rearrange_index] for i in range(len(trajectory_npy[traj_type]))])
  # Remove the trajectory that have lengths <= 100  
  trajectory_npy[traj_type] = np.array([trajectory for trajectory in trajectory_npy[traj_type] if trajectory.shape[0] > 100])
  # Filter the UV
  if args.filter:
    for i in range(len(trajectory_npy[traj_type])):
      # Convolve on u(4) and v(5)
      plt.plot(trajectory_npy[traj_type][i][1:, 3], c='r')
      plt.plot(trajectory_npy[traj_type][i][1:, 4], c='g')
      # Use convolution
      # trajectory_npy[traj_type][i][1:, 4] = np.array(np.convolve(trajectory_npy[traj_type][i][1:, 4], np.ones(window_size)/window_size, 'same'))
      # trajectory_npy[traj_type][i][1:, 5] = np.array(np.convolve(trajectory_npy[traj_type][i][1:, 5], np.ones(window_size)/window_size, 'same'))
      # Use savgol filter
      print("LENGTH : ", len(trajectory_npy[traj_type][i]))
      trajectory_npy[traj_type][i][1:, 3] = signal.savgol_filter(trajectory_npy[traj_type][i][1:, 3], window_size, order)
      trajectory_npy[traj_type][i][1:, 4] = signal.savgol_filter(trajectory_npy[traj_type][i][1:, 4], window_size, order)
      plt.plot(trajectory_npy[traj_type][i][1:, 3], c='b')
      plt.plot(trajectory_npy[traj_type][i][1:, 4], c='c')
  return trajectory_npy

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Preprocess and visualize the trajectory in real world')
  parser.add_argument('--dataset_path', type=str, help='Specify path to dataset')
  parser.add_argument('--process_trial_index', dest='process_trial_index', help='Process trial at given idx only', default=None)
  parser.add_argument('--output_path', type=str, help='Specify output path to save dataset')
  parser.add_argument('--vis', dest='vis', help='Visualize the trajectory', action='store_true')
  parser.add_argument('--no_vis', dest='vis', help='Visualize the trajectory', action='store_false')
  parser.add_argument('--filter', dest='filter', help='Filter the uv-coordinates', action='store_true')
  parser.add_argument('--no_filter', dest='filter', help='Filter the uv-coordinates', action='store_false')
  parser.add_argument('--scale', dest='scale', help='Scale the uv-coordinates', action='store_true')
  parser.add_argument('--no_scale', dest='scale', help='Scale the uv-coordinates', action='store_false')
  parser.add_argument('--smooth', dest='smooth', help='Smooth the uv-coordinates', action='store_true')
  parser.add_argument('--no_smooth', dest='smooth', help='Smooth the uv-coordinates', action='store_false')
  parser.add_argument('--label', dest='label', help='Manually Label UV -> EOT', action='store_true')
  parser.add_argument('--no_label', dest='label', help='No Label UV -> EOT', action='store_false')
  parser.add_argument('--load_label', dest='load_label', help='Loading the label and concatenate to the preprocessed trajectory', action='store_true')
  args = parser.parse_args()
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
    print("Camera properties : ", camera_properties_dict)
    # Check and read .csv file in the given directory
    for traj_type in trajectory_type:
      if os.path.isfile(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i])):
        trajectory_df[traj_type] = pd.read_csv(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i]), names=col_names, delimiter=',')

        # Preprocess by splitting, adding EOT and remove the first incomplete trajectory(throwing)
        # The return from split_by_nan() is the list of splitted dataframe
        trajectory_df[traj_type] = preprocess_split_eot(trajectory_df[traj_type], camera_properties_dict)
        # Get the depth, screen position in unity
        trajectory_df[traj_type] = [get_depth(trajectory_df[traj_type][i], camera_properties_dict) for i in range(len(trajectory_df[traj_type]))]
        if args.label:
          print("Manually label trajectory file...")
          for j in range(len(trajectory_df[traj_type])):
            trajectory_df[traj_type][j]['EOT'] = manually_label_eot(trajectory_df[traj_type][j])
        elif args.load_label:
          print("Loading label file...")
          labeled_eot = np.load(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}_EOT_Labeled.npy".format(traj_type, trial_index[i]), allow_pickle=True)
          for idx in range(labeled_eot.shape[0]):
            trajectory_df[traj_type][idx]['EOT'][:labeled_eot[idx].shape[0]] = labeled_eot[idx].reshape(-1).astype(bool)

    print("Trajectory type in Trial{} : {}".format(trial_index[i], trajectory_df.keys()))
    # Get the parameters that we need to set in unity : Intrinsic and Extrinsic
    get_unity_parameters(camera_properties_dict)

    # Cast to npy format
    # This will compute the displacement and rearrange_index into the order following in unity
    trajectory_npy = computeDisplacement(trajectory_df, trajectory_type=trajectory_df.keys())
    # Save the .npy files in shape and formation that ready to use in train/test
    for traj_type in trajectory_df.keys():
      print("Preprocessed trajectory shape : ", trajectory_npy[traj_type].shape)
      print(trajectory_npy[traj_type][0][:2, [4, 5]])
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

    # With remove the first local minima 
    trajectory_plot = unproject(trajectory_df=trajectory_plot, cam_params=camera_properties_dict)
    fig = visualize_trajectory(trajectory_df=trajectory_plot)
    fig.show()

    # Without remove the first local minima 
    trajectory_plot = unproject(trajectory_df=trajectory_df['Mixed'][vis_idx], cam_params=camera_properties_dict)
    fig = visualize_trajectory(trajectory_df=trajectory_plot)
    fig.show()

