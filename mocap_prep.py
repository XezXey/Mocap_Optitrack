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
parser.add_argument('--unit', dest='unit', help='Scaling unit', required=True, type=str)
parser.add_argument('--scale', dest='scale', help='Scale the uv-coordinates', action='store_true')
parser.add_argument('--no_scale', dest='scale', help='Scale the uv-coordinates', action='store_false')
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
  projectionMatrix = np.array(cam_params['projectionMatrix_unity'])
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
  projectionMatrix = np.array(cam_params['projectionMatrix_unity'])
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
  screen_space[:, 0] /= camera_properties_dict['w']
  screen_space[:, 1] /= camera_properties_dict['h']
  ndc_space = screen_space.copy()
  ndc_space = (ndc_space * 2) - 1

  # NDC space -> CAMERA space (u*depth, v*depth, depth, 1)
  ndc_space = np.hstack((np.multiply(ndc_space[:, 0].reshape(-1, 1), depth), 
                        np.multiply(ndc_space[:, 1].reshape(-1, 1), depth), 
                        depth, 
                        np.ones(ndc_space.shape[0]).reshape(-1, 1)))

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
  print("Intrinsic : \n", I[0][0])
  print("Extrinsic : \n", E[0][0])
  print("Extrinsic_inv : \n", np.linalg.inv(E[0][0]))
  print("World bound : \n", world_bound)
  print("Ray : \n", ray)
  return world_bound, ray

def visualize_trajectory(trajectory_df, fig, group):
  print('=' * 100)
  print('[#] Visualization')
  print('Shape : ', trajectory_df.shape)
  print('Example of data : ')
  print(trajectory_df[['ball_screen_unity_u_project', 'ball_screen_unity_v_project', 'ball_plane_x', 'ball_plane_y', 'ball_plane_z']].tail(5))
  print('=' * 100)
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

  fig.show()
  return fig

def load_config_file(folder_name, idx, cam):
  config_file_extrinsic = folder_name + "/motioncapture_camParams_Trial_{}_{}.yml".format(idx, args.unit)
  extrinsic = cv2.FileStorage(config_file_extrinsic, cv2.FILE_STORAGE_READ)
  config_file_intrinsic = folder_name + "/motioncapture_camIntrinsic_{}.yml".format(args.unit)
  intrinsic = cv2.FileStorage(config_file_intrinsic, cv2.FILE_STORAGE_READ)
  nodes_extrinsic = [cam, "w", "h"]
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
    elif 'K' in each_node:
      '''
       MOCAP
       - ROTATION Matrix: World-space rotation from camera extrinsic
       - Translation Vector : camera position in world-space
      '''

      #rotation = extrinsic.getNode(each_node).mat()[:-1, :-1]
      #rotation_inv = np.linalg.inv(rotation)
      #rotation_inv = np.linalg.inv(extrinsic.getNode(each_node).mat()[:-1, :-1])

      rotation_inv = extrinsic.getNode(each_node).mat()[:-1, :-1]
      rotation = np.linalg.inv(rotation_inv)


      translation = extrinsic.getNode(each_node).mat()[:-1, -1].reshape(-1, 1) / scaling

      camera_properties_dict['Inversed_Extrinsic'] = np.concatenate((np.concatenate((rotation_inv, translation), axis=1), np.array([[0, 0, 0, 1]])))

      camera_properties_dict['Extrinsic'] = np.linalg.inv(camera_properties_dict['Inversed_Extrinsic'].copy())

      # camera_properties_dict['Inversed_Extrinsic'] = np.concatenate((np.concatenate((rotation, translation), axis=1), np.array([[0, 0, 0, 1]])))
      # camera_properties_dict['Extrinsic'] = np.linalg.inv(camera_properties_dict['Inversed_Extrinsic'])

  # ========== Make it similar to unity ==========
  # [#] Note@13 Aug 2020 : Using Extrinsic from Mocap -> Set to unity get the Inversed of rotation Then i need to inversed it.
  # [#] Note@15 Aug 2020 : Unity accept the inversed extrinsic of rotation and translation part.
  # Change of basis to make the extrinsic in mocap be the same as unity
  # worldToCamera = [[1, 0, 0], [0, 1, 0], [0, 0, -1]] @ Rotation_unity
  # Then rotation_unity = change_basis_matrix @ change_basis_matrix @ rotation_optitrack @ change_basis_matrix = I @ rotation_optitrack @ change_basis_matrix
  rotation_unity_inv = np.linalg.inv(rotation @ change_basis_matrix)
  translation_unity = np.array([translation[0], translation[1], -translation[2]])
  camera_properties_dict['Inversed_Extrinsic_unity'] = np.concatenate((np.concatenate((rotation_unity_inv, translation_unity), axis=1), np.array([[0, 0, 0, 1]])))
  camera_properties_dict['Extrinsic_unity'] = np.linalg.inv(camera_properties_dict['Inversed_Extrinsic_unity'].copy())

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

  I_unity = camera_properties_dict['projectionMatrix'].copy()
  I_unity[2, :] = I_unity[3, :]
  I_unity[3, :] = np.array([0, 0, 0, 1])
  camera_properties_dict['projectionMatrix_unity'] = I_unity

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

  print("E")
  for i in range(camera_properties_dict['Extrinsic'].shape[0]):
    tmp = camera_properties_dict['Extrinsic'][i]
    print("realWorldExtrinsic.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))

  print("Einv")
  for i in range(camera_properties_dict['Inversed_Extrinsic'].shape[0]):
    tmp = camera_properties_dict['Inversed_Extrinsic'][i]
    print("realWorldExtrinsic.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))


  print("E_unity")
  for i in range(camera_properties_dict['Extrinsic_unity'].shape[0]):
    tmp = camera_properties_dict['Extrinsic_unity'][i]
    print("E.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))


  print("Einv_unity")
  for i in range(camera_properties_dict['Inversed_Extrinsic_unity'].shape[0]):
    tmp = camera_properties_dict['Inversed_Extrinsic_unity'][i]
    print("E.SetRow({}, new Vector4({}f, {}f, {}f, {}f));".format(i, tmp[0], tmp[1], tmp[2], tmp[3]))

  input()
  return camera_properties_dict

def split_trajectory(trajectory_df, camera_properties_dict):
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
  # Remove some dataset that have an start-end artifact
  ground_annotated = [np.where((trajectory_split[i]['ball_plane_y'].values < ground_threshold) == True)[0] for i in range(len(trajectory_split))]

  trajectory_split_clean = []
  # Add the EOT flag at the last rows
  print("[###] Preprocessing...")
  print("Filter the uncompleteness projectile (Remove : start from air, too short, and outside the screen)")
  for idx in range(len(trajectory_split)):
    # print("IDX : ", idx, " => length ground_annotated : ", len(ground_annotated[idx]), ", length trajectory : ", len(trajectory_split[idx]))
    if len(ground_annotated[idx]) <= 2:
      # Not completely projectile
      #print("At {} : Not completely projectile ===> Continue...".format(idx))
      continue
    else :
      trajectory_split[idx]['ball_screen_opencv_u'] = trajectory_split[idx]['ball_screen_opencv_u'].astype(float)
      trajectory_split[idx]['ball_screen_opencv_v'] = trajectory_split[idx]['ball_screen_opencv_v'].astype(float)
      # Invert the world coordinates to make the same convention
      trajectory_split[idx]['ball_plane_z'] *= -1
      if args.scale:
        # These scale came from possible range in unity divided by possible range at mocap
        trajectory_split[idx]['ball_plane_x'] = (trajectory_split[idx]['ball_plane_x'] * scaling)
        trajectory_split[idx]['ball_plane_y'] = (trajectory_split[idx]['ball_plane_y'] * scaling)
        trajectory_split[idx]['ball_plane_z'] = (trajectory_split[idx]['ball_plane_z'] * scaling)
      # trajectory_split[idx] = trajectory_split[idx].iloc[ground_annotated[idx][0]:ground_annotated[idx][-1], :]
      trajectory_split[idx] = trajectory_split[idx].iloc[ground_annotated[idx][0]:, :]
      trajectory_split[idx].reset_index(drop=True, inplace=True)

      # Initialize the EOT flag to 0
      trajectory_split_clean.append(trajectory_split[idx])

  print("Number of all trajectory : ", len(trajectory_split))

  # Remove the point that stay outside the screen and too short(blinking point).
  trajectory_split_clean = [each_traj for each_traj in trajectory_split_clean if (np.all(each_traj[['ball_screen_opencv_u', 'ball_screen_opencv_v']] > np.finfo(float).eps) and np.all(each_traj[['ball_screen_opencv_u']] < camera_properties_dict['w']) and np.all(each_traj[['ball_screen_opencv_v']] < camera_properties_dict['h']) and each_traj.shape[0] >= length_threshold)]

  print("Number of all trajectory(Inside the screen) : ", len(trajectory_split_clean))
  print("#" * 100)
  return trajectory_split_clean

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
  col_names = ["frame", "camera_index", "camera_serial", "Time", "ball_screen_opencv_u", "ball_screen_opencv_v", "ball_plane_x", "ball_plane_y", "ball_plane_z"]
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
      camera_properties_dict = load_config_file(folder_name=dataset_folder[i], idx=trial_index[i], cam=c_)
      # Check and read .csv file in the given directory
      for traj_type in trajectory_type:
        if os.path.isfile(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i])):
          trajectory_df[traj_type] = pd.read_csv(dataset_folder[i] + "/{}Trajectory_ball_motioncapture_Trial{}.csv".format(traj_type, trial_index[i]), names=col_names, delimiter=',')
          # Preprocess by splitting, adding EOT and remove the first incomplete trajectory
          trajectory_df[traj_type] = split_trajectory(trajectory_df[traj_type], camera_properties_dict)
          # Get the depth, screen position in unity
          print("#"*100)
          print("[###] Get depth from XYZ motion capture using Extrinsic (Move to camera space)")
          trajectory_df[traj_type] = [get_depth(trajectory_df[traj_type][i], camera_properties_dict) for i in range(len(trajectory_df[traj_type]))]

      print("Trajectory type in Trial{} : {}".format(trial_index[i], trajectory_df.keys()))

      # Cast to npy format
      trajectory_npy = to_npy(trajectory_df, trajectory_type=trajectory_df.keys(), cam_params=camera_properties_dict)
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

      trajectory_plot = proj_unproj_verify(trajectory_df=trajectory_plot, cam_params=camera_properties_dict)
      fig = visualize_trajectory(trajectory_df=trajectory_plot, fig=fig, group=c_)
    fig.show()