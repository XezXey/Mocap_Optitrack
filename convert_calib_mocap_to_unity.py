import numpy as np
import json, yaml
import argparse
import glob
import re
import tqdm
import cv2

parser = argparse.ArgumentParser(description='Preprocess and visualize the trajectory in real world')
parser.add_argument('--calib_path', type=str, help='Specify a path to calibration file', required=True)
parser.add_argument('--out_path', type=str, help='Specify an output path', required=True)
parser.add_argument('--unit', dest='unit', help='Scaling unit', required=True, type=str)
args = parser.parse_args()


def load_config_file(calib_path):
  config_file_E = calib_path + "/motioncapture_camParams_Trial_#n_{}.yml".format(args.unit)
  Einv = cv2.FileStorage(config_file_E, cv2.FILE_STORAGE_READ)
  config_file_I = calib_path + "/motioncapture_camIntrinsic_{}.yml".format(args.unit)
  K = cv2.FileStorage(config_file_I, cv2.FILE_STORAGE_READ)
  cam_list = ["K{}".format(i) for i in range(1, 10)]
  cam_list.remove("K4")
  cam_dict = {cam:{} for cam in cam_list}
  B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

  w = Einv.getNode("w").real()
  h = Einv.getNode("h").real()

  if args.unit == 'cm':
    scaling = 1e-2
  elif args.unit == 'm':
    scaling = 1

  # Load an extrinsic
  for cam in cam_list:
    rotation_inv = Einv.getNode(cam).mat()[:-1, :-1]
    translation = Einv.getNode(cam).mat()[:-1, -1].reshape(-1, 1) / scaling

    cam_dict[cam]['Einv_gl'] = np.concatenate((np.concatenate((rotation_inv, translation), axis=1), np.array([[0, 0, 0, 1]])))
    cam_dict[cam]['E_gl'] = np.linalg.inv(cam_dict[cam]['Einv_gl'].copy())

    # ========== Change convention (Opengl -> Unity) ==========
    cam_dict[cam]['Einv_unity'] = B @ cam_dict[cam]['Einv_gl'] @ B
    cam_dict[cam]['E_unity'] = np.linalg.inv(cam_dict[cam]['Einv_unity'].copy())

    # Load an intrinsic and convert to projectionMatrix
    cam_dict[cam]["K_cv"] = K.getNode("K").mat()

    # Set the projectionMatrix to be the same as unity
    far = 1000
    near = 0.1
    fx = cam_dict[cam]['K_cv'][0, 0]
    fy = cam_dict[cam]['K_cv'][1, 1]
    cx = cam_dict[cam]['K_cv'][0, 2]
    cy = cam_dict[cam]['K_cv'][1, 2]

    # Convert the projectionMatrix of real world to unity
    cam_dict[cam]["K_unity"] = np.array([[fx/cx, 0, 0, 0],
                              [0, fy/cy, 0, 0],
                              [0, 0, -(far+near)/(far-near), -2*(far*near)/(far-near)],
                              [0, 0, -1, 0]])
    cam_dict[cam]['w'] = w
    cam_dict[cam]['h'] = h

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

  return cam_dict

def convert_to_unity_json(cam_dict, out_path):
  print("[#] Convert to Unity (Json format)")
  calibs_json = {c:{} for c in cam_dict.keys()}
  for cam in cam_dict.keys():
    calibs_json[cam] = {"width":cam_dict[cam]['w'],
                        "height":cam_dict[cam]['h'],
                        "E":cam_dict[cam]['E_gl'].flatten().tolist(),
                        "Einv":cam_dict[cam]['Einv_gl'].flatten().tolist(),
                        "K":cam_dict[cam]['K_unity'].flatten().tolist()}
  output_path = out_path + "/motioncapture_unity.json"
  with open(output_path, 'w') as outfile:
    json.dump(calibs_json, outfile)
    print("[#] Write the calibration file to : ", out_path)


if __name__ == "__main__":
  # List trial in directory
  cam_dict = load_config_file(calib_path=args.calib_path)
  convert_to_unity_json(cam_dict=cam_dict, out_path=args.out_path)