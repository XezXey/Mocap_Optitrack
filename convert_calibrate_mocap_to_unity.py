import numpy as np
import json
import argparse
import glob
import re
import tqdm
import cv2

parser = argparse.ArgumentParser(description='Preprocess and visualize the trajectory in real world')
parser.add_argument('--dataset_path', type=str, help='Specify path to dataset')
parser.add_argument('--process_trial_index', dest='process_trial_index', help='Process trial at given idx only', default=None)
parser.add_argument('--unit', dest='unit', help='Scaling unit', required=True, type=str)
parser.add_argument('--selected_cam', type=str, help="Camera name", default="K2")
args = parser.parse_args()


def load_config_file(folder_name, idx):
  config_file_extrinsic = folder_name + "/motioncapture_camParams_Trial_{}_{}.yml".format(idx, args.unit)
  extrinsic = cv2.FileStorage(config_file_extrinsic, cv2.FILE_STORAGE_READ)
  config_file_intrinsic = folder_name + "/motioncapture_camIntrinsic_{}.yml".format(args.unit)
  intrinsic = cv2.FileStorage(config_file_intrinsic, cv2.FILE_STORAGE_READ)
  nodes_extrinsic = [args.selected_cam, "w", "h"]
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
    elif each_node == args.selected_cam:
      '''
       MOCAP
       - Rotation Matrix: World-space rotation from camera extrinsic
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

def convert_to_unity_json(folder_name, idx, camera_properties_dict):
  print("[#] Convert to Unity (Json format)")
  config_file_extrinsic = folder_name + "/motioncapture_camParams_Trial_{}_{}.json".format(idx, args.unit)
  config_file_intrinsic = folder_name + "/motioncapture_camIntrinsic_{}.json".format(args.unit)

  E = camera_properties_dict['Extrinsic']
  I = camera_properties_dict['Intrinsic']
  P = np.vstack((np.hstack((I, np.zeros(shape=(3, 1)))), np.array([0, 0, 0, 1])))
  calibs_json = {"firstCam":{"width":camera_properties_dict['w'],
                             "height":camera_properties_dict['h'],
                             "worldToCameraMatrix":E.flatten().tolist(),
                             "projectionMatrix":P.flatten().tolist()}}
  output_path = folder_name + "/motioncapture_unity_{}.json".format(idx)
  with open(output_path, 'w') as outfile:
    json.dump(calibs_json, outfile)

if __name__ == "__main__":
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

  for i in tqdm.tqdm(range(len(dataset_folder)), desc="Loading dataset"):
    camera_properties_dict = load_config_file(folder_name=dataset_folder[i], idx=trial_index[i])
    convert_to_unity_json(folder_name=dataset_folder[i], idx=trial_index[i], camera_properties_dict=camera_properties_dict)

