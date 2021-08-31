import numpy as np
from scipy.spatial.transform import Rotation as R

theta_x = 90
theta_y = 0
theta_z = 0

Rx = np.array([[1, 0, 0],
               [0, np.cos(theta_x), -np.sin(theta_x)],
               [0, np.sin(theta_x), np.cos(theta_x)]])

Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
               [0, 1, 0],
               [-np.sin(theta_y), 0, np.cos(theta_y)]])

Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
               [np.sin(theta_z), np.cos(theta_z), 0],
               [0, 0, 1]])


print("Unity order (ZXY) : ", Rz @ Rx @ Ry)

r = R.from_euler('zxy', [theta_x, theta_y, theta_z], degrees=True)
print(r.as_matrix())

