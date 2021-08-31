import numpy as np
theta = int(input("THETA : "))

rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])

ry = np.array([[np.cos(theta), 0, np.sin(theta)],
               [0, 1, 0],
               [-np.sin(theta), 0, np.cos(theta)]])

rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])

print("[#]RIGHT HANDED COORDINATE SYSTEM")
print(rx)
print(ry)
print(rz)

R = rx @ ry @ rz
print("R GT")
print(R)

theta = -theta

rx_lh = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])

ry_lh = np.array([[np.cos(theta), 0, np.sin(theta)],
               [0, 1, 0],
               [-np.sin(theta), 0, np.cos(theta)]])

rz_lh = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])

print("\n\n\n[#]LEFT HANDED COORDINATE SYSTEM - TRANSPOSE")
print(rx.T)
print(ry.T)
print(rz.T)

print("\n\n\n[#]LEFT HANDED COORDINATE SYSTEM")
print(rx_lh)
print(ry_lh)
print(rz_lh)

R = rx @ ry @ rz
identity = np.identity(3)
identity[1, 1] = -1
print(identity)
print("R")
print(R.T)
R[1, :] *= -1
R[:, 1] *= -1
print("NEGATE")
print(R)
print(rx_lh @ ry_lh @ rz_lh)
