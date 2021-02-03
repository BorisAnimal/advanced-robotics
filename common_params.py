import numpy as np
from matrices import Rx, Ry, Rz, Tx, Ty, Tz

########################## Params ##########################
space_x = space_y = space_z = 1.0  # workspace size
L = 1.0  # condition
links = np.array([L, L])  # links lengths
l = 0.1  # condition (platform link 8-e)
d = 0.2  # assumption (diameter)

# F = np.array([0, 0, 100, 0, 0, 0]).reshape((-1, 1))
# F = np.array([0, 100, 0, 0, 0, 0]).reshape((-1, 1))
F = np.array([100, 0, 0, 0, 0, 0]).reshape((-1, 1))

ang60 = np.pi / 3  # 60 deg
# IK_disp = [[l * np.cos(ang60), l * np.sin(ang60)], [l * np.cos(ang60), l * np.sin(-ang60)], [-l, 0.0]]
IK_disp = [[l * np.sin(ang60), l * np.cos(ang60)], [-l * np.sin(ang60), l * np.cos(ang60)], [0.0, -l]]

K_active = 1e6  # assumption (from paper)
E = 69 * 1e9  # Young's modulus https://en.wikipedia.org/wiki/Young%27s_modulus
G = 25.5 * 1e9  # shear modulus

S = np.pi * (d ** 2) / 4
Iy = np.pi * (d ** 4) / 64
Iz = np.pi * (d ** 4) / 64
J = Iy + Iz

theta = np.zeros(13)  # because we don't know initial theta
theta = [theta, theta, theta]

T_base_z = np.eye(4)  # Also global origin
T_base_y = np.linalg.multi_dot([Tz(space_z), Rx(-np.pi / 2)])
T_base_x = np.linalg.multi_dot([Ty(space_y), Ry(np.pi / 2), Rz(np.pi)])
T_base = [T_base_x, T_base_y, T_base_z]

T_tool_z = np.eye(4)
T_tool_y = np.transpose(Rx(-np.pi / 2))
T_tool_x = np.transpose(np.linalg.multi_dot([Ry(np.pi / 2), Rz(np.pi)]))
T_tool = [T_tool_x, T_tool_y, T_tool_z]

start = 0.01
step = 0.1
step_z = 0.1
