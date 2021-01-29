from scipy.spatial.transform import Rotation as Rot
import numpy as np
from numpy import sin as S
from numpy import cos as C
from math import pi


def expand_rotation(rot):
    tmp = np.eye(4, 4)
    tmp[:3, :3] = rot
    return tmp


def Rx(q):
    """
    :param q: in radians!!
    """
    return expand_rotation(Rot.from_euler('x', q).as_dcm())


def Ry(q):
    """
    :param q: in radians!!
    """
    return expand_rotation(Rot.from_euler('y', q).as_dcm())


def Rz(q):
    """
    :param q: in radians!!
    """
    return expand_rotation(Rot.from_euler('z', q).as_dcm())


def Tx(mov):
    tmp = np.eye(4, 4)
    tmp[0, 3] = mov
    return tmp


def Ty(mov):
    tmp = np.eye(4, 4)
    tmp[1, 3] = mov
    return tmp


def Tz(mov):
    tmp = np.eye(4, 4)
    tmp[2, 3] = mov
    return tmp


def Tr(xyz, rpy):
    [r, p, y] = rpy
    R = np.array([[C(p) * C(y), C(y) * S(p) * S(r) - S(y) * C(r), C(y) * S(p) * C(r) + S(y) * S(r)],
                  [C(p) * S(y), S(y) * S(p) * S(r) + C(y) * C(r), S(y) * S(p) * C(r) - C(y) * S(r)],
                  [-S(p), C(p) * S(r), C(p) * C(r)]
                  ])
    res = np.eye(4, 4)
    res[:3, :3] = R
    [x, y, z] = xyz
    res[0, 3] = x
    res[1, 3] = y
    res[2, 3] = z
    return res


def dRx(q):
    R = Rot.from_euler('x', q + pi / 2).as_dcm()
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = R
    tmp[0, 0] = 0.0
    return tmp


def dRy(q):
    R = Rot.from_euler('y', q + pi / 2).as_dcm()
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = R
    tmp[1, 1] = 0.0
    return tmp


def dRz(q):
    R = Rot.from_euler('z', q + pi / 2).as_dcm()
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = R
    tmp[2, 2] = 0.0
    return tmp


def dTx(mov):
    tmp = np.zeros((4, 4))
    tmp[0, 3] = 1
    return tmp


def dTy(mov):
    tmp = np.zeros((4, 4))
    tmp[1, 3] = 1
    return tmp


def dTz(mov):
    tmp = np.zeros((4, 4))
    tmp[2, 3] = 1
    return tmp


########################## Lambdas ##########################
lambda_r_12_x = np.array([[0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_e_12_x = np.array([1, 0, 0, 0, 0, 0], dtype=float)

lambda_r_12_y = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_e_12_y = np.array([0, 1, 0, 0, 0, 0], dtype=float)

lambda_r_12_z = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_e_12_z = np.array([0, 0, 1, 0, 0, 0], dtype=float)

lambda_r_12 = [lambda_r_12_x, lambda_r_12_y, lambda_r_12_z]
lambda_e_12 = [lambda_e_12_x, lambda_e_12_y, lambda_e_12_z]

lambda_r_34_x = lambda_r_56_x = lambda_r_78_x = np.array([[1, 0, 0, 0, 0, 0],
                                                          [0, 1, 0, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0],
                                                          [0, 0, 0, 0, 1, 0],
                                                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_p_34_x = lambda_p_56_x = lambda_p_78_x = np.array([0, 0, 0, 1, 0, 0], dtype=float)

lambda_r_34_y = lambda_r_56_y = lambda_r_78_y = np.array([[1, 0, 0, 0, 0, 0],
                                                          [0, 1, 0, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0],
                                                          [0, 0, 0, 1, 0, 0],
                                                          [0, 0, 0, 0, 0, 1]], dtype=float)

lambda_p_34_y = lambda_p_56_y = lambda_p_78_y = np.array([0, 0, 0, 0, 1, 0], dtype=float)

lambda_r_34_z = lambda_r_56_z = lambda_r_78_z = np.array([[1, 0, 0, 0, 0, 0],
                                                          [0, 1, 0, 0, 0, 0],
                                                          [0, 0, 1, 0, 0, 0],
                                                          [0, 0, 0, 1, 0, 0],
                                                          [0, 0, 0, 0, 1, 0]], dtype=float)

lambda_p_34_z = lambda_p_56_z = lambda_p_78_z = np.array([0, 0, 0, 0, 0, 1], dtype=float)

lambda_r_34 = [lambda_r_34_x, lambda_r_34_y, lambda_r_34_z]
lambda_r_56 = [lambda_r_56_x, lambda_r_56_y, lambda_r_56_z]
lambda_r_78 = [lambda_r_78_x, lambda_r_78_y, lambda_r_78_z]

lambda_p_34 = [lambda_p_34_x, lambda_p_34_y, lambda_p_34_z]
lambda_p_56 = [lambda_p_56_x, lambda_p_56_y, lambda_p_56_z]
lambda_p_78 = [lambda_p_78_x, lambda_p_78_y, lambda_p_78_z]
