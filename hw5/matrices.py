import numpy as np
import math


def Rx(q):
    q = q * np.pi / 180
    T = np.array([[1, 0, 0, 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, np.sin(q), np.cos(q), 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dRx(q):
    q = q * np.pi / 180
    T = np.array([[0, 0, 0, 0],
                  [0, -np.sin(q), -np.cos(q), 0],
                  [0, np.cos(q), -np.sin(q), 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Ry(q):
    q = q * np.pi / 180
    T = np.array([[np.cos(q), 0, np.sin(q), 0],
                  [0, 1, 0, 0],
                  [-np.sin(q), 0, np.cos(q), 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dRy(q):
    q = q * np.pi / 180
    T = np.array([[-np.sin(q), 0, np.cos(q), 0],
                  [0, 0, 0, 0],
                  [-np.cos(q), 0, -np.sin(q), 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Rz(q):
    q = q * np.pi / 180
    T = np.array([[np.cos(q), -np.sin(q), 0, 0],
                  [np.sin(q), np.cos(q), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dRz(q):
    q = q * np.pi / 180
    T = np.array([[-np.sin(q), -np.cos(q), 0, 0],
                  [np.cos(q), -np.sin(q), 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Tx(x):
    T = np.array([[1, 0, 0, x],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTx(x=None):
    T = np.array([[0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Ty(y):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, y],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTy(y=None):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]], dtype=float)
    return T


def Tz(z):
    T = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, z],
                  [0, 0, 0, 1]], dtype=float)
    return T


def dTz(z=None):
    T = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]], dtype=float)
    return T


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x / (np.pi / 180), y / (np.pi / 180), z / (np.pi / 180)])


def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if np.isclose(R[2, 0], -1.0, rtol=1.e-5, atol=1.e-8):
        theta = math.pi / 2.0
        psi = math.atan2(R[0, 1], R[0, 2])
    elif np.isclose(R[2, 0], 1.0, rtol=1.e-5, atol=1.e-8):
        theta = -math.pi / 2.0
        psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
        theta = -math.asin(R[2, 0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
        phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi / (np.pi / 180), theta / (np.pi / 180), phi / (np.pi / 180)


def rot2rpy(R):
    """ROT2RPY - Transform rotation matrix to roll, pitch, yaw.
    Rotations are around fixed-axes. Roll is around x-axis, pitch is around
    y-axis, yaw is around z-axis.
    Rotations' order is: first roll, then pitch, then yaw.

    Usage: rpy = rot2rpy(R)

    Input:
    R - 3-by-3 rotation matrix

    Output:
    rpy - 3-by-1 vector with [roll, pitch, yaw] values
    """

    A = R[1, 2] + R[0, 1]
    B = R[0, 2] - R[1, 1]
    sinpitch = -R[2, 0]
    if sinpitch == 1:
        yawplusroll = 0
    else:
        yawplusroll = np.arctan2(A / (sinpitch - 1), B / (sinpitch - 1))

    A = R[1, 2] - R[0, 1]
    B = R[0, 2] + R[1, 1]
    if sinpitch == -1:
        yawminusroll = 0
    else:
        yawminusroll = np.arctan2(A / (sinpitch + 1), B / (sinpitch + 1))

    # Output variable
    rpy = np.r_[0., 0., 0.]

    r = (yawplusroll - yawminusroll) / 2.0
    p = np.arctan2(-R[2, 0], R[0, 0] * math.cos(rpy[0]) + R[1, 0] * math.sin(rpy[0]))
    y = (yawplusroll + yawminusroll) / 2.0

    return np.array([r / (np.pi / 180), p / (np.pi / 180), y / (np.pi / 180)])



if __name__ == '__main__':
    angs = [2 * np.pi * step / 11 for step in range(1, 11)]
    for x in angs:
        for y in angs:
            for z in angs:
                x += 0.1
                R = (Rx(x) @ Ry(y) @ Rz(z))[:3, :3]
                xx, yy, zz = euler_angles_from_rotation_matrix(R)
                print(x, y, z)
                print(xx, yy, zz, '!!')
                assert np.isclose(x, xx, 1e-2), (x, xx)
                assert np.isclose(y, yy, 1e-2), (y, yy)
                assert np.isclose(z, zz, 1e-2), (z, zz)
