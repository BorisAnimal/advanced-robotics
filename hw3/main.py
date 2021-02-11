import numpy as np
import matplotlib.pyplot as plt
from matrices import *


def fk(q, theta, links):
    """
    :return: 4x4 FK matrix
    """
    #  Tz(l1)Rz(q1)Tz(q2)Ty(q3)Ty(l3)
    T = np.linalg.multi_dot([Tz(links[0]),  # Tz(l1)
                             Rz(q[0]),  # Rz(q1)
                             Rz(theta[0]),  # 1 DOF virtual spring
                             Tz(q[1]),  # Tz(q2)
                             Tz(theta[1]),  # 1 DOF virtual spring
                             Ty(q[2]),  # Ty(q3)
                             Ty(theta[2]),  # 1 DOF virtual spring
                             Ty(links[2]),  # rigid link
                             ])
    return T


def ik(xyz, links):
    """
    :return: matrix of shape (3,)
    """
    x, y, z = xyz[:3]

    q1 = np.arctan2(y, x) - np.pi / 2
    q2 = z - links[0]
    q3 = np.sqrt(x ** 2 + y ** 2) - links[2]

    return np.array([q1, q2, q3])


def jacobian_theta(q, theta, links):
    """
    :return: matrix of shape (6,3)
    """
    T = fk(q, theta, links)
    T[0:3, 3] = 0
    inv_T = np.transpose(T)

    dT = np.linalg.multi_dot([Tz(links[0]),  # Tz(l1)
                              Rz(q[0]),  # Rz(q1)
                              dRz(theta[0]),  # 1 DOF virtual spring
                              Tz(q[1]),  # Tz(q2)
                              Tz(theta[1]),  # 1 DOF virtual spring
                              Ty(q[2]),  # Ty(q3)
                              Ty(theta[2]),  # 1 DOF virtual spring
                              Ty(links[2]),  # rigid link
                              ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J1 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),  # Tz(l1)
                              Rz(q[0]),  # Rz(q1)
                              Rz(theta[0]),  # 1 DOF virtual spring
                              Tz(q[1]),  # Tz(q2)
                              dTz(theta[1]),  # 1 DOF virtual spring
                              Ty(q[2]),  # Ty(q3)
                              Ty(theta[2]),  # 1 DOF virtual spring
                              Ty(links[2]),  # rigid link
                              ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J2 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([Tz(links[0]),  # Tz(l1)
                              Rz(q[0]),  # Rz(q1)
                              Rz(theta[0]),  # 1 DOF virtual spring
                              Tz(q[1]),  # Tz(q2)
                              Tz(theta[1]),  # 1 DOF virtual spring
                              Ty(q[2]),  # Ty(q3)
                              dTy(theta[2]),  # 1 DOF virtual spring
                              Ty(links[2]),  # rigid link
                              ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J3 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    return np.hstack([J1, J2, J3])


##################### PARAMS #####################
np.random.seed(42)
use_noise = False
links = np.array([0.4, 0.0, 0.1])
theta = np.array([0.0, 0.0, 0.0])
# theta = np.array([0.01, -0.001, 0.06])  # Geometric errors

# This K is constant for SIMULATION model
Kc = np.array([1e6, 2e6, 0.5e6])
Kc = np.diag(Kc)

print(Kc)

##################### TESTS #####################
q = np.array([1.0, 2.0, 3.0])
xyz = fk(q, [0, 0, 0], links)[:3, 3]
q_ik = ik(xyz, links)
assert all(np.isclose(q, q_ik)), (q, '!=', q_ik)

experiments = 30
if __name__ == '__main__':
    dX_1 = np.zeros((3, 3), dtype=float)
    dX_2 = np.zeros(3, dtype=float)
    for i in range(experiments):
        # Random robot's states
        q_revolute = np.random.uniform(-np.pi, np.pi, 1)
        q_prismatic = np.random.uniform(0, 1, 2)
        q = np.hstack([q_revolute, q_prismatic])
        W = np.random.uniform(-1000, 1000, 6)

        J_theta = jacobian_theta(q, theta, links)
        dt = np.linalg.multi_dot([J_theta, np.linalg.inv(Kc), np.transpose(J_theta), W])
        if use_noise:
            dt += np.random.normal(loc=0.0, scale=1e-5)  # measurements noise

        J_theta = J_theta[0:3, :]
        dt = dt[0:3]
        W = W[0:3]

        B = np.zeros(J_theta.shape, dtype=float)
        for i in range(J_theta.shape[1]):
            j = J_theta[:, i].reshape(-1, 1)
            B[:, i] = j.dot(j.T).dot(W)

        dX_1 = dX_1 + np.transpose(B).dot(B)
        dX_2 = dX_2 + np.transpose(B).dot(dt)

    dX = np.linalg.inv(dX_1).dot(dX_2)
    # This K is required for "mirror trajectory" and is not calibrated of Kc above
    Kc_mirror_trajectory = np.diag(np.divide(1.0, dX))  # final stiffness matrix
    # Kc_mirror_trajectory = np.linalg.inv(np.diag(dX))  # Sometimes gives too big error in compare with upper code

    print(Kc_mirror_trajectory)

    # Plotting calculations
    W = np.array([-220.0, 100.0, -1234.0, 0.0, 0.0, 0.0])

    r = 0.1
    xc = 1.1
    yc = 0
    zc = 0.5
    points = 50

    angle = np.linspace(0, 2 * np.pi, points)
    X = xc + r * np.cos(angle)
    Y = yc + r * np.sin(angle)
    Z = zc * np.ones(points)
    traj_desired = np.stack([X, Y, Z])

    joint_states = np.zeros((3, points), dtype=float)
    for i in range(points):
        joint_states[:, i] = ik([X[i], Y[i], Z[i]], links)

    traj_mirror = np.zeros(traj_desired.shape, dtype=float)
    for i in range(points):
        J_theta = jacobian_theta(joint_states[:, i], theta, links)
        dt = np.linalg.multi_dot([J_theta, np.linalg.inv(Kc_mirror_trajectory), np.transpose(J_theta), W])
        if use_noise:
            dt += np.random.normal(loc=0.0, scale=1e-5)  # measurements noise

        traj_mirror[:, i] = traj_desired[:, i] + dt[0:3]

    difference = traj_desired - traj_mirror
    traj_updated = traj_desired + difference

    for i in range(points):
        tool = np.array([traj_updated[0, i], traj_updated[1, i], traj_updated[2, i]])
        joint_states[:, i] = ik(tool, links)

    traj_calibrated = np.zeros(traj_desired.shape, dtype=float)
    for i in range(points):
        J_theta = jacobian_theta(joint_states[:, i], theta, links)
        dt = np.linalg.multi_dot([J_theta, np.linalg.inv(Kc), np.transpose(J_theta), W])

        traj_calibrated[:, i] = traj_updated[:, i] + dt[0:3]

    max_error = np.linalg.norm(traj_calibrated - traj_desired, 2, 0).max()
    print("Maximum error amplitude: {:.2f} microns".format(max_error * 1e6))

    traj_uncalibrated = np.zeros(traj_desired.shape, dtype=float)
    for i in range(points):
        J_theta = jacobian_theta(joint_states[:, i], theta, links)
        dt = np.linalg.multi_dot([J_theta, np.linalg.inv(Kc), np.transpose(J_theta), W])
        if use_noise:
            dt += np.random.normal(loc=0.0, scale=1e-5)  # measurements noise

        traj_uncalibrated[:, i] = traj_desired[:, i] + dt[0:3]

    # Plotting results
    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.plot3D(traj_desired[0], traj_desired[1], traj_desired[2],
              c='gray', linewidth=2, label='desired')
    ax.scatter3D(traj_uncalibrated[0], traj_uncalibrated[1], traj_uncalibrated[2],
                 c='red', s=20, label='uncalibrated')
    ax.scatter3D(traj_calibrated[0], traj_calibrated[1], traj_calibrated[2],
                 c='green', s=25, label='calibrated')
    plt.legend()

    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(5))

    plt.savefig('./imgs/result.png', format="png")
    plt.show()
