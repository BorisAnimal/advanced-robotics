import numpy as np
import matplotlib.pyplot as plt
from matrices import *
from jacobians import jacobian_passive_tripteron, jacobian_theta_tripteron
from IK import IK_tripteron
from common_params import *


def K_theta_leg(K_active, E, G, L):
    """
    :param L: if L is list (links lengths are different) -> Change code :)
    """
    zeros_6_1 = np.zeros((6, 1))
    zeros_6_6 = np.zeros((6, 6))
    K0 = np.zeros(13)
    K0[0] = K_active
    K1_22 = K2_22 = np.array([
        [E * S / L, 0, 0, 0, 0, 0],
        [0, 12 * E * Iz / L ** 3, 0, 0, 0, -6 * E * Iz / L ** 2],
        [0, 0, 12 * E * Iy / L ** 3, 0, 6 * E * Iy / L ** 2, 0],
        [0, 0, 0, G * J / L, 0, 0],
        [0, 0, 6 * E * Iy / L ** 2, 0, 4 * E * Iy / L, 0],
        [0, -6 * E * Iz / L ** 2, 0, 0, 0, 4 * E * Iz / L]
    ])
    K1 = np.hstack([zeros_6_1, K1_22, zeros_6_6])
    K2 = np.hstack([zeros_6_1, zeros_6_6, K2_22])
    K = np.vstack([K0, K1, K2])
    return K


def Kc_tripteron_VJM(K_theta, J_q, J_theta):
    Kc_total = []
    for i in range(len(K_theta)):
        Kc0 = np.linalg.inv(np.linalg.multi_dot([J_theta[i], np.linalg.inv(K_theta[i]), np.transpose(J_theta[i])]))
        Kc = Kc0 - np.linalg.multi_dot(
            [Kc0, J_q[i], np.linalg.inv(np.linalg.multi_dot([np.transpose(J_q[i]), Kc0, J_q[i]])), np.transpose(J_q[i]),
             np.linalg.inv(Kc0)
             ])
        Kc_total.append(Kc)

    Kc_total = Kc_total[0] + Kc_total[1] + Kc_total[2]
    return Kc_total


def plot_deflection(x, y, z, deflection):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(30, 35)

    cmap = ax.scatter3D(x, y, z, c=deflection, marker='^', s=30)
    plt.colorbar(cmap)
    plt.show()


Ktheta = K_theta_leg(K_active, E, G, L)
Ktheta = [Ktheta, Ktheta, Ktheta]

xScatter = np.array([])
yScatter = np.array([])
zScatter = np.array([])
dScatter = np.array([])

for z in np.arange(start, space_z + start, step_z):
    xData = np.array([])
    yData = np.array([])
    zData = np.array([])
    dData = np.array([])
    for x in np.arange(start, space_x + start, step):
        for y in np.arange(start, space_y + start, step):
            T_global = np.array([x, y, z])
            q_active = [[T_global[0]], [T_global[1]], [T_global[2]]]
            q_passive = IK_tripteron(T_base, T_global, links, IK_disp)

            Jq = jacobian_passive_tripteron(T_base, T_tool, q_active, q_passive, theta, links, IK_disp)
            Jtheta = jacobian_theta_tripteron(T_base, T_tool, q_active, q_passive, theta, links, IK_disp)

            Kc = Kc_tripteron_VJM(Ktheta, Jq, Jtheta)

            dt = np.linalg.inv(Kc).dot(F)
            deflection = np.sqrt(dt[0] ** 2 + dt[1] ** 2 + dt[2] ** 2)

            xData = np.append(xData, x)
            yData = np.append(yData, y)
            zData = np.append(zData, z)
            dData = np.append(dData, deflection)

    xScatter = np.append(xScatter, xData)
    yScatter = np.append(yScatter, yData)
    zScatter = np.append(zScatter, zData)
    dScatter = np.append(dScatter, dData)

    N = int(np.sqrt(dData.shape[0]))
    dData = dData.reshape(N, N)

plot_deflection(xScatter, yScatter, zScatter, dScatter)
