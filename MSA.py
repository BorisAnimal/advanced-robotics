import numpy as np
import matplotlib.pyplot as plt
from matrices import *
from IK import IK_tripteron


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


def transform_stiffness(T_base, T_global, q_passive, links):
    Q = []
    for i in range(len(T_base)):
        q = q_passive[i]
        toOrigin = T_base[i]

        toLink1 = np.linalg.multi_dot([toOrigin,  # T_base transform
                                       Tz(T_global[i]),  # active joint
                                       Rz(q[0])])  # passive joint
        rotationLink1 = toLink1[0:3, 0:3]

        toLink2 = np.linalg.multi_dot([toLink1,  # transform to the passive joint
                                       Tx(links[0]),  # rigid link
                                       Rz(q[1])])  # passive joint
        rotationLink2 = toLink2[0:3, 0:3]

        zeros = np.zeros((3, 3))

        Q1 = np.vstack([np.hstack([rotationLink1, zeros]),
                        np.hstack([zeros, rotationLink1])])

        Q2 = np.vstack([np.hstack([rotationLink2, zeros]),
                        np.hstack([zeros, rotationLink2])])

        Q.append([Q1, Q2])
    return Q


########################## Params ##########################
space_x = space_y = space_z = 1.0  # workspace size
L = 1.0  # condition
links = np.array([L, L])  # links lengths
l = 0.1  # condition (platform link 8-e)
d = 0.2  # assumption (diameter)

ang60 = np.pi / 3  # 60 deg
IK_disp = [[l * np.cos(ang60), l * np.sin(ang60)], [l * np.cos(ang60), l * np.sin(-ang60)], [-l, 0.0]]

K_active = 1e6  # assumption (from paper)
E = 69 * 1e9  # Young's modulus https://en.wikipedia.org/wiki/Young%27s_modulus
G = 25.5 * 1e9  # shear modulus

S = np.pi * (d ** 2) / 4
Iy = np.pi * (d ** 4) / 64
Iz = np.pi * (d ** 4) / 64
J = Iy + Iz

F = np.array([0, 0, 100, 0, 0, 0]).reshape((-1, 1))

K11 = np.array([
    [E * S / L, 0, 0, 0, 0, 0],
    [0, 12 * E * Iz / L ** 3, 0, 0, 0, 6 * E * Iz / L ** 2],
    [0, 0, 12 * E * Iy / L ** 3, 0, -6 * E * Iy / L ** 2, 0],
    [0, 0, 0, G * J / L, 0, 0],
    [0, 0, -6 * E * Iy / L ** 2, 0, 4 * E * Iy / L, 0],
    [0, 6 * E * Iz / L ** 2, 0, 0, 0, 4 * E * Iz / L]
])

K12 = np.array([
    [-E * S / L, 0, 0, 0, 0, 0],
    [0, -12 * E * Iz / L ** 3, 0, 0, 0, 6 * E * Iz / L ** 2],
    [0, 0, -12 * E * Iy / L ** 3, 0, -6 * E * Iy / L ** 2, 0],
    [0, 0, 0, -G * J / L, 0, 0],
    [0, 0, 6 * E * Iy / L ** 2, 0, 2 * E * Iy / L, 0],
    [0, -6 * E * Iz / L ** 2, 0, 0, 0, 2 * E * Iz / L]
])
K21 = K12.T
K22 = np.array([
    [E * S / L, 0, 0, 0, 0, 0],
    [0, 12 * E * Iz / L ** 3, 0, 0, 0, -6 * E * Iz / L ** 2],
    [0, 0, 12 * E * Iy / L ** 3, 0, 6 * E * Iy / L ** 2, 0],
    [0, 0, 0, G * J / L, 0, 0],
    [0, 0, 6 * E * Iy / L ** 2, 0, 4 * E * Iy / L, 0],
    [0, -6 * E * Iz / L ** 2, 0, 0, 0, 4 * E * Iz / L]
])

T_base_z = np.eye(4)  # Also global origin
T_base_y = np.linalg.multi_dot([Tz(space_z), Rx(-np.pi / 2)])
T_base_x = np.linalg.multi_dot([Ty(space_y), Ry(np.pi / 2), Rz(np.pi)])
T_base = [T_base_x, T_base_y, T_base_z]

T_tool_z = np.eye(4)
T_tool_y = np.transpose(Rx(-np.pi / 2))
T_tool_x = np.transpose(np.linalg.multi_dot([Ry(np.pi / 2), Rz(np.pi)]))
T_tool = [T_tool_x, T_tool_y, T_tool_z]


def Kc_tripteron_MSA(Q, IK_disp, K11, K12, K21, K22, lambda_e_12, lambda_r_12, lambda_r_34, lambda_r_56, lambda_r_78,
                     lambda_p_34,
                     lambda_p_56, lambda_p_78):
    Kc = []
    # Assembling 108x108 matrix; 108 = 6*18
    for i in range(len(Q)):
        equations = []
        # Equation 16 (base)
        equations.append([np.zeros((6, 6 * 9)),
                          np.eye(6),
                          np.zeros((6, 6 * 8))])

        # Equation 17 (4-5)
        Q1 = Q[i][0]
        K1_11 = np.linalg.multi_dot([Q1, K11, np.transpose(Q1)])
        K1_12 = np.linalg.multi_dot([Q1, K12, np.transpose(Q1)])
        K1_21 = np.linalg.multi_dot([Q1, K21, np.transpose(Q1)])
        K1_22 = np.linalg.multi_dot([Q1, K22, np.transpose(Q1)])

        equations.append([np.zeros((6, 6 * 3)),
                          -np.eye(6),
                          np.zeros((6, 6 * 8)),
                          K1_11,
                          K1_12,
                          np.zeros((6, 6 * 4))])
        equations.append([np.zeros((6, 6 * 4)),
                          -np.eye(6),
                          np.zeros((6, 6 * 7)),
                          K1_21,
                          K1_22,
                          np.zeros((6, 6 * 4))])

        # Equation 17 (6-7)
        Q2 = Q[i][1]
        K2_11 = np.linalg.multi_dot([Q2, K11, np.transpose(Q2)])
        K2_12 = np.linalg.multi_dot([Q2, K12, np.transpose(Q2)])
        K2_21 = np.linalg.multi_dot([Q2, K21, np.transpose(Q2)])
        K2_22 = np.linalg.multi_dot([Q2, K22, np.transpose(Q2)])

        equations.append([np.zeros((6, 6 * 5)),
                          -np.eye(6),
                          np.zeros((6, 6 * 8)),
                          K2_11,
                          K2_12,
                          np.zeros((6, 6 * 2))])
        equations.append([np.zeros((6, 6 * 6)),
                          -np.eye(6),
                          np.zeros((6, 6 * 7)),
                          K2_21,
                          K2_22,
                          np.zeros((6, 6 * 2))])

        # Equation 18 (8-e) 
        # platform transform:
        D = np.eye(6)
        skew = np.array([
            [0, 0, IK_disp[i][1]],
            [0, 0, -IK_disp[i][0]],
            [-IK_disp[i][1], IK_disp[i][0], 0]
        ])
        D[:3, 3:6] = skew
        equations.append([np.zeros((6, 6 * 16)),
                          D,
                          -np.eye(6)])
        equations.append([np.zeros((6, 6 * 7)),
                          np.eye(6),
                          D,
                          np.zeros((6, 6 * 9))])

        # Equation 22 (2-3)
        equations.append([np.zeros((6, 6 * 10)),
                          np.eye(6),
                          -np.eye(6),
                          np.zeros((6, 6 * 6))])
        equations.append([np.zeros((6, 6 * 1)),
                          np.eye(6),
                          np.eye(6),
                          np.zeros((6, 6 * 15))])

        # Equation 23 (1-2) - active elastic joint
        equations.append([np.zeros((5, 6 * 9)),
                          lambda_r_12[i],
                          -lambda_r_12[i],
                          np.zeros((5, 6 * 7))])
        equations.append([np.eye(6),
                          np.eye(6),
                          np.zeros((6, 6 * 16))])
        equations.append([lambda_e_12[i],
                          np.zeros((6 * 8)),
                          K_active * lambda_e_12[i],
                          -K_active * lambda_e_12[i],
                          np.zeros((6 * 7))])

        # Equation 30 (3-4)
        equations.append([np.zeros((5, 6 * 11)),
                          lambda_r_34[i],
                          -lambda_r_34[i],
                          np.zeros((5, 6 * 5))])
        equations.append([np.zeros((5, 6 * 2)),
                          lambda_r_34[i],
                          lambda_r_34[i],
                          np.zeros((5, 6 * 14))])
        equations.append([np.zeros((6 * 2)),
                          lambda_p_34[i],
                          np.zeros((6 * 15))])
        equations.append([np.zeros((6 * 3)),
                          lambda_p_34[i],
                          np.zeros((6 * 14))])

        # Equation 30 (5-6)
        equations.append([np.zeros((5, 6 * 13)),
                          lambda_r_56[i],
                          -lambda_r_56[i],
                          np.zeros((5, 6 * 3))])
        equations.append([np.zeros((5, 6 * 4)),
                          lambda_r_56[i],
                          lambda_r_56[i],
                          np.zeros((5, 6 * 12))])
        equations.append([np.zeros((6 * 4)),
                          lambda_p_56[i],
                          np.zeros((6 * 13))])
        equations.append([np.zeros((6 * 5)),
                          lambda_p_56[i],
                          np.zeros((6 * 12))])

        # Equation 30 (7-8)
        equations.append([np.zeros((5, 6 * 15)),
                          lambda_r_78[i],
                          -lambda_r_78[i],
                          np.zeros((5, 6 * 1))])
        equations.append([np.zeros((5, 6 * 6)),
                          lambda_r_78[i],
                          lambda_r_78[i],
                          np.zeros((5, 6 * 10))])
        equations.append([np.zeros((6 * 6)),
                          lambda_p_78[i],
                          np.zeros((6 * 11))])
        equations.append([np.zeros((6 * 7)),
                          lambda_p_78[i],
                          np.zeros((6 * 10))])

        # Equation 37 External load
        equations.append([np.zeros((6, 6 * 8)),
                          -np.eye(6),
                          np.zeros((6, 6 * 9))])

        # Aggregated matrix
        equations = [np.hstack(x) for x in equations]
        agg = np.vstack(equations)

        A = agg[0:102, 0:102]
        B = agg[0:102, 102:108]
        C = agg[102:108, 0:102]
        D = agg[102:108, 102:108]

        K_leg = D - np.linalg.multi_dot([C, np.linalg.inv(A), B])
        Kc.append(K_leg)
    Kc = Kc[0] + Kc[1] + Kc[2]
    return Kc


xScatter = np.array([])
yScatter = np.array([])
zScatter = np.array([])
dScatter = np.array([])

start = 0.01
step = 0.1
step_z = 0.1
for z in np.arange(start, space_z + start, step_z):
    xData = np.array([])
    yData = np.array([])
    zData = np.array([])
    dData = np.array([])
    for x in np.arange(start, space_x + start, step):
        for y in np.arange(start, space_y + start, step):
            T_global = np.array([x, y, z])

            q_passive = IK_tripteron(T_base, T_global, links, IK_disp)

            Q = transform_stiffness(T_base, T_global, q_passive, links)

            Kc = Kc_tripteron_MSA(Q, IK_disp, K11, K12, K21, K22, lambda_e_12,
                                  lambda_r_12, lambda_r_34, lambda_r_56, lambda_r_78,
                                  lambda_p_34, lambda_p_56, lambda_p_78)

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
