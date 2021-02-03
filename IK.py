import numpy as np


def IK_leg(T_base, T_global, links, IK_disp):
    T_global[0] += IK_disp[0]
    T_global[1] += IK_disp[1]
    R_base = T_base[0:3, 0:3]
    T_base = T_base[0:3, 3]
    T_local = np.transpose(R_base).dot(T_global - T_base)

    x, y, z = T_local

    Cq2 = (x ** 2 + y ** 2 - links[0] ** 2 - links[1] ** 2) / (2 * links[0] * links[1])
    Sq2 = np.sqrt(1 - Cq2 ** 2)
    q2 = np.arctan2(Sq2, Cq2)
    q1 = np.arctan2(y, x) - np.arctan2(links[1] * np.sin(q2), links[0] + links[1] * np.cos(q2))
    q3 = -q1 - q2
    return np.array([q1, q2, q3])


def IK_tripteron(T_base, T_global, links, IK_disp):
    """
    :return: list of legs' states
    """

    q = []
    for leg in range(len(T_base)):
        q_leg = IK_leg(T_base[leg], T_global, links, IK_disp[leg])
        q.append(q_leg)
    return q
