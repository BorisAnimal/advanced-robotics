import numpy as np
from matrices import *


def FK_leg(T_base, T_tool, q_active, q_passive, theta, link, IK_disp):
    T_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                       Tz(theta[0]),  # 1 DOF virtual spring
                                       Rz(q_passive[0]),  # passive joint
                                       Tx(link[0]),  # rigid link
                                       Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                       Rz(theta[6]),  # 6 DOF virtual spring
                                       Rz(q_passive[1]),  # passive joint
                                       Tx(link[1]),  # rigid link
                                       Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                       Rz(theta[12]),  # 6 DOF virtual spring
                                       Rz(q_passive[2]),  # passive joint
                                       ])

    T_leg = np.linalg.multi_dot([T_base, T_leg_local, T_tool,
                                 Tx(-IK_disp[0]), Ty(-IK_disp[1])
                                 ])
    return T_leg


def FK_tripteron(T_base, T_tool, q_active, q_passive, theta, link, IK_disp):
    T = []
    for leg in range(len(T_base)):
        T_leg = FK_leg(T_base[leg], T_tool[leg], q_active[leg], q_passive[leg], theta[leg], link, IK_disp[leg])
        T.append(T_leg)
    return T
