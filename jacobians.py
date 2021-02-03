import numpy as np
from matrices import *
from FK import FK_tripteron


def jacobian_passive_leg(T_fk, T_base, T_tool, q_active, q_passive, theta, link, platform_link):
    T_fk[0:3, 3] = 0
    inv_T_fk = np.transpose(T_fk)

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        dRz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J1 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        dRz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J2 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        dRz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J3 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    J = np.hstack([J1, J2, J3])
    return J


def jacobian_passive_tripteron(T_base, T_tool, q_active, q_passive, theta, link, platform_link):
    T_fk = FK_tripteron(T_base, T_tool, q_active, q_passive, theta, link, platform_link)

    Jq = []
    for leg in range(len(T_base)):
        J = jacobian_passive_leg(T_fk[leg], T_base[leg], T_tool[leg], q_active[leg], q_passive[leg], theta[leg], link,
                                 platform_link)
        Jq.append(J)
    return Jq


def jacobian_theta_leg(T_fk, T_base, T_tool, q_active, q_passive, theta, link, platform_link):
    T_fk[0:3, 3] = 0
    inv_T_fk = np.transpose(T_fk)

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        dTz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J1 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        dTx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J2 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), dTy(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J3 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), dTz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J4 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), dRx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J5 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), dRy(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J6 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        dRz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J7 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        dTx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J8 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), dTy(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J9 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), dTz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J10 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), dRx(theta[10]), Ry(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J11 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), dRy(theta[11]),
                                        Rz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J12 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    dT_leg_local = np.linalg.multi_dot([Tz(q_active[0]),  # active joint
                                        Tz(theta[0]),  # 1 DOF virtual spring
                                        Rz(q_passive[0]),  # passive joint
                                        Tx(link[0]),  # rigid link
                                        Tx(theta[1]), Ty(theta[2]), Tz(theta[3]), Rx(theta[4]), Ry(theta[5]),
                                        Rz(theta[6]),  # 6 DOF virtual spring
                                        Rz(q_passive[1]),  # passive joint
                                        Tx(link[1]),  # rigid link
                                        Tx(theta[7]), Ty(theta[8]), Tz(theta[9]), Rx(theta[10]), Ry(theta[11]),
                                        dRz(theta[12]),  # 6 DOF virtual spring
                                        Rz(q_passive[2]),  # passive joint
                                        Tx(platform_link)  # platform link
                                        ])

    dT_leg = np.linalg.multi_dot([T_base, dT_leg_local, T_tool, inv_T_fk])
    J13 = np.vstack([dT_leg[0, 3], dT_leg[1, 3], dT_leg[2, 3], dT_leg[2, 1], dT_leg[0, 2], dT_leg[1, 0]])

    J = np.hstack([J1, J2, J3, J4, J5, J6, J7, J8, J9, J10, J11, J12, J13])
    return J


def jacobian_theta_tripteron(T_base, T_tool, q_active, q_passive, theta, link, platform_link):
    T_fk = FK_tripteron(T_base, T_tool, q_active, q_passive, theta, link, platform_link)

    Jtheta = []
    for leg in range(len(T_base)):
        J = jacobian_theta_leg(T_fk[leg], T_base[leg], T_tool[leg], q_active[leg], q_passive[leg], theta[leg], link,
                               platform_link)
        Jtheta.append(J)
    return Jtheta
