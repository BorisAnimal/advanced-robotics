import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matrices import *
import math
np.set_printoptions(2)

test_q = np.array([
    [0,-90, 0, 0, 0, 0, -1000],
    # [110.95, -128.78, 104.12, -36.49, 55.24, 16.07, -1471.69],
    [110.95, -128.78, 104.12, -36.49, 55.24, 16.07, -1219.38],
    [76.96, -128.78, 104.12, -36.49, 55.24, 16.07, -1219.38],
    [76.96, -136.17, 104.12, -36.49, 55.24, 16.07, -1219.38],
    [76.96, -136.17, 96.20, -36.49, 55.24, 16.07, -1219.38],


])

test_p = np.array([
    [586.11, 226.56, 2646.1, -33.48, 61.56, 139.53],
    [333.80, 226.56, 2646.1, -33.48, 61.56, 139.53],
    [358.73, -307.99, 2646.1, 0.56, 61.56, 139,53],
    [523.8, -269.61, 2707.2, -3.44, 68.71, 135.12],
    [608.42, -249.93, 2869.72, -12.43, 76.13, 125.71]
])

links = np.array([645, 330, 1150, 115, 1220, 215])

def fk(q, links):
    T = np.linalg.multi_dot([
                            Tx(-q[6]),
                            Tz(links[0]),
                            Rz(q[0]),
                            Tx(links[1]),
                            Ry(q[1]),
                            Tx(links[2]),
                            Ry(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            Rx(q[3]),
                            Ry(q[4]),
                            Rx(q[5]),
                            Tz(-links[5]),
                             ])
    return T

def jacobian(q, links):
    """
    :return: matrix of shape (6,4)
    """
    T = fk(q, links)
    T[0:3, 3] = 0
    inv_T = np.transpose(T)

    dT = np.linalg.multi_dot([
                            Tx(-q[6]),
                            Tz(links[0]),
                            dRz(q[0]),
                            Tx(links[1]),
                            Ry(q[1]),
                            Tx(links[2]),
                            Ry(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            Rx(q[3]),
                            Ry(q[4]),
                            Rx(q[5]),
                            Tz(-links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J1 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
                            Tx(-q[6]),
                            Tz(links[0]),
                            Rz(q[0]),
                            Tx(links[1]),
                            dRy(q[1]),
                            Tx(links[2]),
                            Ry(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            Rx(q[3]),
                            Ry(q[4]),
                            Rx(q[5]),
                            Tz(-links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J2 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
                            Tx(-q[6]),
                            Tz(links[0]),
                            Rz(q[0]),
                            Tx(links[1]),
                            Ry(q[1]),
                            Tx(links[2]),
                            dRy(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            Rx(q[3]),
                            Ry(q[4]),
                            Rx(q[5]),
                            Tz(-links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J3 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
                            Tx(-q[6]),
                            Tz(links[0]),
                            Rz(q[0]),
                            Tx(links[1]),
                            Ry(q[1]),
                            Tx(links[2]),
                            Ry(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            dRx(q[3]),
                            Ry(q[4]),
                            Rx(q[5]),
                            Tz(-links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J4 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
                            Tx(-q[6]),
                            Tz(links[0]),
                            Rz(q[0]),
                            Tx(links[1]),
                            Ry(q[1]),
                            Tx(links[2]),
                            Ry(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            Rx(q[3]),
                            dRy(q[4]),
                            Rx(q[5]),
                            Tz(-links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J5 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
                            Tx(-q[6]),
                            Tz(links[0]),
                            Rz(q[0]),
                            Tx(links[1]),
                            Ry(q[1]),
                            Tx(links[2]),
                            Ry(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            Rx(q[3]),
                            Ry(q[4]),
                            dRx(q[5]),
                            Tz(-links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J6 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
                            dTx(-q[6]),
                            Tz(links[0]),
                            Rz(q[0]),
                            Tx(links[1]),
                            Ry(q[1]),
                            Tx(links[2]),
                            Ry(q[2]),
                            Tx(links[3]),
                            Tz(-links[4]),
                            Rx(q[3]),
                            Ry(q[4]),
                            Rx(q[5]),
                            Tz(-links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J7 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])


    return np.hstack([J1, J2, J3, J4, J5, J6, J7])




from mpl_toolkits.mplot3d import Axes3D
def plot_state(q, links, ax):
    trans = [
        Tx(0),
        'O',
        Tx(-q[6]),
        'O',
        Tz(links[0]),
        'O',
        Rz(q[0]),
        Tx(links[1]),
        'O',
        Ry(q[1]),
        Tx(links[2]),
        'O',
        Ry(q[2]),
        Tx(links[3]),
        'O',
        Tz(-links[4]),
        'O',
        Rx(q[3]),
        'O',
        Ry(q[4]),
        'O',
        Rx(q[5]),
        'O',
        Tz(-links[5]),
        'O',
    ]
    res = trans[0]
    points = [res[0:3, 3].flatten()]
    for t in trans[1:]:
        if t != 'O':
            res = res @ t
            points.append(res[0:3, 3].flatten())
    points = np.array(points)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], s=10)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=0.7)
    # ax.view_init(0, 30)

def get_pos(q, links):
    transl = fk(q, links)
    # phi = np.arctan2(transl[1, 0], transl[0,0])
    xyf_0 = np.vstack([transl[0, 3], transl[1, 3], transl[2, 3], transl[2, 1], transl[0, 2], transl[1, 0]])
    return xyf_0

def get_difference(current_xyf, target_xyf, ext=False):
    dx = target_xyf[0] - current_xyf[0]
    dy = target_xyf[1] - current_xyf[1]
    dz = target_xyf[2] - current_xyf[2]
    # dphi = (target_xyf[2] - current_xyf[2] + 3 * np.pi)%(2*np.pi) - np.pi
    # if ext:
    #     return np.vstack([[dx], [dy], [dphi]])
    # else:
    return np.vstack([[dx], [dy], [dz],[0], [0], [0]])

def update_fraction(d_xyf, max_d = 0.01):
    kx = abs(d_xyf[0]) / min(abs(d_xyf[0]),max_d)
    kx = np.nan_to_num(kx)[0]
    dkx = min(abs(d_xyf[0]),max_d)

    ky = abs(d_xyf[1]) / min(abs(d_xyf[1]),max_d)
    ky = np.nan_to_num(ky)[0]
    dky = min(abs(d_xyf[1]),max_d)

    kz = abs(d_xyf[2]) / min(abs(d_xyf[2]),max_d)
    kz = np.nan_to_num(kz)[0]
    dkz = min(abs(d_xyf[2]),max_d)
    return d_xyf / max(max(kx, ky), kz)

def update_fraction_ang(d_xyf, max_d = 0.01):
    kx = abs(d_xyf[0]) / min(abs(d_xyf[0]),max_d)
    kx = np.nan_to_num(kx)[0]
    dkx = min(abs(d_xyf[0]),max_d)

    ky = abs(d_xyf[1]) / min(abs(d_xyf[1]),max_d)
    ky = np.nan_to_num(ky)[0]
    dky = min(abs(d_xyf[1]),max_d)

    return d_xyf / max(kx, ky)

def update_dq(d_q, d_q_max):
    d_q_new = np.maximum(np.minimum(d_q, d_q_max), -d_q_max)
    dk = d_q / d_q_new
    dk_abs = np.abs(d_q / d_q_new)
    if np.max(dk_abs) > 1.0:
        pp = np.max(dk_abs)
        d_q_new = d_q_new / np.max(dk_abs)
        # print(dk_abs)
    return d_q_new

def pseudoinverse(q, links, xyf_target, W=np.eye(7)):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)
    while np.sqrt(np.sum(d_xyf ** 2)) > 1:
        del_e = update_fraction(d_xyf,10)

        J = jacobian(q, links)[:3,:]
        J_psevd = np.linalg.multi_dot([
            np.linalg.inv(W),
            J.T,
            np.linalg.inv(
                np.linalg.multi_dot([
                    J,
                    np.linalg.inv(W),
                    J.T
                ])
            )
        ])
        d_q = J_psevd.dot(del_e[:3,:])

        # d_q = update_dq(d_q, d_q_max)
        # q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        q = q + d_q
        q_out.append(q.copy())

        d_xyf = get_difference(get_pos(q, links), xyf_target)
        print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, dq: {d_q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
    return q_out


def dls(q, links, xyf_target, mu=0.001):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)

    while np.sqrt(np.sum(d_xyf ** 2)) > 1:
        del_e = update_fraction(d_xyf,20)

        J = jacobian(q, links)[:3,:]

        d_q = np.linalg.multi_dot([
            J.T,
            np.linalg.inv(
                np.linalg.multi_dot([
                    J,
                    J.T
                ])
                +
                mu**2 *
                np.eye(3)
            ),
            del_e[:3,:]
        ])

        # d_q = update_dq(d_q, d_q_max)
        # q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        q = q + d_q
        q_out.append(q.copy())

        xyf_0 = get_pos(q, links)
        d_xyf = get_difference(xyf_0, xyf_target)

        print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, dq: {d_q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
    return q_out

# the manipulability measure
def dH_manipulability(q, links):
    dH = []
    dq = 0.05
    J = jacobian(q, links)
    H = np.sqrt(np.linalg.det(J.dot(J.T)))
    for j in range(q.shape[0]):
        t_q_old = q.copy()
        t_q_old[j] += dq
        J_old = jacobian(t_q_old, links)
        H_old = np.sqrt(np.linalg.det(J_old.dot(J_old.T)))
        dH.append((H - H_old) / (q[j] - t_q_old[j]))
    return np.array(dH).reshape((-1, 1))


def null_space(q, links, xyf_target, k0=0.01):
    # q_min = np.array([[-np.pi*2/4], [-np.pi*2/4], [-np.pi*2/4], [-np.pi*2/4]])
    # q_max = np.array([[np.pi*2/4], [np.pi*2/4], [np.pi*2/4], [np.pi*2/4]])
    d_q_max = np.array([[12], [15], [12], [17], [17], [20], [10]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0, xyf_target)
    q_old = None
    cnt = 0

    while np.sqrt(np.sum(d_xyf**2))> 2:
        del_e = update_fraction(d_xyf,10)


        dH = dH_manipulability(q, links)
        d_q_0 = k0 * np.nan_to_num(dH)
        d_q_0 = update_dq(d_q_0, d_q_max)

        J = jacobian(q, links)
        J_psevd =J.T.dot(np.linalg.inv(J.dot(J.T)))
        d_q = J_psevd.dot(del_e) + (np.eye(7) - J_psevd.dot(J)).dot(d_q_0)

        d_q = update_dq(d_q, d_q_max)
        # q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        q = q + d_q
        q_out.append(q.copy())

        xyf_0 = get_pos(q, links)
        d_xyf = get_difference(xyf_0, xyf_target)

        print(f" Dist: {np.sqrt(np.sum(d_xyf**2))}, q: {q.T}")
        cnt +=1
        # if cnt > 100:
        #     break

    return q_out

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


def task_priority(q, links, xyf_target):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    # d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    d_q_max = np.array([[12], [15], [12], [17], [17], [20], [10]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)
    cnt = 0
    print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
    while np.sqrt(np.sum(d_xyf ** 2)) > 2:

        del_e = update_fraction(get_difference(xyf_0, xyf_target),10)

        J = jacobian(q, links)
        J_1 = J[:3,:]
        J_2 = J[3:5,:].reshape((2,-1))
        J_psevd_1 = np.linalg.multi_dot([
            J_1.T,
            np.linalg.inv(
                J_1.dot(J_1.T)
            )
        ])
        P_1 = np.eye(7) - J_psevd_1.dot(J_1)

        J_psevd_2 = np.linalg.multi_dot([
            J_2.T,
            np.linalg.inv(
                J_2.dot(J_2.T)
            )
        ])

        psi, theta, phi = euler_angles_from_rotation_matrix(fk(q,links))
        d_psi = (0 - psi + 3 * 180) % (2 * 180) - 180
        # d_theta = (-90 - theta + 3* 180)%(2*180) - 180
        d_theta = (0 - theta + 3* 180)%(2*180) - 180
        rot = np.vstack([d_psi, d_theta])
        rot = update_fraction_ang(rot, 5)

        P_2_1 = rot - \
            np.linalg.multi_dot([
            J_2,
            J_psevd_1,
            del_e[:3,:]
            ])

        d_q_0 = J_psevd_1.dot(del_e[:3,:])
        d_q_0 = update_dq(d_q_0, d_q_max)

        d_q_1 = np.linalg.multi_dot([
                    P_1,
                    np.linalg.pinv(J_2.dot(P_1)),
                    P_2_1
               ])
        d_q_1 = update_dq(d_q_1, d_q_max)

        d_q = d_q_0 + 0.04 * d_q_1

        d_q = update_dq(d_q, d_q_max)
        q = q + d_q
        q_out.append(q.copy())

        xyf_0 = get_pos(q, links)
        d_xyf = get_difference(xyf_0, xyf_target)
        cnt +=1
        print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T},d_phi: {rot.T}, dq0: {d_q_0.T }, dq1: {d_q_1.T}")
        # if cnt > 100:
        #     break
    return q_out


logs = []
if __name__ == '__main__':
    # q_0 = np.array([[10, -90, 10, 0, 0, 0, -1000]], dtype=float).reshape((-1, 1))
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # plot_state(q_0, links, ax)
    # # plot_state(test_q[1], links, ax)
    # ax.view_init(30, 30)
    # plt.show()
    # exit()
    i = 0
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')
    i = 1
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')
    i = 2
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')

    q_0 = np.array([[10,-90, 10, 10, 3, -10, -1000]], dtype=float).reshape((-1,1))
    print((fk(q_0,links)[:3,3]))

    q = q_0.copy()
    logs.append(q_0)
    W = np.eye(7)
    # W[4,4] = 10

    # [2734.22  305.74 1648.2]
    xyf_targets = [
        np.array([[2734.22], [305.74], [1648.2], [0], [0], [0]]),
        np.array([[2534.22], [305.74], [1648.2], [0], [0], [0]]),
        np.array([[2534.22], [165.74], [1788.2], [0], [0], [0]]),
        np.array([[2734.22], [165.74], [1788.2], [0], [0], [0]]),
        np.array([[2734.22], [305.74], [1648.2], [0], [0], [0]]),
    ]

    drop = False
    for xyf_target in xyf_targets:
        q_old = q
        # q = pseudoinverse(q, links, xyf_target, W=W)
        # q = dls(q, links, xyf_target)
        # q = null_space(q_old, links, xyf_target)
        q = task_priority(q, links, xyf_target)

        if drop:
            drop = False
        else:
            logs += q
        # break
        q = q[-1] if len(q) > 0 else q_old
    print(len(logs))
    N = len(logs)

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0, N, max(N // 15, 1)):
        plot_state(logs[i], links, ax)
    # ax.view_init(30, 75)
    ax.view_init(30, 30)
    # plt.savefig('pseudoinverse_general.png')
    # plt.savefig('dls_general.png')
    # plt.savefig('null_space_general.png')
    plt.savefig('task_priority_general.png')
    plt.show()

    px = []
    py = []
    pz = []
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0, N):
        pos = fk(logs[i], links)
        px.append(pos[0,3])
        py.append(pos[1,3])
        pz.append(pos[2,3])
    ax.scatter(px, py, pz, s=3)
    ax.scatter(px[0], py[0], pz[0], s=20)
    ax.view_init(30, 75)
    # plt.savefig('pseudoinverse_path.png')
    # plt.savefig('dls_path.png')
    # plt.savefig('null_space_path.png')
    plt.savefig('task_priority_path.png')

    plt.show()

