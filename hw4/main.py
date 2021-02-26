import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matrices import *
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
                            Tx(links[5]),
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
                            Tx(links[5]),
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
                            Tx(links[5]),
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
                            Tx(links[5]),
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
                            Tx(links[5]),
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
                            Tx(links[5]),
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
                            Tx(links[5]),
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
                            Tx(links[5]),
                             ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J7 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])


    return np.hstack([J1, J2, J3, J4, J5, J6, J7])




from mpl_toolkits.mplot3d import Axes3D
def plot_state(q, links, ax):
    trans = [
        Tx(-q[6]),
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
        Tz(-links[4]),
        'O',
        Rx(q[3]),
        'O',
        Ry(q[4]),
        'O',
        Rx(q[5]),
        'O',
        Tx(links[5]),
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
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=30)
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
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
    return np.vstack([[dx], [dy], [dz],[0], [0] [0]])

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

def update_dq(d_q, d_q_max):
    d_q_new = np.maximum(np.minimum(d_q, d_q_max), -d_q_max)
    dk = d_q / d_q_new
    dk_abs = np.abs(d_q / d_q_new)
    if np.max(dk_abs) > 1.0:
        print(dk_abs)
    return d_q_new

def pseudoinverse(q, links, xyf_target, W=np.eye(7)):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)
    while np.sqrt(np.sum(d_xyf ** 2)) > 0.01:
        del_e = update_fraction(d_xyf,5)

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


N = 100
# dt = N / 5
x = np.linspace(0, -1, N)
y = 0.5*np.sin(-10*x)
dx = (x[1:] - x[:-1])
dx = np.hstack([dx[0], dx])
dy = (y[1:] - y[:-1])
dy = np.hstack([dy[0], dy])




logs = []
if __name__ == '__main__':
    fig = plt.figure()
    ax = Axes3D(fig)
    plot_state(test_q[0], links, ax)
    # plot_state(test_q[1], links, ax)
    ax.view_init(0, 90)
    plt.show()
    exit()
    i = 0
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')
    i = 1
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')
    i = 2
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')

    q_0 = np.array([[0,-90, 0, 0, 0, 0, 0]]).reshape((-1,1))
    # plot_state(q_0, links)
    # plt.show()

    # transl = fk(q_0, links)
    # x += transl[0, 2]
    # y += transl[1, 2]
    # phi = np.arcsin(transl[1, 0])
    xyf_0 = get_pos(q_0, links)
    q = q_0.copy()
    logs.append(q_0)
    W = np.eye(7)
    # W[0,0] = 10
    xyf_target = np.array([[1150.0], [100.0], [2000], [0], [0], [0]])

    for i in range(N):
        q_old = q
        # xyf_target = np.array([[x[i]], [y[i]], [np.pi/2]])
        q = pseudoinverse(q, links, xyf_target, W=W)
        # q = task_priority(q, links, xyf_target)
        # q = sdv(q, links, xyf_target)
        # q = dls(q, links, xyf_target)
        # q = null_space(q_old, links, xyf_target)
        logs += q
        break
        q = q[-1] if len(q) > 0 else q_old
    print(len(logs))
    N = len(logs)

    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(0, N, max(N // 20, 1)):
        plot_state(logs[i], links, ax)
    ax.view_init(10, 45)
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
    ax.scatter(px, py, pz, s=10)
    plt.show()

