import numpy as np
import matplotlib.pyplot as plt
from numpy import inf
from matplotlib.animation import FuncAnimation


def Rz(q):
    T = np.array([[np.cos(q), -np.sin(q), 0],
                  [np.sin(q), np.cos(q), 0],
                  [0, 0, 1]], dtype=float)
    return T


def dRz(q):
    T = np.array([[-np.sin(q), -np.cos(q), 0],
                  [np.cos(q), -np.sin(q),  0],
                 [0, 0, 0]], dtype=float)
    return T


def Tx(x):
    T = np.array([[1, 0, x],
                  [0, 1, 0],
                  [0, 0, 1]], dtype=float)
    return T


def dTx(x=None):
    T = np.array([[0, 0, 1],
                  [0, 0, 0],
                  [0, 0, 0]], dtype=float)
    return T


def fk(q, links):
    T = np.linalg.multi_dot([
        Rz(q[0]),
        Tx(links[0]),
        Rz(q[1]),
        Tx(links[1]),
        Rz(q[2]),
        Tx(links[2]),
        Rz(q[3]),
        Tx(links[3]),
    ])
    return T


def jacobian(q, links):
    """
    :return: matrix of shape (6,4)
    """
    T = fk(q, links)
    T[0:2, -1] = 0
    inv_T = np.transpose(T)

    dT = np.linalg.multi_dot([
        dRz(q[0]),
        Tx(links[0]),
        Rz(q[1]),
        Tx(links[1]),
        Rz(q[2]),
        Tx(links[2]),
        Rz(q[3]),
        Tx(links[3]),
    ])
    dT = np.linalg.multi_dot([dT, inv_T])
    J1 = np.vstack([dT[0, 2], dT[1, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
        Rz(q[0]),
        Tx(links[0]),
        dRz(q[1]),
        Tx(links[1]),
        Rz(q[2]),
        Tx(links[2]),
        Rz(q[3]),
        Tx(links[3]),
    ])
    dT = np.linalg.multi_dot([dT, inv_T])
    J2 = np.vstack([dT[0, 2], dT[1, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
        Rz(q[0]),
        Tx(links[0]),
        Rz(q[1]),
        Tx(links[1]),
        dRz(q[2]),
        Tx(links[2]),
        Rz(q[3]),
        Tx(links[3]),
    ])
    dT = np.linalg.multi_dot([dT, inv_T])
    J3 = np.vstack([dT[0, 2], dT[1, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
        Rz(q[0]),
        Tx(links[0]),
        Rz(q[1]),
        Tx(links[1]),
        Rz(q[2]),
        Tx(links[2]),
        dRz(q[3]),
        Tx(links[3]),
    ])
    dT = np.linalg.multi_dot([dT, inv_T])
    J4 = np.vstack([dT[0, 2], dT[1, 2], dT[1, 0]])
    return np.hstack([J1, J2, J3, J4])


def plot_traj(q, links, bigger=False):
    trans = [
        Rz(q[0]),
        Tx(links[0]),
        'O',
        Rz(q[1]),
        Tx(links[1]),
        'O',
        Rz(q[2]),
        Tx(links[2]),
        'O',
        Rz(q[3]),
        Tx(links[3]),
        'O',
    ]
    res = trans[0]
    points = [res[0:2, 2].flatten()]
    for t in trans[1:]:
        if t != 'O':
            res = res @ t
            points.append(res[0:2, 2].flatten())
    points = np.array(points)
    if not bigger:
        plt.scatter(points[-1, 0], points[-1, 1],  s=3)
    else:
        plt.scatter(points[-1, 0], points[-1, 1],  s=10)


def plot_state(q, links, bigger=False):
    trans = [
        Rz(q[0]),
        Tx(links[0]),
        'O',
        Rz(q[1]),
        Tx(links[1]),
        'O',
        Rz(q[2]),
        Tx(links[2]),
        'O',
        Rz(q[3]),
        Tx(links[3]),
        'O',
    ]
    res = trans[0]
    points = [res[0:2, 2].flatten()]
    for t in trans[1:]:
        if t != 'O':
            res = res @ t
            points.append(res[0:2, 2].flatten())
    points = np.array(points)
    plt.scatter(points[:, 0], points[:, 1],  s=2)
    if not bigger:
        plt.scatter(points[-1, 0], points[-1, 1],  s=15)
    else:
        plt.scatter(points[-1, 0], points[-1, 1],  s=25)
    plt.plot(points[:, 0], points[:, 1])


def get_difference(current_xyf, target_xyf, ext=False):
    dx = target_xyf[0] - current_xyf[0]
    dy = target_xyf[1] - current_xyf[1]
    dphi = (target_xyf[2] - current_xyf[2] + 3* np.pi)%(2*np.pi) - np.pi
    if ext:
        return np.vstack([[dx], [dy], [dphi]])
    else:
        return np.vstack([[dx], [dy], [0]])

def update_fraction(d_xyf, max_d = 0.01):
    kx = abs(d_xyf[0]) / min(abs(d_xyf[0]),max_d)
    dkx = min(abs(d_xyf[0]),max_d)

    ky = abs(d_xyf[1]) / min(abs(d_xyf[1]),max_d)
    dky = min(abs(d_xyf[1]),max_d)

    return d_xyf / max(kx, ky)

def update_dq(d_q, d_q_max):
    d_q_new = np.maximum(np.minimum(d_q, d_q_max), -d_q_max)
    dk = d_q / d_q_new
    dk_abs = np.abs(d_q / d_q_new)
    if np.max(dk_abs) > 1.0:
        print(dk_abs)
    return d_q_new

def pseudoinverse(q, links, xyf_target, W=np.eye(4)):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)
    while np.sqrt(np.sum(d_xyf ** 2)) > 0.01:
        del_e = update_fraction(d_xyf,0.01)

        J = jacobian(q, links)[:2,:]
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
        d_q = J_psevd.dot(del_e[:2,:])

        d_q = update_dq(d_q, d_q_max)
        q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        q_out.append(q.copy())

        d_xyf = get_difference(get_pos(q, links), xyf_target)
        print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, dq: {d_q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
    return q_out

def task_priority(q, links, xyf_target):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)
    cnt = 0
    print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
    while np.sqrt(np.sum(d_xyf ** 2)) > 0.01:

        del_e = update_fraction(get_difference(xyf_0, xyf_target,ext=True),0.02)

        J = jacobian(q, links)
        J_1 = J[:2,:].reshape((2,-1))
        J_2 = J[2,:].reshape((1,-1))
        J_psevd_1 = np.linalg.multi_dot([
            J_1.T,
            np.linalg.inv(
                J_1.dot(J_1.T)
            )
        ])
        P_1 = np.eye(4) - J_psevd_1.dot(J_1)

        J_psevd_2 = np.linalg.multi_dot([
            J_2.T,
            np.linalg.inv(
                J_2.dot(J_2.T)
            )
        ])


        P_2_1 = del_e[2,:] - \
            np.linalg.multi_dot([
            J_2,
            J_psevd_1,
            del_e[:2,:]
            ])

        d_q = J_psevd_1.dot(del_e[:2,:]) + \
              np.linalg.multi_dot([
                np.linalg.pinv(J_2.dot(P_1)),
                P_2_1
            ])


        # d_q = np.maximum(np.minimum(d_q, d_q_max), -d_q_max)
        d_q = update_dq(d_q, d_q_max)
        q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        # q = q + d_q
        q_out.append(q.copy())

        xyf_0 = get_pos(q, links)
        d_xyf = get_difference(xyf_0, xyf_target)
        cnt +=1
        print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, dq: {d_q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
        if cnt > 100:
            break
    return q_out


def sdv(q, links, xyf_target):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)
    while np.sqrt(np.sum(d_xyf ** 2)) > 0.01:
        del_e = update_fraction(d_xyf,0.01)

        J = jacobian(q, links)[:2,:]

        U, S, Vh = np.linalg.svd(J)
        Ss = np.zeros((J.shape))
        S = np.diag(S)
        Ss[:S.shape[0], :S.shape[1]] = S
        Ss_inv = np.linalg.pinv(Ss)
        J_psevd = np.linalg.multi_dot([
            Vh.T,
            Ss_inv,
            U.T
        ])
        d_q = J_psevd.dot(del_e[:2,:])

        d_q = update_dq(d_q, d_q_max)
        q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        q_out.append(q.copy())

        xyf_0 = get_pos(q, links)
        d_xyf = get_difference(xyf_0, xyf_target)

        print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, dq: {d_q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
    return q_out

def dls(q, links, xyf_target, mu=0.001):
    q_min = np.array([[-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4], [-np.pi * 2 / 4]])
    q_max = np.array([[np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4], [np.pi * 2 / 4]])
    d_q_max = np.array([[np.pi / 100], [np.pi / 100], [np.pi / 100], [np.pi / 100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0,xyf_target)

    while np.sqrt(np.sum(d_xyf ** 2)) > 0.01:
        del_e = update_fraction(d_xyf,0.01)

        J = jacobian(q, links)[:2,:]

        d_q = np.linalg.multi_dot([
            J.T,
            np.linalg.inv(
                np.linalg.multi_dot([
                    J,
                    J.T
                ])
                +
                mu**2 *
                np.eye(2)
            ),
            del_e[:2,:]
        ])

        d_q = update_dq(d_q, d_q_max)
        q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        q_out.append(q.copy())

        xyf_0 = get_pos(q, links)
        d_xyf = get_difference(xyf_0, xyf_target)

        print(f" Dist: {np.sqrt(np.sum(d_xyf ** 2))}, q: {q.T}, dq: {d_q.T}, xyf: {xyf_0.T}, d xyf: {d_xyf.T}")
    return q_out



# the manipulability measure
def dH_manipulability(q, q_old, links):
    dH = []
    dq = 0.001
    for j in range(q.shape[0]):
        t_q_old = q.copy()
        t_q_old[j] +=dq
        J_old = jacobian(t_q_old, links)[:2,:]
        H_old = np.sqrt(np.linalg.det(J_old.dot(J_old.T)))
        J = jacobian(q, links)[:2,:]
        H = np.sqrt(np.linalg.det(J.dot(J.T)))
        dH.append((H - H_old) / (q[j] - t_q_old[j]))
    return np.array(dH).reshape((-1, 1))


def dH_joint_limits(q, q_old, links, q_min, q_max):
    dH = []
    dq = 0.001
    # the distance frome mechanical joint limits
    no_of_joints = q.shape[0]
    q_dash =(q_max+q_min) /2
    for j in range(q.shape[0]):
        t_q_old = q.copy()
        t_q_old[j] += dq
        h = 0
        h_old = 0
        for k in range(no_of_joints):
            h += ((q[k] - q_dash[k]) / (q_max[k] - q_min[k])) ** 2
            h_old += ((t_q_old[k] - q_dash[k]) / (q_max[k] - q_min[k])) ** 2
        H = 1 / (2 * no_of_joints) * h
        H_old = 1 / (2 * no_of_joints) * h_old
        # t_dh = (H - H_old) / (dq)
        t_dh = -(-H + H_old) / (dq)
        t_dh[t_dh == -inf] = 0
        t_dh[t_dh == inf] = 0
        dH.append(t_dh)
    return np.array(dH).reshape((-1, 1))


def get_pos(q, links):
    transl = fk(q, links)
    phi = np.arctan2(transl[1, 0], transl[0,0])
    xyf_0 = np.vstack([transl[0, 2], transl[1, 2], [phi]])
    return xyf_0



def null_space(q, links, xyf_target, k0=0.01):
    q_min = np.array([[-np.pi*2/4], [-np.pi*2/4], [-np.pi*2/4], [-np.pi*2/4]])
    q_max = np.array([[np.pi*2/4], [np.pi*2/4], [np.pi*2/4], [np.pi*2/4]])
    d_q_max = np.array([[np.pi/100], [np.pi/100], [np.pi/100], [np.pi/100]])
    q_out = []
    xyf_0 = get_pos(q, links)
    d_xyf = get_difference(xyf_0, xyf_target)
    q_old = None
    cnt = 0

    while np.sqrt(np.sum(d_xyf**2))> 0.01:
        del_e = update_fraction(d_xyf,0.01)


        dH = dH_manipulability(q, q_old, links)
        # dH = dH_joint_limits(q, q_old, links, q_min, q_max)
        d_q_0 = k0 * np.nan_to_num(dH)

        J = jacobian(q, links)[:2,:]
        J_psevd =J.T.dot(np.linalg.inv(J.dot(J.T)))
        d_q = J_psevd.dot(del_e[:2,:]) + (np.eye(4) - J_psevd.dot(J)).dot(d_q_0)


        d_q = update_dq(d_q, d_q_max)
        q_old = q
        q = np.maximum(np.minimum(q + d_q, q_max), q_min)
        q_out.append(q.copy())

        xyf_0 = get_pos(q, links)
        d_xyf = get_difference(xyf_0, xyf_target)

        print(f" Dist: {np.sqrt(np.sum(d_xyf**2))}, q: {q.T}, dq: {d_q.T}, xyf: {xyf_0.T}")
        cnt +=1
        # if cnt > 1000:
        #     break

    return q_out





links = [1.0, 1.0, 1.0, 1.0]

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')




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
    q_0 = np.array([[np.pi/6],
                    [np.pi/3],
                    [np.pi*0.9/2],
                    [np.pi/4]])
    # q_0 = np.array([[-0.79646527],  [1.02904732],  [1.51238868], [-0.16789199]])
    # q_0 = np.array([[-0.59543411],  [1.42382634],  [1.57079633],  [0.02103457]])
    # q_0 = np.array([[-0.4160008],   [0.51277499],  [0.84304539],  [0.63097674]])
    q_0 = np.array([[-0.99059589],  [1.44742583],  [1.57079633], [-0.46189092]])
    # plot_state(q_0, links)
    # plt.show()

    transl = fk(q_0, links)
    x += transl[0,2]
    y += transl[1,2]
    phi = np.arcsin(transl[1,0])
    xyf_0 = get_pos(q_0, links)
    q = q_0.copy()
    logs.append(q_0)
    W = np.eye(4)
    # W[0,0] = 10
    xyf_target = np.array([[1.5], [1.5], [np.pi/2]])
    # xyf_target = np.array([[1.0], [1.5], [np.pi/2]])
    # xyf_target = np.array([[-0.5], [0.5], [np.pi/2]])
    # xyf_target = np.array([[-0.5], [0.5], xyf_0[2]])

    for i in range(N):
        q_old = q
        xyf_target = np.array([[x[i]], [y[i]], [np.pi/2]])
        # q = pseudoinverse(q, links, xyf_target, W=W)
        q = task_priority(q, links, xyf_target)
        # q = sdv(q, links, xyf_target)
        # q = dls(q, links, xyf_target)
        # q = null_space(q_old, links, xyf_target)
        # logs.append(q.copy())
        logs += q
        # print("hh")
        # break
        q = q[-1] if len(q) > 0 else q_old
    # print(f"{i}: {q.T}")
    print(len(logs))
    N = len(logs)
    # plt.plot(x, y, c='black', linewidth=2)

    for i in range(0, N, max(N//70,1)):
    # for i in range(0, N):
        plot_state(logs[i], links, bigger=(i == 0))
    # plot_state(logs[1], links, bigger=True)
    plt.show()

    for i in range(0, N):
        plot_traj(logs[i], links, bigger=(i == 0))
    plt.show()



