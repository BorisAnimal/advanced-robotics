from matrices import *
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

np.set_printoptions(2)

links = [50, 100, 200]


def fk1(q):
    return np.linalg.multi_dot([
        Rz(q[0]),
        Tz(links[0]),
        Rx(q[1]),
        Tz(links[1]),
        Ty(q[2] + links[2])
    ])


def fk2(M, S, q):
    return np.linalg.multi_dot([
        skrew_tr(S[0][0], S[0][1], q[0]),
        skrew_tr(S[1][0], S[1][1], q[1]),
        skrew_tr(S[2][0], S[2][1], q[2]),
        M
    ])


def skrew_tr(Sw, Sv, theta):
    m = np.eye(4, 4)
    if Sw == [0, 0, 0] and np.linalg.norm(Sv) == 1:
        m[:3, 3] = np.array(Sv) * theta
    elif np.linalg.norm(Sw) == 1:
        m[:3, :3] = rot(Sw, theta)
        Skew_w = skew_m(Sw)
        m[:3, 3] = (np.eye(3, 3) * theta + (1 - cos(theta)) * Skew_w + (theta - sin(theta)) * Skew_w ** 2) @ Sv
    else:
        raise Exception("some error. {} | {}".format(Sw, Sv))
    return m


def skew_m(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def rot(w, theta):
    C = cos(theta)
    S = sin(theta)
    [w1, w2, w3] = w
    return np.array([
        [C + w1 ** 2 * (1 - C), w1 * w2 * (1 - C) - w3 * S, w1 * w3 * (1 - C) + w2 * S],
        [w1 * w2 * (1 - C) + w3 * S, C + w2 ** 2 * (1 - C), w2 * w3 * (1 - C) - w1 * S],
        [w1 * w3 * (1 - C) - w2 * S, w2 * w3 * (1 - C) + w1 * S, C + w3 ** 2 * (1 - C)],
    ])


def jacobian1(q):
    T = fk1(q)
    T[0:3, 3] = 0
    inv_T = np.transpose(T)

    dT = np.linalg.multi_dot([
        dRz(q[0]),
        Tz(links[0]),
        Rx(q[1]),
        Tz(links[1]),
        Ty(q[2] + links[2])
    ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J1 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
        Rz(q[0]),
        Tz(links[0]),
        dRx(q[1]),
        Tz(links[1]),
        Ty(q[2] + links[2])
    ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J2 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    dT = np.linalg.multi_dot([
        Rz(q[0]),
        Tz(links[0]),
        Rx(q[1]),
        Tz(links[1]),
        Ty(links[2]),
        dTy(q[2])
    ])

    dT = np.linalg.multi_dot([dT, inv_T])
    J3 = np.vstack([dT[0, 3], dT[1, 3], dT[2, 3], dT[2, 1], dT[0, 2], dT[1, 0]])

    return np.hstack([J1, J2, J3])


def jacobian2(S, q):
    w1 = S[0][0]
    w2 = rot(S[0][0], q[0]) @ np.array(S[1][0])
    w3 = [0, 0, 0]  # rot(S[0][0], q[0]) @ rot(S[1][0], q[1]) @ np.array(S[2][1])
    v1 = -skew_m(w1) @ np.array(S[0][1])
    v2 = -skew_m(w2) @ np.array(S[1][1])
    v3 = rot(S[0][0], q[0]) @ rot(S[1][0], q[1]) @ np.array(S[2][1])

    return np.array([
        np.hstack([w1, v1]),
        np.hstack([w2, v2]),
        np.hstack([w3, v3]),
    ]).T


def plot_state(q):
    trans = [
        Rz(q[0]),
        'O',
        Tz(links[0]),
        'O',
        Rx(q[1]),
        'O',
        Tz(links[1]),
        'O',
        Ty(q[2] + links[2]),
        'O',
    ]
    res = trans[0]
    points = [res[0:3, 3].flatten()]
    for t in trans[1:]:
        if t != 'O':
            res = res @ t
            points.append(res[0:3, 3].flatten())
    points = np.array(points)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10)
    ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=0.7)
    plt.show()


if __name__ == '__main__':
    q = [-pi/4, 0, 2]
    T1 = fk1(q)
    print("T1:\n", T1)

    M = np.eye(4, 4)
    M[:3, 3] = [0, links[2], links[0] + links[1]]
    # print(M)
    S = [
        [[0, 0, 1], [0, 0, 0]],
        [[1, 0, 0], [0, 0, links[0]]],
        [[0, 0, 0], [0, 1, 0]],
    ]

    T2 = fk2(M, S, q)
    print("T2:\n", T2)
    assert np.isclose(T1, T2).all(), (T1, T2)

    J1 = jacobian1(q)
    print("Jacobian usual:\n", J1)

    J2 = jacobian2(S, q)
    print("Jacobian skrew:\n", J2)

    plot_state(q)
