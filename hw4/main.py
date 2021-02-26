import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matrices import *
np.set_printoptions(2)

test_q = np.array([
    [0,-90, 0, 0, 0, 0, 0],
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
                            # Tx(-791.42),
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

from mpl_toolkits.mplot3d import Axes3D
def plot_state(q, links):
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
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=30)
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    ax.view_init(0, 30)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plot_state(test_q[0], links)
    plt.show()
    i = 0
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')
    i = 1
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')
    i = 2
    print(f'{i}: {fk(test_q[i], links)[:3, 3]} || {test_p[i]}')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
