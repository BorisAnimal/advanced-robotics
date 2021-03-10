from matrices import *
import numpy as np
import matplotlib.pyplot as plt
from math import atan, tan, sin, cos, pi

Sb = 567
Sp = 76

sqrt3 = 3 ** 0.5
Wb = Sb * sqrt3 / 6
Ub = Sb / sqrt3

Wp = Sp * sqrt3 / 6
Up = Sp / sqrt3

links = [300, 800]


def fk(thetas):
    [theta1, theta2, theta3] = thetas
    [L, l] = links

    x1 = 0.0
    y1 = Up - Wb - L * cos(theta1)
    z1 = -L * sin(theta1)

    x2 = sqrt3 * (Wb + L * cos(theta2)) / 2 - Sp / 2
    y2 = (Wb + L * cos(theta2)) / 2 - Wp
    z2 = -L * sin(theta2)

    x3 = - sqrt3 * (Wb + L * cos(theta3)) / 2 + Sp / 2
    y3 = (Wb + L * cos(theta3)) / 2 - Wp
    z3 = -L * sin(theta3)

    r = l

    a = 2 * (x3 - x1)
    b = 2 * (y3 - y1)
    c = x3 ** 2 + y3 ** 2 - x1 ** 2 - y1 ** 2
    d = 2 * (x3 - x2)
    e = 2 * (y3 - y2)
    f = x3 ** 2 + y3 ** 2 - x2 ** 2 - y2 ** 2

    x = (c * e - b * f) / (a * e - b * d)
    y = (a * f - c * d) / (a * e - b * d)

    B = -2 * z1
    C = z1 ** 2 - r ** 2 + (x - x1) ** 2 + (y - y1) ** 2
    z = (-B - (B ** 2 - 4 * C) ** 0.5) / 2

    return [x, y, z]


def ik(EE):
    a = Wb - Up
    b = Sp / 2 - sqrt3 / 2 * Wb
    c = Wp - Wb / 2
    [x, y, z] = EE
    [L, l] = links

    E1 = 2 * L * (y + a)
    F1 = 2 * z * L
    G1 = x ** 2 + y ** 2 + z ** 2 + a ** 2 + L ** 2 + 2 * y * a - l ** 2

    E2 = -L * (sqrt3 * (x + b) + y + c)
    F2 = 2 * z * L
    G2 = x ** 2 + y ** 2 + z ** 2 + b ** 2 + c ** 2 + L ** 2 + 2 * (x * b + y * c) - l ** 2

    E3 = L * (sqrt3 * (x - b) - y - c)
    F3 = 2 * z * L
    G3 = x ** 2 + y ** 2 + z ** 2 + b ** 2 + c ** 2 + L ** 2 + 2 * (y * c - x * b) - l ** 2

    E = [E1, E2, E3]
    F = [F1, F2, F3]
    G = [G1, G2, G3]

    thetas = []
    for (e, f, g) in zip(E, F, G):
        ti = (-f - (e ** 2 + f ** 2 - g ** 2) ** 0.5) / (g - e)
        thetas.append(2 * atan(ti))
        # thetas.append((ti,2 * atan((-f + (e ** 2 + f ** 2 - g ** 2) ** 0.5) / (g - e))))
    return thetas


def plot_z_circle(xyz, r):
    [x, y, z] = xyz
    n = np.linspace(0, 2 * pi, 200)
    plt.plot(x + r * np.cos(n),
             y + r * np.sin(n),
             z)


def plot_robot(xyz, thetas):
    [x, y, z] = xyz
    [q1, q2, q3] = thetas
    [L, l] = links
    plt.figure()
    ax = plt.axes(projection='3d')

    # Plot fixed base
    plot_z_circle([0, 0, 0], Wb)
    # Plot moving platform
    plot_z_circle(xyz, Up)

    # Joints' positions on side of base
    B1 = [0, -Wb, 0]
    B2 = [sqrt3 * Wb / 2, Wb / 2, 0]
    B3 = [-sqrt3 * Wb / 2, Wb / 2, 0]

    # Joints' positions on side of platform
    P1 = [x, y - Up, z]
    P2 = [x + Sp / 2, y + Wp, z]
    P3 = [x - Sp / 2, y + Wp, z]

    A1 = [
        0.0,
        - Wb - L * cos(q1),
        -L * sin(q1)
    ]
    A2 = [
        sqrt3 * (Wb + L * cos(q2)) / 2,
        (Wb + L * cos(q2)) / 2,
        -L * sin(q2)
    ]
    A3 = [
        - sqrt3 * (Wb + L * cos(q3)) / 2,
        (Wb + L * cos(q3)) / 2,
        - L * sin(q3)
    ]
    for (b, a, p) in zip([B1, B2, B3], [A1, A2, A3], [P1, P2, P3]):
        plt.plot(*np.vstack([b, a, p]).T)

    plt.show()


def __ik(x0, y0, z0):
    [r1, r2] = links

    cos120 = -0.5
    sin120 = np.sin(np.deg2rad(120))

    q1 = ik([x0, y0, z0])
    [theta1, theta2, theta3] = q1

    alpha1 = np.arcsin(y0 / r2)
    alpha2 = np.arcsin((y0 * cos120 - x0 * sin120) / r2)
    alpha3 = np.arcsin((y0 * cos120 + x0 * sin120) / r2)
    q3 = [alpha1, alpha2, alpha3]

    beta1 = np.arccos((-(r1 * np.cos(theta1) - x0) * np.cos(theta1) + np.sqrt(
        -r1 ** 2 * np.cos(theta1) ** 2 + 2 * r1 * x0 * np.cos(theta1) + r2 ** 2 * np.cos(
            alpha1) ** 2 - x0 ** 2) * np.sin(theta1)) / (r2 * np.cos(alpha1)))

    beta2 = np.arccos((-(r1 * np.cos(theta2) - (x0 * cos120 + y0 * sin120)) * np.cos(theta2) + np.sqrt(
        -r1 ** 2 * np.cos(theta2) ** 2 + 2 * r1 * (x0 * cos120 + y0 * sin120) * np.cos(theta2) + r2 ** 2 * np.cos(
            alpha2) ** 2 - (x0 * cos120 + y0 * sin120) ** 2) * np.sin(theta2)) / (r2 * np.cos(alpha2)))

    beta3 = np.arccos((-(r1 * np.cos(theta3) - (x0 * cos120 - y0 * sin120)) * np.cos(theta3) + np.sqrt(
        -r1 ** 2 * np.cos(theta3) ** 2 + 2 * r1 * (x0 * cos120 - y0 * sin120) * np.cos(theta3) + r2 ** 2 * np.cos(
            alpha3) ** 2 - (x0 * cos120 - y0 * sin120) ** 2) * np.sin(theta3)) / (r2 * np.cos(alpha3)))

    q2 = [beta1, beta2, beta3]

    return q1, q2, q3


def jacobian_theta(q1, q2, q3):
    L = links[0]
    eq1 = -L * (-np.sin(q1[0]) * np.cos(q1[0] + q2[0]) - np.cos(q1[0]) * np.sin(q1[0] + q2[0]))
    eq2 = -L * (-np.sin(q1[1]) * np.cos(q1[1] + q2[1]) - np.cos(q1[1]) * np.sin(q1[1] + q2[1]))
    eq3 = -L * (-np.sin(q1[2]) * np.cos(q1[2] + q2[2]) - np.cos(q1[2]) * np.sin(q1[2] + q2[2]))
    J = np.diag([eq1, eq2, eq3])
    return J


def jacobian_z(q1, q2, q3):
    eq1 = np.hstack([-np.cos(q1[0] + q2[0]), -np.tan(q3[0]), np.sin(q1[0] + q2[0])])
    eq2 = np.hstack([-np.cos(q1[1] + q2[1]), -np.tan(q3[1]), np.sin(q1[1] + q2[1])])
    eq3 = np.hstack([-np.cos(q1[2] + q2[2]), -np.tan(q3[2]), np.sin(q1[2] + q2[2])])

    return np.vstack([eq1, eq2, eq3])


def singularity_2d_z(z=-300, amp=900):
    t = list(range(-amp, amp, 5))
    X, Y, M = [], [], []
    for x in t:
        for y in t:
            try:
                Qs = __ik(x, y, z)

                Jz = jacobian_z(*Qs)
                J_theta = jacobian_theta(*Qs)

                J = np.linalg.inv(J_theta) @ Jz
                m = np.linalg.det(J @ J.T) ** 0.5
                if np.isnan(m):
                    m = 1.0
            except:
                m = 1.0
            X.append(x)
            Y.append(y)
            M.append(m)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("z=" + str(z))
    plt.scatter(X, Y, c=M)
    plt.show()


def singularity_2d_y(y=0, amp=900):
    t = list(range(-amp, amp, 5))
    Z, X, M = [], [], []
    for z in range(-1200, 0, 5):
        for x in t:
            try:
                Qs = __ik(x, y, z)

                Jz = jacobian_z(*Qs)
                J_theta = jacobian_theta(*Qs)

                J = np.linalg.inv(J_theta) @ Jz
                m = np.linalg.det(J @ J.T) ** 0.5
                if np.isnan(m):
                    m = 1.0
            except:
                m = 1.0
            Z.append(z)
            X.append(x)
            M.append(m)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title("y=" + str(y))
    plt.scatter(X, Z, c=M)
    plt.show()


def singularity_3d(amp=900):
    plt.figure()
    ax = plt.axes(projection='3d')
    t = list(range(-amp, amp, 40))
    X, Y, Z, M = [], [], [], []
    for z in range(-1200, -40, 40):
        for x in t:
            for y in t:
                try:
                    Qs = __ik(x, y, z)

                    Jz = jacobian_z(*Qs)
                    J_theta = jacobian_theta(*Qs)

                    J = np.linalg.inv(J_theta) @ Jz
                    m = np.linalg.det(J @ J.T) ** 0.5
                    if np.isnan(m):
                        m = 1.0
                except:
                    m = 1.0
                X.append(x)
                Y.append(y)
                Z.append(z)
                M.append(m)
    cmap = ax.scatter3D(X, Y, Z, c=M, s=0.5, alpha=0.7)
    plt.colorbar(cmap)
    plt.show()


if __name__ == '__main__':
    xyz = [100, -200, -805]
    # xyz = [0, 0, -800]
    thetas = ik(xyz)
    print(thetas)
    xyz_prime = fk(thetas)
    print(xyz, xyz_prime)

    plot_robot(xyz, thetas)

    singularity_2d_z(z=-300)
    singularity_2d_z(z=-850)
    singularity_2d_z(z=-1000)
    singularity_2d_y()
    singularity_3d()

    # thetas = [pi / 4, -pi / 6, -pi / 3]
    # xyz = fk(thetas)
    # print(xyz)
    # thetas_prime = ik(xyz)
    # print(thetas)
    # print(thetas_prime)
