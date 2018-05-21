  # Needed for python 2/3 print function; for float division

from math import sqrt, tan, pi

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


# Todo: Look again at article on numpy speeds, make sure I'm making the best out of it


class Airfoil:
    """
    Class that aims at making the translation of coordinates into various systems (CST, XY, PARSEC) easy.
    """

    def __init__(self, **kwargs):
        """

        :param fpath: File path of the XY coordinates
        :param p: PARSEC array description
        :param x_u: Upper panel points distribution
        :param x_l: Lower panel points distribution
        """

        keys = list(kwargs.keys())

        # Some default values, in case none is specified.
        self.fpath = None
        self.n_points = 100

        # Only assign default x_l and x_u if they are not defined as arguments; nor in a XY file
        if 'x_l' and 'fpath' not in keys:
            self.x_l = -0.5 * np.cos(np.linspace(pi, 0, self.n_points)) + 0.5

        if 'x_u' and 'fpath' not in keys:
            self.x_u = -0.5 * np.cos(np.linspace(0, pi, self.n_points)) + 0.5

        # Need to start with those inputs first.
        for key, value in list(kwargs.items()):
            if key == 'fpath':  # XY CSV coordinates file
                self.fpath = value
                self.read_coords()
            if key == 'x_u':
                self.x_u = value
            if key == 'x_l':
                self.x_l = value

        for key, value in list(kwargs.items()):
            if key == 'p':  # array [R_le, X_up, Z_up, ZXXup, X_low, Z_low, ZXXlow, Y_TE, t_TE, alpha_TE, beta_TE]
                self.p = value
                self.r_le = value[0]
                self.x_up, self.z_up, self.zxx_up = value[1:4]
                self.x_low, self.z_low, self.zxx_low = value[4:7]
                self.y_te, self.t_te, self.alpha_te, self.beta_te = value[7:]

                self.y_l, self.y_u = self.parsec_to_xy(self.p)
                self.y = np.concatenate([self.y_l, self.y_u])

        if not self.fpath:
            self.x = np.concatenate([self.x_l, self.x_u])

    def read_coords(self):
        """
        Reads Xu, Xl, Yu, Yl from a standard airfoil file
        """
        if self.fpath:
            with open(self.fpath, 'r') as datafile:
                _x = []
                _y = []
                for line in datafile:
                    x, y = line.split(' ', 1)
                    _x.append(float(x))
                    _y.append(float(y))

            self.x = np.array(_x)
            self.y = np.array(_y)

            # Todo: need to make it clearer that I need the same number of points for both surfaces to work!!
            self.x_u, self.x_l = np.split(self.x, 2)
            self.y_u, self.y_l = np.split(self.y, 2)

    def plot_airfoil(self, color='k'):
        """ Plots the airfoil. """

        plt.plot(self.x, self.y, color)
        plt.axis('equal')
        plt.title('Airfoil PARSEC')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def parsec_to_xy(self, p):
        """
        Solve the linear system to get PARSEC y coordinates
        Required inputs: Xu, Xl, p
        :return:
        """
        # todo: Need to refind source and double check this. Espacially only p[2] in c_up??

        # Define matrix
        c1 = np.array([1, 1, 1, 1, 1, 1])
        c2 = np.array(
            [p[1] ** 0.5, p[1] ** (3 / 2), p[1] ** (5 / 2), p[1] ** (7 / 2), p[1] ** (9 / 2), p[1] ** (11 / 2)])
        c3 = np.array([1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2])
        c4 = np.array(
            [(1 / 2) * p[1] ** (-1 / 2), (3 / 2) * p[1] ** (1 / 2), (5 / 2) * p[1] ** (3 / 2),
             (7 / 2) * p[1] ** (5 / 2),
             (9 / 2) * p[1] ** (7 / 2), (11 / 2) * p[1] ** (9 / 2)])
        c5 = np.array([(-1 / 4) * p[1] ** (-3 / 2), (3 / 4) * p[1] ** (-1 / 2), (15 / 4) * p[1] ** (1 / 2),
                       (35 / 4) * p[1] ** (3 / 2), (53 / 4) * p[1] ** (5 / 2), (99 / 4) * p[1] ** (7 / 2)])
        c6 = np.array([1, 0, 0, 0, 0, 0])

        c_up = np.array([c1, c2, c3, c4, c5, c6])

        c7 = np.array([1, 1, 1, 1, 1, 1])
        c8 = np.array([p[4] ** (1 / 2), p[4] ** (3 / 2),
                       p[4] ** (5 / 2), p[4] ** (7 / 2), p[4] ** (9 / 2), p[4] ** (11 / 2)])
        c9 = np.array([1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2])
        c10 = np.array([(1 / 2) * p[4] ** (-1 / 2), (3 / 2) * p[4] ** (1 / 2), (5 / 2) * p[4] ** (3 / 2),
                        (7 / 2) * p[4] ** (5 / 2), (9 / 2) * p[4] ** (7 / 2), (11 / 2) * p[4] ** (9 / 2)])
        c11 = np.array([(-1 / 4) * p[4] ** (-3 / 2), (3 / 4) * p[4] ** (-1 / 2), (15 / 4) * p[4] ** (1 / 2),
                        (35 / 4) * p[4] ** (3 / 2), (53 / 4) * p[4] ** (5 / 2), (99 / 4) * p[4] ** (7 / 2)])
        c12 = np.array([1, 0, 0, 0, 0, 0])

        c_low = np.array([c7, c8, c9, c10, c11, c12])

        # todo: unsure why use radians instead of just tan here. Just following old matlab script for now.
        b_up = np.array([p[7] + p[8] / 2, p[2], tan(p[9] - p[10] / 2), 0, p[3], sqrt(2 * p[0])])
        b_low = np.array([-p[7] + p[8] / 2, p[5], tan(p[9] - p[10] / 2), 0, p[6], sqrt(2 * p[0])])

        a_up = np.linalg.solve(c_up, b_up)
        a_low = np.linalg.solve(c_low, b_low)
        a = np.concatenate([a_up, a_low])

        # STATUS: a is ok too
        y_u = a[0] * self.x_u ** 0.5 + a[1] * self.x_u ** 1.5 + a[2] * self.x_u ** 2.5 + a[3] * self.x_u ** 3.5 + \
              a[4] * self.x_u ** 4.5 + a[5] * self.x_u ** 5.5
        y_l = -(a[6] * self.x_l ** 0.5 + a[7] * self.x_l ** 1.5 + a[8] * self.x_l ** 2.5 + a[9] * self.x_l ** 3.5 +
                a[10] * self.x_l ** 4.5 + a[11] * self.x_l ** 5.5)

        return y_l, y_u

    def xy_to_parsec(self, disp=False):
        """
        Runs an optimization in order to deduce the p array from XY coordinates.
        Input: XY coords, output: p array
        """

        # p = [R_le, X_up, Z_up, ZXXup, X_low, Z_low, ZXXlow, Y_TE, t_TE, alpha_TE, beta_TE]
        p_init = np.array([0.032, 0.3, 0.08, 0, 0.3, 0.06, 0.0, 0.0211, 0.0, 0.0, 0.5])
        bounds = ((0.0, 0.5),  # r_le
                  (0.0, 1.0),  # x_up
                  (0.0, 0.2),  # z_up
                  (-10, +10),  # zxx_up
                  (0.0, 1.0),  # x_low
                  (0.0, 0.3),  # z_low
                  (-10, 10),  # zxx_low
                  (0.0, 0.1),  # y_te
                  (0.0, 0.1),  # t_te
                  (-pi / 4, pi / 4),  # alpha_te
                  (-pi / 4, pi / 4))  # beta_te

        res = opt.minimize(fun=self._obj_function_parsec,
                           x0=p_init,
                           bounds=bounds,
                           options={'disp': disp})

        return res.x

    def _obj_function_parsec(self, p):
        """
        Computes the difference between self.y and the Y coordinates obtained with p
        :return: error
        """

        y_u_p, y_l_p = self.parsec_to_xy(p)
        y_parsec = np.concatenate([y_l_p, y_u_p])
        diff = np.subtract(y_parsec, self.y)
        diff_squared = np.square(diff)

        return np.sum(diff_squared)


def test_simple_parsec():
    """ Creates an airfoil from a P array. Xu, Xl automatically generated with cosine distrib. """
    a1 = Airfoil(p=np.array([0.012, 0.3, 0.08, 0, 0.3, 0.06, 0.0, 0.0211, 0.0, 0.0, 0.5]))
    a1.plot_airfoil()


def test_xy_to_parsec():
    """ Reads naca0012 coordinates and then finds best PARSEC array to fit it. """
    naca0012 = Airfoil(fpath='../data/Coordinates_naca0012')
    parsec_naca = Airfoil(x_u=naca0012.x_u,
                          x_l=naca0012.x_l,
                          p=naca0012.xy_to_parsec())
    parsec_naca.plot_airfoil(color='g')
    naca0012.plot_airfoil()


if __name__ == '__main__':
    test_simple_parsec()
