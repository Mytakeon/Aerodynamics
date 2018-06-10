# -*- coding: utf-8 -*-
"""
Airfoil class. See examples for possible uses.
"""

import logging
import math

import numpy as np
import scipy.optimize as opt

from settings import FPATH_OUT

log = logging.getLogger(__name__)


class Airfoil:
    """
    Class that aims at making the translation of coordinates into various systems (CST, XY, PARSEC) seamless.
    """

    # List of possible input arguments
    _init_kwargs = ('fpath', 'parsec', 'x_u', 'x_l', 'naca_string', 'n_points')
    n_points_default = 100
    closed_te = False  # If false, TE of NACA4 series has small thickness

    def __init__(self, **kwargs):
        """

        :param fpath: File path of the XY coordinates
        :param p: PARSEC array description
        :param x_u: Upper panel points distribution
        :type x_u: list
        :param x_l: Lower panel points distribution
        :type x_l: list
        :param naca_string: NACA string designation, e.g. NACA2412
        :type naca_string: str
        :param n_points: number of points per surface
        :type n_points: int
        """

        self._p = None
        self.tol = 1e-4
        self.kwargs = kwargs  # Saving, only for the title of the plot
        log.debug("Airfoil%s" % str(locals()))
        keys = list(kwargs.keys())

        for arg in self._init_kwargs:
            if arg in kwargs:
                setattr(self, arg, kwargs[arg])  # Given input
            else:
                setattr(self, arg, None)  # Unkown input, set as None

        # Default number of points
        self.n_points = kwargs['n_points'] if 'n_points' in keys else self.n_points_default

        # Only assign default x_l and x_u if they are not defined as arguments; nor in a XY file
        if 'fpath' not in keys:
            if 'x_l' not in keys:
                self.x_l = -0.5 * np.cos(np.linspace(math.pi, 0, self.n_points)) + 0.5
            if 'x_u' not in keys:
                self.x_u = -0.5 * np.cos(np.linspace(0, math.pi, self.n_points)) + 0.5

        else:  # XY coordinate file has been given as an input
            self.read_xy_coords()

        if 'parsec' in keys:  # PARSEC definition has been given as an input
            self._p = kwargs['parsec']
            self.unpack_parsec()
            self.y_l, self.y_u = self.parsec_to_y(self.p)

        if 'naca_string' in keys:
            self.naca_string = kwargs['naca_string']
            self.naca_handler()

    @property
    def p(self):
        if self._p is None:
            self._p = self.xy_to_parsec()
            self.unpack_parsec()
        return self._p

    def __repr__(self):
        """
        Representation of an Airfoil, is returned if Airfoil object is printed
        """
        return 'Airfoil(%s)' % self.kwargs

    def __eq__(self, other):
        """
        An airfoil is 'equal' to another one if their their y-coordinates are very close.
        """
        e = self._compute_delta_y(other)
        if e:
            if e < self.tol:
                return True
            else:
                return False
        else:
            return False

    def _compute_delta_y(self, other):
        """

        :param other: Another airfoil to compare
        :type other: Airfoil
        :return: the sum of the squared difference between the two airfoil coordinates
        :rtype: float
        """
        if isinstance(other, Airfoil):
            if np.allclose(self.x, other.x, atol=1e-3):  # Now we can compare the y values
                delta_y = np.subtract(self.y, other.y)
                e = np.sum(np.square(delta_y))
                return e

            else:
                log.error('If the x vectors are not the same, problem is more complicated')
                return self.tol + 1

    @property
    def x(self):
        """
        Useful to make it a property because can be defined at various places in code.
        No need to unvalidate at any time.
        :return: np.ndarray
        """

        return np.concatenate([self.x_l, self.x_u])

    @property
    def y(self):
        """
        Useful to make it a property because can be defined at various places in code.
        No need to unvalidate at any time.
        :return: np.ndarray
        """

        return np.concatenate([self.y_l, self.y_u])

    def unpack_parsec(self):
        """
        Unpacking values in p array
        p = [R_le, X_up, Z_up, ZXXup, X_low, Z_low, ZXXlow, Y_TE, t_TE, alpha_TE, beta_TE]
        """

        self.r_le = self.p[0]
        self.x_up, self.z_up, self.zxx_up = self.p[1:4]
        self.x_low, self.z_low, self.zxx_low = self.p[4:7]
        self.y_te, self.t_te, self.alpha_te, self.beta_te = self.p[7:]

    def read_xy_coords(self):
        """
        Reads Xu, Xl, Yu, Yl from a standard airfoil file
        """
        with open(self.fpath, 'r') as datafile:
            _x = []
            _y = []
            for line in datafile:
                x, y = line.split(' ', 1)
                _x.append(float(x))
                _y.append(float(y))

        x = np.array(_x)
        y = np.array(_y)

        # Todo: need to make it clearer that I need the same number of points for both surfaces to work!!
        if len(x) % 2 == 0:
            self.x_u, self.x_l = np.split(x, 2)
            self.y_u, self.y_l = np.split(y, 2)
        else:
            raise NotImplementedError('Case of uneven number of point not implemented')

    def plot(self, plt, color='k', title=None):
        """ Plots the airfoil. """

        if not title:
            title = self.__repr__()

        plt.plot(self.x, self.y, color)
        plt.axis('equal')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def parsec_to_y(self, p):
        """
        Solve the linear system to get PARSEC y coordinates

        Source:
        "An airfoil shape optimization technique coupling PARSEC parameterization and evolutionary algorithm"

        :return:
        """

        # Define matrix
        def c(k):
            r1 = np.array([1, 1, 1, 1, 1, 1])
    
            r2 = np.array([p[k] ** (1 / 2),
                           p[k] ** (3 / 2),
                           p[k] ** (5 / 2),
                           p[k] ** (7 / 2),
                           p[k] ** (9 / 2),
                           p[k] ** (11 / 2)])
    
            r3 = np.array([1 / 2, 3 / 2, 5 / 2, 7 / 2, 9 / 2, 11 / 2])
    
            r4 = np.array([(1 / 2) * p[k] ** (-1 / 2),
                           (3 / 2) * p[k] ** (1 / 2),
                           (5 / 2) * p[k] ** (3 / 2),
                           (7 / 2) * p[k] ** (5 / 2),
                           (9 / 2) * p[k] ** (7 / 2),
                           (11 / 2) * p[k] ** (9 / 2)])
    
            r5 = np.array([(-1 / 4) * p[k] ** (-3 / 2),
                           (3 / 4) * p[k] ** (-1 / 2),
                           (15 / 4) * p[k] ** (1 / 2),
                           (35 / 4) * p[k] ** (3 / 2),  # Todo: 1/2 instead of 3/2, looks like mistake in article?
                           (63 / 4) * p[k] ** (5 / 2),
                           (99 / 4) * p[k] ** (7 / 2)])
    
            r6 = np.array([1, 0, 0, 0, 0, 0])
    
            return np.array([r1, r2, r3, r4, r5, r6])

        c_up = c(1)
        c_low = c(4)

        # todo: unsure why use radians instead of just tan here. Just following old matlab script for now.
        b_up = np.array([p[7] + p[8] / 2,
                         p[2],
                         math.tan(p[9] - p[10] / 2),
                         0,
                         p[3],
                         math.sqrt(2 * p[0])])

        b_low = np.array([-p[7] + p[8] / 2,
                          p[5],
                          math.tan(p[9] - p[10] / 2),
                          0,
                          p[3],
                          math.sqrt(2 * p[0])])

        a_up = np.linalg.solve(c_up, b_up)
        a_low = np.linalg.solve(c_low, b_low)
        a = np.concatenate([a_up, a_low])

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
        # p_init = np.array([0.032, 0.3, 0.08, 0, 1, 0.06, 0.0, 0.03, 0.0, 0.0, 0])
        bounds = ((0.0, 0.5),  # r_le
                  (0.0, 1.0),  # x_up
                  (0.0, 0.2),  # z_up
                  (-10, +10),  # zxx_up
                  (0.0, 1.0),  # x_low
                  (0.0, 0.3),  # z_low
                  (-10, 10),  # zxx_low
                  (0.0, 0.1),  # y_te
                  (0.0, 0.1),  # t_te
                  (-math.pi / 4, math.pi / 4),  # alpha_te
                  (-math.pi / 4, math.pi / 4))  # beta_te

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

        y_u_p, y_l_p = self.parsec_to_y(p)
        y_parsec = np.concatenate([y_l_p, y_u_p])
        diff = np.subtract(y_parsec, self.y)
        diff_squared = np.square(diff)

        return np.sum(diff_squared)

    def _naca4_to_xy(self, m, p, t):
        """
        See NACA equations: https://en.wikipedia.org/wiki/NACA_airfoil
        """

        x = self.x_u
        if self.closed_te:
            thickness = t / 0.20 * (0.29690 * np.sqrt(x) - 0.12600 * x - 0.35160 *
                                    np.power(x, 2) + 0.28430 * np.power(x, 3) - 0.10360 *
                                    np.power(x, 4))
        else:
            thickness = t / 0.20 * (0.29690 * np.sqrt(x) - 0.12600 * x - 0.35160 *
                                    np.power(x, 2) + 0.28430 * np.power(x, 3) - 0.10150 *
                                    np.power(x, 4))

        fwd_x = x[x < p]
        aft_x = x[x >= p]

        if 0 < p < 1 and 0 < m < 1:
            fwd_camber = m / p ** 2 * (2 * p * fwd_x - np.power(fwd_x, 2))
            aft_camber = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * aft_x -
                                             np.power(aft_x, 2))
            camber = np.append(fwd_camber, aft_camber)
        else:
            camber = np.zeros(np.size(x))

        y_low = np.add(camber, thickness)
        y_up = np.subtract(camber, thickness)
        self.y_u = y_up
        self.y_l = np.flipud(y_low)

    def naca_handler(self):
        """
        Depending on naca_string, formats and calls the appropriate function.
        """

        if 'NACA' in self.naca_string:
            code = self.naca_string.replace('NACA', '')
        else:
            code = self.naca_string

        if len(code) is 4:  # 4 digits naca
            m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
            self._naca4_to_xy(m, p, t)

        if len(code) is 5:
            return NotImplementedError('NACA5  not supported yet')
        else:
            return AttributeError('NACA string definition %s not recognized' % self.naca_string)

    def write_xy(self, name=None, fpath=FPATH_OUT):
        """
        Save the X,Y cooridnates of the airfoil into a .dat file
        :param name: name of the file
        :param fpath: output directory path
        :return: None
        """

        coord = np.column_stack((self.x, self.y))

        if not name:
            name = self.naca_string

        fpath = fpath + '/' + name + ".dat"
        np.savetxt(fpath, coord, delimiter=' ', fmt='%f')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n2412_1 = Airfoil(naca_string='NACA2412')
    n2412_1.plot(plt)
