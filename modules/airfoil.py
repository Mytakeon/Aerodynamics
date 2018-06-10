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
    _init_kwargs = ('fpath', 'p', 'x_u', 'x_l', 'naca_string', 'n_points')
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

        self.tol = 1e-4
        self.kwargs = kwargs  # Saving, only for the title of the plot
        log.debug("Airfoil%s" % str(locals()))
        keys = list(kwargs.keys())

        for key, value in kwargs.items():
            if key in self._init_kwargs:
                setattr(self, key, value)  # Given input
            else:
                setattr(object=self, name=key, value=None)  # Unkown input, set as None

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

        if 'p' in keys:  # PARSEC definition has been given as an input
            self.p = kwargs['p']
            self.unpack_parsec()
            self.y_l, self.y_u = self.parsec_to_y(self.p)

        if 'naca_string' in keys:
            self.naca_string = kwargs['naca_string']
            self.naca_handler()

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
        b_up = np.array([p[7] + p[8] / 2, p[2], math.tan(p[9] - p[10] / 2), 0, p[3], math.sqrt(2 * p[0])])
        b_low = np.array([-p[7] + p[8] / 2, p[5], math.tan(p[9] - p[10] / 2), 0, p[6], math.sqrt(2 * p[0])])

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


def CST(x, c, deltasz=None, Au=None, Al=None):
    """
    Based on the paper "Fundamental" Parametric Geometry Representations for
    Aircraft Component Shapes" from Brenda M. Kulfan and John E. Bussoletti. 
    The code uses a 1st order Bernstein Polynomial for the "Class Function" / 
    "Shape Function" airfoil representation.
    
    The degree of polynomial is dependant on how many values there are for the
    coefficients. The algorithm is able to use Bernstein Polynomials for any
    order automatically. The algorithm is also able to analyze only the top or
    lower surface if desired. It will recognize by the inputs given. i.e.:
    for CST(x=.2,c=1.,deltasx=.2,Au=.7), there is only one value for Au, so it
    is a Bernstein polynomial of 1st order for the upper surface. By ommiting 
    Al the code will only input and return the upper surface.
    
    Although the code is flexible, the inputs need to be coesive. len(deltasz)
    must be equal to the number of surfaces. Au and Al need to have the same 
    length if a full analysis is being realized.    
    
    The inputs are:

        - x:list or numpy. array of points along the chord, from TE and the LE,
          or vice-versa. The code works both ways.
          
        - c: chord
    
        - deltasz: list of thicknesses on the TE. In case the upper and lower
          surface are being analyzed, the first element in the list is related 
          to the upper surface and the second to the lower surface. There are 
          two because the CST method treats the airfoil surfaces as two 
          different surfaces (upper and lower)
        
        - Au: list/float of Au coefficients, which are design parameters. If 
          None,the surface is not analyzed. len(Au) equals  
        
        - Al: list/float of Al coefficients, which are design parameters. If 
          None,the surface is not analyzed. len(Al) equals  
         
    The outputs are:
        - y:
          - for a full analysis: disctionary with keys 'u' and 'l' each with 
            a list of the y positions for a surface.
          - for a half analysis: a list with the list of the y postions of the
            the desired surface
        
    Created on Sun Jan 19 16:36:55 2014
    
    Updated on Mon May 19 18:13:26 2014

    @author: Pedro Leal
    """

    # Bersntein Polynomial
    def K(r, n):
        K = math.factorial(n) / (math.factorial(r) * math.factorial(n - r))
        return K

    # Shape Function   
    def S(r, n, psi):
        S = K(r, n) * (psi ** r) * (1. - psi) ** (n - r)
        return S

    # Class Function    
    def C(N1, N2, psi):
        C = ((psi) ** N1) * ((1. - psi) ** N2)
        return C

    if type(x) == list:
        x = np.array(x)
    # Adimensionalizing
    psi = x / c;

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                           Class Function
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The Coefficients for an airfoil with a rounded leading edge and a sharp
    # trailing edge are N1=0.5 and N2=1.0.
    N1 = 0.5;
    N2 = 1.0;
    C = C(N1, N2, psi);

    # ==========================================================================
    #                   Defining the working surfaces
    # ==========================================================================
    deltaz = {}
    eta = {}
    y = {}
    Shape = {}

    if Al and Au:
        deltaz['u'] = deltasz[0]
        deltaz['l'] = deltasz[1]

        if len(Au) != len(Al):
            raise Exception("Au and Al need to have the same dimensions")
        elif len(deltasz) != 2:
            raise Exception("If both surfaces are being analyzed, two values for deltasz are needed")

    elif Au and not Al:
        if type(deltasz) == list:
            if len(deltaz['u']) != 1:
                raise Exception("If only one surface is being analyzed, one value for deltasz is needed")
            else:
                deltaz['u'] = float(deltasz)
        else:
            deltaz['u'] = deltasz

    elif Al and not Au:
        if type(deltasz) == list:
            if (deltaz['l']) != 1:
                raise Exception("If only one surface is being analyzed, one value for deltasz is needed")
            else:
                deltaz['l'] = float(deltasz)
        else:
            deltaz['l'] = deltasz
    else:
        raise Exception("Au or Al need to have at least one value")
    A = {'u': Au, 'l': Al}
    for surface in ['u', 'l']:
        if A[surface]:

            if type(A[surface]) == int or type(A[surface]) == float:
                A[surface] = [A[surface]]
            # the degree of the Bernstein polynomial is given by the number of
            # coefficients
            n = len(A[surface]) - 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #                           Shape Function
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            Shape[surface] = 0
            for i in range(len(A[surface])):
                #                print A
                #                print S(i,n,psi)
                if surface == 'l':
                    Shape[surface] -= A[surface][i] * S(i, n, psi)
                else:
                    Shape[surface] += A[surface][i] * S(i, n, psi)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    #                           Airfoil Shape (eta=z/c)
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Airfoil Shape (eta=z/c)
            if surface == 'l':
                eta[surface] = C * Shape[surface] - psi * deltaz[surface] / c;
            else:
                eta[surface] = C * Shape[surface] + psi * deltaz[surface] / c;
                # Giving back the dimensions
            y[surface] = c * eta[surface]
    if Al and Au:
        return y
    elif Au:
        return y['u']
    else:
        return y['l']


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n2412_1 = Airfoil(naca_string='NACA2412')
    n2412_1.plot(plt)
