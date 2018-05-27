import matplotlib.pyplot as plt

from modules.xfoil import find_pressure_coefficients

if __name__ == '__main__':
    Cp = find_pressure_coefficients(airfoil='naca0012',
                                    alpha=0.0,
                                    Reynolds=0)

    # print(Cp['x'])
    # print(Cp['Cp'])
    #
    # plt.plot(Cp['x'], Cp['Cp'])
    # plt.xlabel('x-coordinate')
    # plt.ylabel('Cp')
    # plt.grid()
    # plt.show()
