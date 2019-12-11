from typing import Union, Optional
import numpy as np
from mesh import TriangleMesh, Meshes
from poisson import NonLinearPoissonProblemDefinition, NonLinearPoisson
from math import e, pi

# import sys
#
# sys.path.append("../")

EPS0 = 8.854e-12
MU0 = 4e-7 * pi


def square_wave(x: float) -> int:
    if 1 / 2 <= x % 1 < 1:
        return -1
    return 1


def nu(x: float, div: bool) -> float:
    if div:
        return -(((-0.04416689710046033 * e ** (
                1.4181115724710933 * (-0.7754925571416307 + x) ** 2) * (
                           -0.7754925571416307 + x)) /
                  - (1. + e ** (1.4181115724710933 * (-0.7754925571416307 + x) ** 2)) ** 3 +
                  - (0.04263697112349125 * (0.04885906655766546 + x)) / e ** (
                          42.077372984456105 * (0.04885906655766546 + x) ** 2)) /
                 - (0.0005066496325619219 / e ** (
                         42.077372984456105 * (0.04885906655766546 + x) ** 2) -
                    - 0.007786216888333133 / (1. + e ** (
                                 1.4181115724710933 * (-0.7754925571416307 + x) ** 2)) ** 2) ** 2)

    return 1 / (-0.0005066496325619219 / e ** (
            42.077372984456105 * (0.04885906655766546 + x) ** 2) + 0.007786216888333133 / (
                        1. + e ** (1.4181115724710933 * (-0.7754925571416307 + x) ** 2)) ** 2)


class TestProblem(NonLinearPoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.unit_disk, name: str = 'nonlinear'):
        super().__init__(mesh, name)

    def material(self, element_marker: int, norm_grad_phi: Optional[float] = None,
                 div: bool = False) -> float:
        if norm_grad_phi is None:
            return 1.
        if div:
            return 24 * np.exp(2 * norm_grad_phi) + 1 / (norm_grad_phi + .5)
        return 12 * np.exp(2 * norm_grad_phi) + np.log(norm_grad_phi + .5)

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return np.log(coordinate[0] ** 2 + coordinate[1] ** 2 + .1)

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return 0

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None


class MathematicaDemo(NonLinearPoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.bldc_old):
        super().__init__(mesh)

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if element_marker == 4:
            return 10
        if element_marker == 6:
            return -10
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if coordinate[0] ** 2 + coordinate[1] ** 2 > .95:
            return 0
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None

    def material(self, element_marker: int, norm_grad_phi: Optional[float] = None, div: bool = False) -> float:
        if element_marker == 2 or element_marker == 3:
            if norm_grad_phi is None:
                norm_grad_phi = 0
            return nu(norm_grad_phi, div)
        if div:
            return 0
        return 1 / MU0


class BLDC(NonLinearPoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.bldc):
        super().__init__(mesh)
        self.coil_current = 3
        self.magnet_factor = 30

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if element_marker == 4:
            return self.coil_current
        if element_marker == 5:
            return -self.coil_current
        if element_marker == 6:
            return self.coil_current
        if element_marker == 7:
            return -self.coil_current
        if element_marker == 2:
            x, y = coordinate
            (10 * (-0.3125 + np.sqrt(x ** 2 + y ** 2)) * square_wave(np.arctan2(x, y) / (2 * pi) - 1 / 4))
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if coordinate[0] ** 2 + coordinate[1] ** 2 > .95:
            return 0
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None

    def material(self, element_marker: int, norm_grad_phi: Optional[float] = None, div: bool = False) -> float:
        # Stator
        if element_marker == 3:
            if norm_grad_phi is None:
                norm_grad_phi = 0
            return nu(norm_grad_phi, div)

        if div:
            return 0

        # Magnet
        if element_marker == 2:
            return 1 / (self.magnet_factor * MU0)

        # Air
        return 1 / MU0


def fit_bh_curve():
    import autograd.numpy as anp
    from autograd import elementwise_grad

    HField = np.array([10, 25, 50, 75, 108, 137, 160, 180, 200, 235, 285, 320, 365,
                       480, 560, 660, 800, 1000, 1150, 1550, 1840, 2250, 2800, 3450,
                       4350, 5500, 7707, 11863, 15839, 21076, 27281, 35289, 45642, 59046,
                       76413, 98907, 128000, 134764, 141880, 149364, 157235])
    BField = np.array([0.0025, 0.0066, 0.0151, 0.0264, 0.05, 0.1, 0.15, 0.2, 0.25,
                       0.35, 0.5, 0.6, 0.7, 0.9, 1., 1.1, 1.2, 1.3, 1.354, 1.45, 1.5,
                       1.55, 1.6, 1.65, 1.7, 1.75, 1.81, 1.89, 1.945, 2., 2.05, 2.1, 2.15,
                       2.2, 2.25, 2.3, 2.35, 2.36, 2.37, 2.38, 2.39])
    xdata, ydata = BField, BField / HField

    def gaussian_model(x, a, b, c, d, e, f):
        return a * anp.exp(-((x - b) / c) ** 2) + d / ((1 + anp.exp(-((x - e) / f) ** 2)) ** 2)

    def curried_model(params):
        def func(x):
            return gaussian_model(x, *params)

        return func

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    plt.plot(xdata, ydata, 'b-', label='data')
    popt, pcov = curve_fit(gaussian_model, xdata, ydata)
    xdata = np.linspace(xdata[0], xdata[-1])
    print(popt)
    plt.plot(xdata, gaussian_model(xdata, *popt), 'r-', label='fit')
    # plt.plot(xdata, elementwise_grad(curried_model(popt))(xdata), 'g-', label='grad')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # fit_bh_curve()
    problem = NonLinearPoisson(BLDC())
    problem.solve_and_export()
