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
        return -((0.0000657508 - 0.0000385014 * x + 0.000830739 * x ** 2 - 0.00163689 * x ** 3 + 0.00104806 * x ** 4
                  - 0.000228563 * x ** 5) / (0.0000379551 + 0.0000657508 * x - 0.0000192507 * x ** 2
                                             + 0.000276913 * x ** 3 - 0.000409223 * x ** 4 + 0.000209611 * x ** 5
                                             - 0.0000380938 * x ** 6) ** 2)

    return 1 / (
            0.000037955129560483474 + 0.00006575078254916768 * x - 0.000019250684240476186 * x ** 2 +
            0.0002769129642720887 * x ** 3 - 0.00040922344805632805 * x ** 4 + 0.0002096110387525365 * x ** 5
            - 0.00003809378487585398 * x ** 6)


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
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.bldc6):
        super().__init__(mesh)
        self.coil_current = 20 / (pi / 4 * (1e-3 ** 2))
        self.magnet_factor = 1.05

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if element_marker == 3 or element_marker == 8:
            return -self.coil_current
        if element_marker == 4 or element_marker == 7:
            return self.coil_current
        if element_marker == 2:
            x, y = coordinate
            norm = np.sqrt(x ** 2 + y ** 2)
            if 0.015 <= norm <= 0.023:
                return 6.25e7 * square_wave(5 * np.arctan2(x, y) / (2 * pi) + 1 / 4)
            if 0.037 <= norm <= 0.045:
                return -6.25e7 * square_wave(5 * np.arctan2(x, y) / (2 * pi) + 1 / 4)
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if coordinate[0] ** 2 + coordinate[1] ** 2 > 0.009025:
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


if __name__ == '__main__':
    problem = NonLinearPoisson(BLDC())
    problem.solve_and_export()
