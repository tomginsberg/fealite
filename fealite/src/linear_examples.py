from typing import Union, Optional
import numpy as np
from mesh import TriangleMesh, Meshes
from poisson import PoissonProblemDefinition, LinearPoisson
from math import pi
from nonlinear_examples import MU0, square_wave


class DielectricObjectInUniformField(PoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.cylinder_in_square, name: str = 'dielectric',
                 source_marker: int = 2, sink_marker: int = 4, dielectric_marker: int = 2):
        super().__init__(mesh, name)
        self.source_marker = source_marker
        self.sink_marker = sink_marker
        self.dielectric_marker = dielectric_marker

    def material(self, element_marker: int) -> float:
        if element_marker == self.dielectric_marker:
            return 1
        return 5

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if boundary_marker == self.source_marker:
            return 1
        elif boundary_marker == self.sink_marker:
            return -1
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None


class SampleProblem(PoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.unit_disk, name: str = 'laplace'):
        super().__init__(mesh, name)

    def material(self, element_marker: int) -> float:
        return 1

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return coordinate[0] * coordinate[1]

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None


class InsulatingObject(PoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.cylinder_in_square, name: str = 'insulator'):
        super().__init__(mesh, name)

    def material(self, element_marker: int) -> float:
        return 1

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return 1 if element_marker == 2 else None

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return -np.log(np.linalg.norm(coordinate)) / 2 - 1 / 4 if np.linalg.norm(coordinate) > 2 else None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None


class DielectricHeart(DielectricObjectInUniformField):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.heart, name: str = 'dielectric'):
        super().__init__(mesh, name, source_marker=1, sink_marker=3)


class Airfoil(PoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.airfoil, name: str = 'euler-flow'):
        super().__init__(mesh, name)

    def material(self, element_marker: int) -> float:
        return 1

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if boundary_marker == 4:
            return 1
        if boundary_marker == 2:
            return -1
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        pass


class BLDC(PoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.bldc6):
        super().__init__(mesh, name='linearized')
        self.coil_current = 20 / (pi / 4 * (.001 ** 2))
        self.magnet_factor = 1.05

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if element_marker == 3 or element_marker == 8:
            return -self.coil_current
        if element_marker == 4 or element_marker == 7:
            return self.coil_current
        if element_marker == 1:
            x, y = coordinate
            norm = np.sqrt(x ** 2 + y ** 2)
            if 0.015 <= norm <= 0.023:
                return 6.25e7 * square_wave(np.arctan2(y, x) / (2 * pi) + 1 / 4)
            if 0.037 <= norm <= 0.045:
                return -6.25e7 * square_wave(np.arctan2(y, x) / (2 * pi) + 1 / 4)
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        if coordinate[0] ** 2 + coordinate[1] ** 2 > 0.009025:
            return 0
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        return None

    def material(self, element_marker) -> float:
        # Stator
        if element_marker == 2:
            return 1 / (100 * MU0)

        # Magnet
        if element_marker == 1:
            return 1 / (self.magnet_factor * MU0)

        # Air
        return 1 / MU0


if __name__ == '__main__':
    problem = LinearPoisson(BLDC())
    problem.solve_and_export()
