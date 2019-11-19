import numpy as np
from typing import Union

from mesh import TriangleMesh
from abc import ABC
from math import pi
from scipy.sparse.linalg import spsolve
import matrix_assmbly


class PoissonProblemDefinition(ABC):
    EPS0 = 8.854e-12
    MU0 = 4e-7 * pi

    def __init__(self, mesh: Union[str, TriangleMesh], name: str):
        if type(mesh) is str:
            self.mesh = TriangleMesh(mesh)
        else:
            self.mesh = mesh
        self.name = name

    def linear_material(self, element_marker: int) -> float:
        raise NotImplementedError

    def source(self, element_marker: int, coordinate: np.ndarray) -> float:
        raise NotImplementedError

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        raise NotImplementedError

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        raise NotImplementedError


class Poisson:
    """
    Assembles the FEA matrix for the general Poisson problem -∇ · ( alpha(material) ∇phi(x,y) ) = f(x,y; material)
    alpha is a spatial function only of the material properties (i.e ε)
    f is a source function of a material and coordinate (i.e ρ)
    """

    def __init__(self, definition: PoissonProblemDefinition):
        self.name = definition.name
        self.mesh = definition.mesh
        self.alpha = definition.linear_material
        self.f = definition.source
        self.p = definition.dirichlet_boundary
        self.q = definition.neumann_boundary
        self.K = matrix_assmbly.assemble_global_stiffness_matrix(self.mesh,
                                                                 self.alpha)
        self.b = matrix_assmbly.assemble_global_vector(self.mesh, self.f, self.K.shape[0])
        self.apply_dirichlet()
        # self.apply_neumann()
        self.K = self.K.tocsc()
        self.solution = spsolve(self.K, self.b)

    def apply_dirichlet(self):
        for v, marker in self.mesh.boundary_master.items():
            boundary_value = self.p(marker, self.mesh.coordinates[v])
            if boundary_value is not None:
                for i in range(self.K.shape[0]):
                    self.b[i, 0] -= self.K[i, v] * boundary_value
                    self.K[i, v] = 0
                self.K[v] = 0
                self.K[v, v] = 1
                self.b[v, 0] = boundary_value

    def apply_neumann(self):
        raise NotImplementedError

    def export_solution(self):
        export_path = f'solutions/{self.mesh.short_name}_{self.name}.txt'
        with open(export_path, 'w') as f:
            f.write('{' + ','.join(['{' + ','.join([f'{c:f}' for c in cord]) + f',{z:f}}}' for cord, z in
                                    zip(self.mesh.coordinates, self.solution)]) + '}')


class DielectricCylinder(PoissonProblemDefinition):

    def linear_material(self, element_marker: int) -> float:
        if element_marker == 1:
            return self.EPS0
        return 5 * self.EPS0

    def source(self, element_marker: int, coordinate: np.ndarray) -> float:
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        if boundary_marker == 1:
            return 1
        elif boundary_marker == 3:
            return -1
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        return None


class SampleProblem(PoissonProblemDefinition):

    def linear_material(self, element_marker: int) -> float:
        return 1

    def source(self, element_marker: int, coordinate: np.ndarray) -> float:
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        return coordinate[0] * coordinate[1]

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        return None


class InsulatingObject(PoissonProblemDefinition):

    def linear_material(self, element_marker: int) -> float:
        return self.EPS0

    def source(self, element_marker: int, coordinate: np.ndarray) -> float:
        return 1 if element_marker == 2 else 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        return 1 if np.linalg.norm(coordinate) < 1.3 else None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        return None


if __name__ == '__main__':
    # problem = Poisson(DielectricCylinder('meshes/sample-mesh1.tmh'))
    # problem = Poisson(DielectricCylinder('meshes/cylinder-in-square-fine.tmh', 'dielectric'))
    problem = Poisson(DielectricCylinder('meshes/heart.tmh', 'dielectric'))
    problem.export_solution()
