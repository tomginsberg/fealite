from abc import ABC
from math import pi
from typing import Union, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve

import matrix_assmbly
from mesh import Meshes, TriangleMesh


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

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError


def distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


class Poisson:
    """
    Assembles the FEA matrix for the general Poisson problem  - ∇·( alpha(material) ∇phi(x,y) ) = f(x,y; material)
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
        self.apply_neumann()
        self.K = self.K.tocsc()
        self.solution = spsolve(self.K, self.b)

    def apply_dirichlet(self):
        for v, marker in self.mesh.boundary_dict.items():
            boundary_value = self.p(marker, self.mesh.coordinates[v])
            if boundary_value is not None:
                for i in range(self.K.shape[0]):
                    self.b[i, 0] -= self.K[i, v] * boundary_value
                    self.K[i, v] = 0
                self.K[v] = 0
                self.K[v, v] = 1
                self.b[v, 0] = boundary_value

    def apply_neumann(self):
        for element, marker in zip(self.mesh.boundary_elements, self.mesh.boundary_markers):
            # This could also be made more accurate, but not that important
            el_length = distance(*[self.mesh.coordinates[i] for i in element])
            for e_id in element:
                neumann_value = self.q(boundary_marker=marker, coordinate=self.mesh.coordinates[e_id])
                if neumann_value is not None:
                    self.b[e_id] += el_length / 2 * neumann_value

    def export_solution(self):
        export_path = f'solutions/{self.mesh.short_name}_{self.name}.txt'
        with open(export_path, 'w') as f:
            f.write('\n'.join(['\t'.join([f'{c:f}' for c in cord]) + f'\t{z:f}' for cord, z in
                               zip(self.mesh.coordinates, self.solution)]))


class DielectricObjectInUniformField(PoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.cylinder_in_square, name: str = 'dielectric',
                 source_marker: int = 2, sink_marker: int = 4, dielectric_marker: int = 2):
        super().__init__(mesh, name)
        self.source_marker = source_marker
        self.sink_marker = sink_marker
        self.dielectric_marker = dielectric_marker

    def linear_material(self, element_marker: int) -> float:
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

    def linear_material(self, element_marker: int) -> float:
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

    def linear_material(self, element_marker: int) -> float:
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

    def linear_material(self, element_marker: int) -> float:
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

class Nonlinear(PoissonProblemDefinition):
    def __init__(self, mesh: Union[str, TriangleMesh] = Meshes.heart, name: str = 'nonlinear'):
        super().__init__(mesh, name)
        self.K = matrix_assmbly.assemble_global_stiffness_matrix(self.mesh, 1)
        self.b = matrix_assmbly.assemble_global_vector(self.mesh, PoissonProblemDefinition.source, self.K.shape[0])
        self.fxn_array = matrix_assmbly.assemble_global_stiffness_matrix_nonlinear(self.mesh,
                                                                  PoissonProblemDefinition.linear_material, self.b)
        self.x0 = spsolve(self.K, self.b) #give approximate solution using constant value for linear material parameter
        self.solution = fsolve(self.fxn_array, self.x0)


    def linear_material(self, element_marker: int) -> float: #each diff type of element marker takes fxn from diff text file in reluctances
        #assume we can fit each distribution for now
        #if element_marker = 1:
            #give certain distn back
        #else if element_marker = 2:    need to figure out representation of distn we will get
        pass

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        pass

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        pass

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        pass
    #need to understand how these parts actually work


if __name__ == '__main__':
    problem = Poisson(DielectricObjectInUniformField(Meshes.annulus))
    problem.export_solution()
