import numpy as np
from typing import Union

from mesh import TriangleMesh
from abc import ABC
from math import pi
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D


class PoissonProblemDefinition(ABC):
    EPS0 = 8.854e-2
    MU0 = 4e-7 * pi

    def __init__(self, mesh: Union[str, TriangleMesh], ):
        if type(mesh) is str:
            self.mesh = TriangleMesh(mesh)
        else:
            self.mesh = mesh

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
        self.mesh = definition.mesh
        self.alpha = definition.linear_material
        self.f = definition.source
        self.p = definition.dirichlet_boundary
        self.q = definition.neumann_boundary
        self.K = self.assemble_matrix()
        self.b = self.assemble_vector()
        self.apply_dirichlet()
        self.K = self.K.tocsc()
        self.solution = spsolve(self.K, self.b)

    def assemble_matrix(self):
        # take each stiffness matrix convert it into COOrdinate format
        rows, cols, values = [], [], []
        for element, shp_fn, marker in zip(self.mesh.mesh_elements, self.mesh.mesh_shape_functions,
                                           self.mesh.mesh_markers):
            for i in range(3):
                for j in range(i, 3):
                    rows.append(element[i])
                    cols.append(element[j])
                    values.append(shp_fn.stiffness_matrix[i, j] * self.alpha(marker))
                    if i != j:
                        rows.append(element[j])
                        cols.append(element[i])
                        values.append(shp_fn.stiffness_matrix[j, i] * self.alpha(marker))

        return sparse.coo_matrix((values, (rows, cols))).tolil()

    def assemble_vector(self):
        # TODO: Implement this
        return sparse.coo_matrix((self.K.shape[0], 1)).tolil()

    def apply_dirichlet(self):
        for v, marker in self.mesh.boundary_master.items():
            boundary_value = self.p(marker, self.mesh.coordinates[v])
            if boundary_value is not None:
                for i in range(self.K.shape[0]):
                    self.b[i, 0] -= self.K[i, v]
                    self.K[i, v] = 0
                self.K[v, :] *= 0
                self.K[v, v] = 1
                self.b[v, 0] = boundary_value

    def interpolating_function(self):
        return interp2d(*np.array(self.mesh.coordinates).transpose(), self.solution)


class DielectricCylinder(PoissonProblemDefinition):

    def linear_material(self, element_marker: int) -> float:
        if element_marker == 1:
            return self.EPS0
        return 2 * self.EPS0

    def source(self, element_marker: int, coordinate: np.ndarray) -> float:
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        if boundary_marker == 1:
            return 1
        elif boundary_marker == 2:
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
        return None

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Union[float, None]:
        return None


if __name__ == '__main__':
    # problem = Poisson(DielectricCylinder('meshes/sample-mesh1.tmh'))
    problem = Poisson(SampleProblem('meshes/unit-disk.tmh'))
    problem.interpolating_function()
