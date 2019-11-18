import numpy as np
from mesh import TriangleMesh
from abc import ABC
from math import pi
from shapefunction import LinearShapeFunction
from scipy import sparse
import matplotlib.pyplot as plt


class PoissonProblemDefinition(ABC):
    EPS0 = 8.854e-12
    MU0 = 4e-7 * pi

    def linear_material(self, element_marker: int):
        raise NotImplementedError

    def source(self, element_marker: int, coordinate: np.ndarray):
        raise NotImplementedError

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray):
        raise NotImplementedError

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray):
        raise NotImplementedError


class Poisson:
    """
    Assembles the FEA matrix for the general Poisson problem -∇ · ( alpha(material) ∇phi(x,y) ) = f(x,y; material)
    alpha is a spatial function only of the material properties (i.e ε)
    f is a source function of a material and coordinate (i.e ρ)
    """

    def __init__(self, mesh: TriangleMesh, definition: PoissonProblemDefinition):
        self.mesh = mesh
        self.alpha = definition.linear_material
        self.f = definition.source
        self.p = definition.dirichlet_boundary
        self.q = definition.neumann_boundary
        self.K = self.assemble_matrix()

    def assemble_matrix(self):
        # take each stiffness matrix convert it into COOrdinate format
        rows, cols, values = [], [], []
        for element, shp_fn, marker in zip(self.mesh.mesh_elements, self.mesh.mesh_shape_functions,
                                           self.mesh.mesh_markers):
            for i, v1 in enumerate(element):
                for j, v2 in enumerate(element):
                    rows.append(v1)
                    cols.append(v2)
                    values.append(shp_fn.stiffness_matrix[i, j] * self.alpha(marker))
        return sparse.coo_matrix((values, (rows, cols)))

    def assemble_vector(self):
        pass

class DielectricCylinder(PoissonProblemDefinition):

    def linear_material(self, element_marker: int) -> float:
        if element_marker == 1:
            return self.EPS0
        return 2 * self.EPS0

    def source(self, element_marker: int, coordinate: np.ndarray) -> float:
        return 0

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> float:
        return 0

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> float:
        return 0


if __name__ == '__main__':
    mesh = TriangleMesh(file_name='meshes/simple-mesh.tmh')
    problem = Poisson(mesh, DielectricCylinder())
    for shp, el in zip(problem.mesh.mesh_shape_functions, problem.mesh.mesh_elements):
        print(shp.stiffness_matrix)
        print(el)
    # plt.matshow(problem.K.todense(), cmap='Greys')
    # plt.colorbar()
    # plt.show()
