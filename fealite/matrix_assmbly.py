from scipy import sparse
from mesh import TriangleMesh
from poisson import PoissonProblemDefinition
import numpy as np


def assemble_global_stiffness_matrix(mesh: TriangleMesh,
                                     alpha: PoissonProblemDefinition.linear_material) -> sparse.lil_matrix:
    rows, cols, values = [], [], []
    for element, shp_fn, marker in zip(mesh.mesh_elements, mesh.mesh_shape_functions,
                                       mesh.mesh_markers):
        n = len(element)
        for i in range(n):
            for j in range(i, n):
                rows.append(element[i])
                cols.append(element[j])
                values.append(shp_fn.stiffness_matrix[i, j] * alpha(marker))
                if i != j:
                    rows.append(element[j])
                    cols.append(element[i])
                    values.append(shp_fn.stiffness_matrix[j, i] * alpha(marker))

    return sparse.coo_matrix((values, (rows, cols))).tolil()

def assemble_global_stiffness_matrix_nonlinear(mesh: TriangleMesh,
                                     alpha: PoissonProblemDefinition.linear_material, b: PoissonProblemDefinition.b):
    #want to return function of A = [A_1, A_2, ... A_n] where n is number of nodes
    #not stiffness matrix anymore, just the functions necessary for solving recursively
    rows, cols, values = [], [], []

    def fxn(A: np.array):

        for element, shp_fn, marker in zip(mesh.mesh_elements, mesh.mesh_shape_functions,
                                           mesh.mesh_markers):
            n = len(element)
            for i in range(n):
                for j in range(i, n):
                    rows.append(element[i])
                    cols.append(element[j])
                    values.append(shp_fn.stiffness_matrix[i, j] * alpha(marker)(A[j])) #alpha returns reluctance as fxn of A_j
                    if i != j:
                        rows.append(element[j])
                        cols.append(element[i])
                        values.append(shp_fn.stiffness_matrix[j, i] * alpha(marker))

        matrix_rep = sparse.coo_matrix((values, (rows, cols))).tolil()
        return np.subtract(np.matmul(matrix_rep, A), b)

    return fxn

def assemble_global_vector(mesh: TriangleMesh, f: PoissonProblemDefinition.source, size: int) -> sparse.lil_matrix:
    rows, values = [], []
    for element, shp_fn, marker in zip(mesh.mesh_elements, mesh.mesh_shape_functions,
                                       mesh.mesh_markers):
        for v in element:
            # TODO: A more exact method can be used here, but it's not that important
            source_value = f(marker, mesh.coordinates[v])
            if source_value is not None:
                rows.append(v)
                values.append(shp_fn.double_area / 6 * source_value)

    return sparse.coo_matrix((values, (rows, np.zeros_like(rows))), shape=(size, 1)).tolil()

