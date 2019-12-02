from scipy import sparse
from typing import Tuple, List

from mesh import TriangleMesh
from poisson import PoissonProblemDefinition, NonLinearPoissonProblemDefinition
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
                                               alpha: NonLinearPoissonProblemDefinition.non_linear_material,
                                               b: sparse.lil_matrix, curr: np.array):
    # want to return function of A = [A_1, A_2, ... A_n] where n is number of nodes
    # not stiffness matrix anymore, just the functions necessary for solving recursively
    rows, cols, values = [], [], []

    def fxn(a: np.array):

        for element, shp_fn, marker in zip(mesh.mesh_elements, mesh.mesh_shape_functions,
                                           mesh.mesh_markers):
            local_props = local_properties(curr, element, marker, shp_fn, alpha)
            n = len(element)
            for i in range(n):
                for j in range(i, n):
                    rows.append(element[i])
                    cols.append(element[j])
                    values.append(
                        shp_fn.stiffness_matrix[i, j] * local_props[2])
                    if i != j:
                        rows.append(element[j])
                        cols.append(element[i])
                        values.append(shp_fn.stiffness_matrix[j, i] * local_props[2])

        matrix_rep = sparse.coo_matrix((values, (rows, cols))).tolil()
        return np.subtract(np.matmul(matrix_rep, a), b)

    return fxn


def local_properties(curr: np.ndarray, element: Tuple[int, int, int], marker, shp_fn,
                     alpha: NonLinearPoissonProblemDefinition.non_linear_material) -> Tuple[np.ndarray, float, float,
                                                                                            np.ndarray]:
    a_ijk = np.array([[curr[i] for i in element]])
    # compute field approximation on this element
    field2norm = np.linalg.norm([np.dot(shp_fn.div_N_ijk__x, a_ijk), np.dot(shp_fn.div_N_ijk__x, a_ijk)])
    # compute non linear material property for this field strength
    alpha_field = alpha(marker, field2norm)
    # compute the gradient of the non linear material property for this field strength w respect to the node values
    grad_alpha = alpha(marker, field2norm, div=True) * (
            (shp_fn.div_N_ijk__x + shp_fn.div_N_ijk__y) * np.dot((shp_fn.div_N_ijk__x + shp_fn.div_N_ijk__y),
                                                                 a_ijk.transpose())) / (
                             (shp_fn.double_area ** 2) * field2norm)
    return a_ijk, field2norm, alpha_field, grad_alpha


def assemble_jacobian(mesh: TriangleMesh, alpha: NonLinearPoissonProblemDefinition.non_linear_material,
                      curr: np.ndarray):
    rows, cols, values = [], [], []
    for element, shp_fn, marker in zip(mesh.mesh_elements, mesh.mesh_shape_functions,
                                       mesh.mesh_markers):

        a_ijk, field2norm, alpha_field, grad_alpha = local_properties(curr, element, marker, shp_fn, alpha)
        local_jacobian = alpha_field * np.matmul(np.matmul(shp_fn.stiffness_matrix, a_ijk.transpose()),
                                                 grad_alpha) * np.dot(
            shp_fn.div_N_ijk__x + shp_fn.div_N_ijk__y, a_ijk.transpose()) + shp_fn.stiffness_matrix * alpha_field

        n = len(element)
        for i in range(n):
            for j in range(i, n):
                rows.append(element[i])
                cols.append(element[j])
                values.append(local_jacobian[i, j])
                if i != j:
                    rows.append(element[j])
                    cols.append(element[i])
                    values.append(local_jacobian[j, i])

    return sparse.coo_matrix((values, (rows, cols))).tolil()


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
