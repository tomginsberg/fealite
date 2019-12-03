from scipy import sparse
from scipy.sparse import lil_matrix
from typing import Tuple, Union

from mesh import TriangleMesh
from problem_definitions import PoissonProblemDefinition, NonLinearPoissonProblemDefinition
import poisson
import numpy as np


def distance(c1: np.ndarray, c2: np.ndarray) -> float:
    x1, y1 = c1
    x2, y2 = c2
    return np.linalg.norm([x1 - x2, y1 - y2])


def apply_dirichlet(mesh: TriangleMesh, k: lil_matrix, b: Union[lil_matrix, np.ndarray],
                    p: PoissonProblemDefinition.dirichlet_boundary):
    for v, marker in mesh.boundary_dict.items():
        boundary_value = p(marker, mesh.coordinates[v])
        if boundary_value is not None:
            for i in range(k.shape[0]):
                if len(b.shape) > 1:
                    b[i, 0] -= k[i, v] * boundary_value
                else:
                    b[i] -= k[i, v] * boundary_value
                k[i, v] = 0
            k[v] = 0
            k[v, v] = 1
            if len(b.shape) > 1:
                b[v, 0] = boundary_value
            else:
                b[v] = boundary_value


def apply_neumann(mesh: TriangleMesh, b: lil_matrix, q: PoissonProblemDefinition.neumann_boundary):
    for element, marker in zip(mesh.boundary_elements, mesh.boundary_markers):
        el_length = distance(*mesh.element_coords(element))
        for e_id in element:
            neumann_value = q(boundary_marker=marker, coordinate=mesh.coordinates[e_id])
            if neumann_value is not None:
                if len(b.shape) > 1:
                    b[e_id, 0] += el_length / 2 * neumann_value
                else:
                    b[e_id] += el_length / 2 * neumann_value


def assemble_global_stiffness_matrix(mesh: TriangleMesh,
                                     alpha: PoissonProblemDefinition.material) -> sparse.lil_matrix:
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


class NonLinearSystem:
    def __init__(self, mesh: TriangleMesh, alpha: NonLinearPoissonProblemDefinition.material,
                 p: NonLinearPoissonProblemDefinition.dirichlet_boundary,
                 q: NonLinearPoissonProblemDefinition.neumann_boundary,
                 b: np.ndarray):
        self.mesh = mesh
        self.alpha = alpha
        self.p = p
        self.q = q
        self.b = b

    def sys_eval(self, curr: np.array) -> np.ndarray:
        rows, cols, values = [], [], []
        b = self.b.copy()
        for element, shp_fn, marker in zip(self.mesh.mesh_elements, self.mesh.mesh_shape_functions,
                                           self.mesh.mesh_markers):
            _, _, alpha_field, _ = local_properties(curr, element, marker, shp_fn, self.alpha)
            n = len(element)
            for i in range(n):
                for j in range(i, n):
                    rows.append(element[i])
                    cols.append(element[j])
                    values.append(
                        shp_fn.stiffness_matrix[i, j] * alpha_field)
                    if i != j:
                        rows.append(element[j])
                        cols.append(element[i])
                        values.append(shp_fn.stiffness_matrix[j, i] * alpha_field)

        k = sparse.coo_matrix((values, (rows, cols))).tolil()
        apply_dirichlet(self.mesh, k, b, self.p)
        return k @ curr - b

    def jacobian(self, curr) -> np.ndarray:
        rows, cols, values = [], [], []
        for element, shp_fn, marker in zip(self.mesh.mesh_elements, self.mesh.mesh_shape_functions,
                                           self.mesh.mesh_markers):

            a_ijk, field2norm, alpha_field, grad_alpha = local_properties(curr, element, marker, shp_fn, self.alpha)
            # local_jacobian = alpha_field * np.matmul(np.matmul(shp_fn.stiffness_matrix, a_ijk.transpose()),
            #                                          grad_alpha) * np.dot(
            #     shp_fn.div_N_ijk__x + shp_fn.div_N_ijk__y, a_ijk.transpose()) + shp_fn.stiffness_matrix * alpha_field
            local_jacobian = \
                shp_fn.stiffness_matrix @ (alpha_field * np.identity(3) + a_ijk.transpose() @ grad_alpha)

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
        # test is this can be sparse
        j = np.array(sparse.coo_matrix((values, (rows, cols))).todense())
        for v, marker in self.mesh.boundary_dict.items():
            if self.p(marker, self.mesh.coordinates[v]) is not None:
                # zero all fixed terms in the jacobian
                j[marker, :] *= 0
                j[marker, marker] = 1
        return j


def local_properties(curr: np.ndarray, element: Tuple[int, int, int], marker, shp_fn,
                     alpha: NonLinearPoissonProblemDefinition.material) \
        -> Tuple[np.ndarray, float, float, np.ndarray]:
    a_ijk = np.array([[curr[i] for i in element]])
    # compute field approximation on this element
    field2norm = np.linalg.norm(
        a_ijk @ np.vstack((shp_fn.div_N_ijk__x, shp_fn.div_N_ijk__y)).transpose()
    )
    # compute non linear material property for this field strength
    alpha_field = alpha(marker, field2norm)
    # compute the gradient of the non linear material property for this field strength w respect to the node values
    grad_alpha = alpha(marker, field2norm, div=True) / field2norm * (
            a_ijk @ shp_fn.stiffness_matrix * 2 / shp_fn.double_area)

    return a_ijk, field2norm, alpha_field, grad_alpha


def assemble_global_vector(mesh: TriangleMesh, f: PoissonProblemDefinition.source, size: int) -> sparse.lil_matrix:
    rows, values = [], []
    for element, shp_fn, marker in zip(mesh.mesh_elements, mesh.mesh_shape_functions,
                                       mesh.mesh_markers):
        for v in element:
            source_value = f(marker, mesh.coordinates[v])
            if source_value is not None:
                rows.append(v)
                values.append(shp_fn.double_area / 6 * source_value)

    return sparse.coo_matrix((values, (rows, np.zeros_like(rows))), shape=(size, 1)).tolil()
