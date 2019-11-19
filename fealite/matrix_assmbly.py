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


def assemble_global_vector(mesh: TriangleMesh, f: PoissonProblemDefinition.source, size: int) -> sparse.lil_matrix:
    rows, values = [], []
    for element, shp_fn, marker, coord in zip(mesh.mesh_elements, mesh.mesh_shape_functions,
                                              mesh.mesh_markers, mesh.coordinates):
        n = len(element)
        for i in range(n):
            source_value = f(marker, coord)
            if source_value is not None:
                rows.append(element[i])
                values.append(shp_fn.double_area / 6 * source_value)

    return sparse.coo_matrix((values, (rows, np.zeros_like(rows))), shape=(size, 1)).tolil()
