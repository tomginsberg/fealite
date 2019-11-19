from scipy import sparse


def assemble_global_stiffness_matrix(mesh_elements, mesh_shape_functions, mesh_markers, alpha):
    rows, cols, values = [], [], []
    for element, shp_fn, marker in zip(mesh_elements, mesh_shape_functions,
                                       mesh_markers):
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
