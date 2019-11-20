import numpy as np


def linear_shape_function_row(xj, yj, xk, yk):
    return [(xj * yk - xk * yj), (yj - yk), (xk - xj)]


def linear_shape_function_element_derivative(xj, yj, xk, yk, variable='x'):
    if variable == 'x':
        return yj - yk
    return xk - xj


class LinearShapeFunction:
    def __init__(self, c1, c2, c3):
        # Positive area if points are in a clockwise orientation
        self.double_area = np.cross(c2 - c1, c3 - c1)

        self.N_ijk = np.array(
            [linear_shape_function_row(*x, *y) for x, y in [(c2, c3), (c3, c1), (c1, c2)]]) / self.double_area

        self.div_N_ijk__x = np.array(
            [[linear_shape_function_element_derivative(*x, *y, 'x') for x, y in
              [(c2, c3), (c3, c1), (c1, c2)]]]) / self.double_area

        self.div_N_ijk__y = np.array(
            [[linear_shape_function_element_derivative(*x, *y, 'y') for x, y in
              [(c2, c3), (c3, c1), (c1, c2)]]]) / self.double_area

        self.stiffness_matrix = np.array(np.matmul(self.div_N_ijk__x.transpose(), self.div_N_ijk__x) + np.matmul(
            self.div_N_ijk__y.transpose(), self.div_N_ijk__y)) * self.double_area / 2

    def shape_values(self, x, y):
        return np.matmul(self.N_ijk, [1, x, y])

    def data_value(self, ai, aj, ak, x, y):
        return np.dot(self.shape_values(x, y), [ai, aj, ak])


if __name__ == '__main__':
    coords = [np.array(x) for x in [(0, 0), (1.8, 2.2), (3.1, .8)]]
    shp = LinearShapeFunction(*coords)
    vals = shp.shape_values(1 / 2, 1 / 2)
