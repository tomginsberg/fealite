from typing import Tuple, List
import numpy as np
from subprocess import run
from os.path import realpath
from shapefunction import LinearShapeFunction


class TriangleMesh:
    def __init__(self, file_name: str):
        self.file_name = realpath(file_name)
        # coordinate lookup for vertex i
        self.coordinates: List[np.ndarray] = []

        # list of triangle elements with material markers
        self.mesh_elements: List[Tuple[int, int, int]] = []
        self.mesh_markers: List[int] = []
        self.mesh_shape_functions: List[LinearShapeFunction] = []

        # list of line elements with boundary markers
        self.boundary_elements: List[Tuple[int, int]] = []
        self.boundary_markers: List[int] = []

        # default: material[0] = permittivity of free space
        # for magnetic simulations use mu = 4π * 10 ^-7
        self.material_constants: List[float] = [8.854e-12]

        self._parse_file(file_name)

    def _parse_file(self, file_name: str):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        section = None
        for line in lines:
            if '#' in line:
                section = line[1:].strip()
            elif section == 'Coordinates':
                self._add_coordinate(*[float(x) for x in line.split('\t')])
            elif section == 'Triangle Elements':
                self._add_mesh_element(*[int(x) for x in line.split('\t')])
            elif section == 'Boundary Elements':
                self._add_boundary_element(*[int(x) for x in line.split('\t')])

    def _compute_shape_functions(self):
        self.mesh_shape_functions = [LinearShapeFunction(*self.element_coords(element)) for element in
                                     self.mesh_elements]

    def _add_coordinate(self, x: float, y: float):
        self.coordinates.append(np.array((x, y), dtype='float16'))

    def _add_mesh_element(self, v1: int, v2: int, v3: int, mesh_marker: int):
        self.mesh_elements.append((v1 - 1, v2 - 1, v3 - 1))
        self.mesh_markers.append(mesh_marker)

    def _add_boundary_element(self, v1: int, v2: int, boundary_marker: int):
        self.boundary_elements.append((v1 - 1, v2 - 1))
        self.boundary_markers.append(boundary_marker)

    def set_material_constants(self, mapping: List[float]):
        self.material_constants = mapping

    def element_coords(self, element: Tuple[int, int, int]) -> List[np.ndarray]:
        return [self.coordinates[i] for i in element]

    def render_mesh(self, input_file=None, output_file=None):
        if input_file is None:
            input_file = self.file_name
        if output_file is None:
            output_file = self.file_name[:-3] + 'pdf'
        return run(['wolframscript', '-f', f'{input_file}', f'{output_file}'])


if __name__ == '__main__':
    mesh = TriangleMesh(file_name='meshes/sample-mesh1.tmh')
