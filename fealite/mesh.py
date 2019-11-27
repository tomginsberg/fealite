from typing import Tuple, List, Union, Dict, Optional
import numpy as np
from subprocess import run
from os.path import realpath
from shapefunction import LinearShapeFunction
import matplotlib.pyplot as plt


class TriangleMesh:
    def __init__(self, file_name: str):
        # coordinate lookup for vertex i
        self.coordinates: List[np.ndarray] = []

        # list of triangle elements with material markers
        self.mesh_elements: List[Tuple[int, int, int]] = []
        self.mesh_markers: List[int] = []
        self.mesh_shape_functions: List[LinearShapeFunction] = []

        # list of line elements with boundary markers
        self.boundary_elements: List[Tuple[int, int]] = []
        self.boundary_markers: List[int] = []
        self.boundary_dict: Dict[int, int] = {}

        self.file_name = realpath(file_name)
        self.short_name = self.file_name.split('/')[-1][:-4]

        self._parse_file(file_name)
        self._compute_shape_functions()

    def _parse_file(self, file_name: str):
        with open(file_name, 'r') as f:
            lines = f.readlines()
        section = None
        for line in lines:
            if '#' in line:
                section, count = line[1:].strip().split('-')

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
        self.coordinates.append(np.array((x, y)))

    def _add_mesh_element(self, v1: int, v2: int, v3: int, mesh_marker: int):
        self.mesh_elements.append((v1 - 1, v2 - 1, v3 - 1))
        self.mesh_markers.append(mesh_marker)

    def _add_boundary_element(self, v1: int, v2: int, boundary_marker: int):
        self.boundary_elements.append((v1 - 1, v2 - 1))
        self.boundary_markers.append(boundary_marker)
        self.boundary_dict[v1 - 1] = boundary_marker
        self.boundary_dict[v2 - 1] = boundary_marker

    def element_coords(self, element: Tuple[int, int, int]) -> List[np.ndarray]:
        return [self.coordinates[i] for i in element]

    def render_mesh(self, input_file=None, output_file=None):
        if input_file is None:
            input_file = self.file_name
        if output_file is None:
            output_file = self.file_name[:-3] + 'pdf'
        return run(['wolframscript', '-f', f'{input_file}', f'{output_file}'])

    def show_mesh(self, title: Optional[str] = 'filename', material_labels: bool = False,
                  boundary_labels: bool = False):
        plt.axes()
        for mesh_id, (i, j, k) in enumerate(self.mesh_elements):
            vertices = self.element_coords((i, j, k))
            tri = plt.Polygon(vertices, edgecolor='black',
                              linewidth=.3, facecolor='white')
            plt.gca().add_patch(tri)
            if material_labels:
                plt.text(*sum(vertices) / 3, f'{self.mesh_markers[mesh_id]}', size=12)
        if boundary_labels:
            for mesh_id, element in enumerate(self.boundary_elements):
                plt.text(*sum([self.coordinates[i] for i in element]) / 2, f'{self.boundary_markers[mesh_id]}', size=12)

        if title == 'filename':
            plt.title(self.short_name)
        if title is not None:
            plt.title(title)
        plt.axis('scaled')
        plt.show()

    def __len__(self):
        return len(self.coordinates)


class Meshes:
    cylinder_in_square = 'meshes/cylinder-in-square.tmh'
    airfoil = 'meshes/airfoil.tmh'
    annulus = 'meshes/annulus.tmh'
    cylinder_in_square_fine = 'meshes/cylinder-in-square-fine.tmh'
    heart = 'meshes/heart.tmh'
    unit_disk = 'meshes/unit_disk.tmh'


if __name__ == '__main__':
    # mesh = TriangleMesh(file_name='meshes/cylinder-in-square.tmh')
    mesh = TriangleMesh(Meshes.heart)
    mesh.show_mesh(title=None)
    # mesh.show_mesh(title=None, label_everything=False)
