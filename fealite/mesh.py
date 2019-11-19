from typing import Tuple, List, Union, Dict
import numpy as np
from subprocess import run
from os.path import realpath
from shapefunction import LinearShapeFunction
import matplotlib.pyplot as plt


class TriangleMesh:
    def __init__(self, file_name: str):
        self.file_name = realpath(file_name)
        self.short_name = file_name.split('/')[-1][:-4]
        # coordinate lookup for vertex i
        self.coordinates: List[np.ndarray] = []

        # list of triangle elements with material markers
        self.mesh_elements: List[Tuple[int, int, int]] = []
        self.mesh_markers: List[int] = []
        self.mesh_shape_functions: List[LinearShapeFunction] = []
        self.neighbors: List[List[int]] = []

        # list of line elements with boundary markers
        self.boundary_elements: List[Tuple[int, int]] = []
        self.boundary_markers: List[int] = []
        self.boundary_master: Dict[int, int] = {}

        self._parse_file(file_name)
        self._compute_shape_functions()
        # self._compute_neighbors()

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

    def _compute_neighbors(self):
        self.neighbors = [[] for _ in self.coordinates]
        for element in self.mesh_elements:
            for v1 in element:
                for v2 in element:
                    self.neighbors[v1].append(v2)

    def _add_coordinate(self, x: float, y: float):
        self.coordinates.append(np.array((x, y)))

    def _add_mesh_element(self, v1: int, v2: int, v3: int, mesh_marker: int):
        self.mesh_elements.append((v1 - 1, v2 - 1, v3 - 1))
        self.mesh_markers.append(mesh_marker)

    def _add_boundary_element(self, v1: int, v2: int, boundary_marker: int):
        self.boundary_elements.append((v1 - 1, v2 - 1))
        self.boundary_markers.append(boundary_marker)
        self.boundary_master[v1 - 1] = boundary_marker
        self.boundary_master[v2 - 1] = boundary_marker

    def element_coords(self, element: Tuple[int, int, int]) -> List[np.ndarray]:
        return [self.coordinates[i] for i in element]

    def render_mesh(self, input_file=None, output_file=None):
        if input_file is None:
            input_file = self.file_name
        if output_file is None:
            output_file = self.file_name[:-3] + 'pdf'
        return run(['wolframscript', '-f', f'{input_file}', f'{output_file}'])

    def show_mesh(self, title: Union[str, None] = 'filename', label_everything=False):
        plt.axes()
        for mesh_id, (i, j, k) in enumerate(self.mesh_elements):
            vertices = self.element_coords((i, j, k))
            tri = plt.Polygon(vertices, edgecolor='black',
                              linewidth=.3, facecolor='white')
            plt.gca().add_patch(tri)
            if label_everything:
                plt.text(*vertices[0], f'{i}', size=12)
                plt.text(*vertices[1], f'{j}', size=12)
                plt.text(*vertices[2], f'{k}', size=12)
                plt.text(*sum(vertices) / 3, f'{mesh_id}', size=12)

        if title == 'filename':
            plt.title(self.file_name.split('/')[-1][:-4])
        if title is not None:
            plt.title(title)
        plt.axis('scaled')
        plt.show()

    def __len__(self):
        return len(self.coordinates)


if __name__ == '__main__':
    # mesh = TriangleMesh(file_name='meshes/cylinder-in-square.tmh')
    mesh = TriangleMesh('meshes/verysimple.tmh')
    mesh.show_mesh(title=None, label_everything=False)
