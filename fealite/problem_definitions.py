from abc import ABC
from typing import Optional, Union, Tuple

from mesh import TriangleMesh
import numpy as np


class PoissonProblemDefinition(ABC):

    def __init__(self, mesh: Union[str, TriangleMesh], name: str):
        if type(mesh) is str:
            self.mesh = TriangleMesh(mesh)
        else:
            self.mesh = mesh
        self.name = name

    def material(self, element_marker: int) -> float:
        raise NotImplementedError

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError


def distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


class NonLinearPoissonProblemDefinition(ABC):
    def __init__(self, mesh: Union[str, TriangleMesh], name: str = 'nonlinear'):
        if type(mesh) is str:
            self.mesh = TriangleMesh(mesh)
        else:
            self.mesh = mesh
        self.name = name

    def material(self, element_marker: int, norm_grad_phi: Optional[float] = None,
                 div: bool = False) -> Optional[float]:
        raise NotImplementedError

    def source(self, element_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def dirichlet_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError

    def neumann_boundary(self, boundary_marker: int, coordinate: np.ndarray) -> Optional[float]:
        raise NotImplementedError
