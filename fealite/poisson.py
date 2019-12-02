from abc import ABC
from math import pi
from typing import Union, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
from scipy.optimize import fsolve
from problem_definitions import PoissonProblemDefinition, NonLinearPoissonProblemDefinition
import matrix_assmbly
from matrix_assmbly import NonLinearSystem, apply_neumann, apply_dirichlet
from mesh import TriangleMesh

EPS0 = 8.854e-12
MU0 = 4e-7 * pi


class Poisson(ABC):
    def __init__(self, definition: Union[PoissonProblemDefinition, NonLinearPoissonProblemDefinition]):
        self.name = definition.name
        self.mesh = definition.mesh
        self.alpha = definition.material
        self.f = definition.source
        self.p = definition.dirichlet_boundary
        self.q = definition.neumann_boundary
        self.K = matrix_assmbly.assemble_global_stiffness_matrix(self.mesh, self.alpha)
        self.b = matrix_assmbly.assemble_global_vector(self.mesh, self.f, self.K.shape[0])
        apply_dirichlet(self.mesh, self.K, self.b, self.p)
        apply_neumann(self.mesh, self.b, self.q)

    def _export_solution(self, solution):
        export_path = f'solutions/{self.mesh.short_name}_{self.name}.txt'
        with open(export_path, 'w') as f:
            f.write('\n'.join(['\t'.join([f'{c:f}' for c in cord]) + f'\t{z:f}' for cord, z in
                               zip(self.mesh.coordinates, solution)]))

    def solve_and_export(self):
        raise NotImplementedError


class LinearPoisson(Poisson):
    """
    Assembles the linear FEA system for the general Poisson problem  - ∇·( alpha(material) ∇phi(x,y) ) = f(x,y; material)
    alpha is a spatial function only of the material properties (i.e ε)
    f is a source function of a material and coordinate (i.e ρ)
    """

    def __init__(self, definition: PoissonProblemDefinition):
        super().__init__(definition)

    def solve_and_export(self):
        super()._export_solution(spsolve(self.K.tocsc(), self.b))


class NonLinearPoisson(Poisson):
    """
        Assembles the non linear FEA system for the general Poisson problem
        - ∇·( alpha(material, norm(∇phi)) ∇phi(x,y) ) = f(x,y; material)
        alpha is a spatial function only of the material properties (i.e ε)
        f is a source function of a material and coordinate (i.e ρ)
        """

    def __init__(self, definition: NonLinearPoissonProblemDefinition):
        super().__init__(definition)

    def solve_and_export(self):
        a_0 = spsolve(self.K, self.b)
        n_sys = NonLinearSystem(self.mesh, self.alpha, self.p, self.q, np.array(self.b.todense()).transpose().squeeze())
        solution = fsolve(n_sys.sys_eval, a_0, fprime=n_sys.jacobian)
        super()._export_solution(solution)
