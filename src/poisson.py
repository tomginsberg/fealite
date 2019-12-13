from abc import ABC
from typing import Union
from time import time
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve
from problem_definitions import PoissonProblemDefinition, NonLinearPoissonProblemDefinition
import matrix_assmbly
from matrix_assmbly import NonLinearSystem, apply_neumann, apply_dirichlet


class Poisson(ABC):
    def __init__(self, definition: Union[PoissonProblemDefinition, NonLinearPoissonProblemDefinition]):
        self.name = definition.name
        self.mesh = definition.mesh
        self.alpha = definition.material
        self.f = definition.source
        self.p = definition.dirichlet_boundary
        self.q = definition.neumann_boundary
        print("Assembling Global Matrix")
        self.K = matrix_assmbly.assemble_global_stiffness_matrix(self.mesh, self.alpha)
        self.b = matrix_assmbly.assemble_global_vector(self.mesh, self.f, self.K.shape[0])

    def _export_solution(self, solution):
        print('Exporting Solution')
        export_path = f'../solutions/{self.mesh.short_name}_{self.name}.txt'
        print(export_path)
        with open(export_path, 'w') as f:
            f.write('\n'.join(['\t'.join([f'{c:.12f}' for c in cord]) + f'\t{z:.12f}' for cord, z in
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
        apply_dirichlet(self.mesh, self.K, self.b, self.p)
        apply_neumann(self.mesh, self.b, self.q)
        print('Solving Linear System')
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
        start = time()
        print('Starting Non Linear Solver...')
        apply_neumann(self.mesh, self.b, self.q)
        # Save a copy of b that hasn't had dirichlet modifications applied to it
        b_un_modified = self.b.copy()

        # Apply dirichlet conditions
        apply_dirichlet(self.mesh, self.K, self.b, self.p)

        # Solve for the first guess using material none value
        a_0 = np.array(spsolve(self.K.tocsc(), self.b))
        print(f'Initial Guess Calculated: {time() - start:.3f}s')
        n_sys = NonLinearSystem(self.mesh, self.alpha, self.p, self.q,
                                np.array(b_un_modified.todense()).transpose().squeeze())
        solution = fsolve(n_sys.sys_eval, a_0, fprime=n_sys.jacobian)
        print(f'Solved. Total Time: {time() - start:.3f}s')
        super()._export_solution(solution)


def format_matrix(x: np.ndarray) -> str:
    return '\n'.join(map(lambda row: '\t'.join(map(str, row)), x))


def export_matrix(name: str, matrix: np.ndarray):
    with open(name, 'w') as f:
        f.write(format_matrix(matrix))
        print(f'{name} is written')
