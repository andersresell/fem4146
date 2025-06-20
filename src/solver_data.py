from element_utils import *
import shape_functions
import numpy as np
from utils import Config
from mesh import Mesh
from scipy.sparse import csr_matrix


class SolverData:

    def __init__(self):
        self.r = None
        self.R_ext = None
        self.R_int = None  #For postprocessing
        self.K = None

        self.A = None
        self.b = None


def create_solver_data(config: Config, mesh: Mesh):
    solver_data = SolverData()
    nN = mesh.get_nN()
    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    n_eqs = NUM_DOFS * nN
    solver_data.r = np.zeros(n_eqs)
    solver_data.R_ext = np.zeros_like(solver_data.r)
    solver_data.R_int = np.zeros_like(solver_data.r)
    return solver_data


def unpack_solution(config: Config, mesh: Mesh, field: np.ndarray):
    nN = mesh.get_nN()
    assert field is not None
    assert field.shape[0] == 2 * nN or field.shape[0] == 3 * nN
    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        field_x = np.zeros(nN)
        field_y = np.zeros(nN)
        for i in range(nN):
            field_x[i] = field[i * 2 + DOF_U]
            field_y[i] = field[i * 2 + DOF_V]
        return field_x, field_y

    else:
        assert config.problem_type == PROBLEM_TYPE_PLATE
        field_z = np.zeros(nN)
        field_thetax = np.zeros(nN)
        field_thetay = np.zeros(nN)
        for i in range(nN):
            field_z[i] = field[i * 3 + DOF_W]
            field_thetax[i] = field[i * 3 + DOF_THETAX]
            field_thetay[i] = field[i * 3 + DOF_THETAY]
        return field_z, field_thetax, field_thetay
