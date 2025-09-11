from src.fem_utils import *
from src.utils import Config
from src.mesh import Mesh


class SolverData:
    """Data structure to hold the solver data, including system vectors and matrices."""

    def __init__(self):
        self.r = None  # Generalized solution vector (displacements, angles, etc. depending on the problem type)
        self.R_ext = None  # External forces vector
        self.R_int = None  # Internal forces vector (calculated from the stiffness matrix and the solution vector after solving)
        self.K = None  # Stiffness matrix
        self.K_hg = None  # Hourglass stiffness matrix (only used for some reduced integration elements)

        self.A = None  # The system matrix that is solved (K with BCs applied)
        self.b = None  # The right-hand side vector that is solved (R_ext with BCs applied)


def create_solver_data(config: Config, mesh: Mesh):
    """Creates the data structures (system vectors and matrices) needed by the solver."""
    solver_data = SolverData()
    nN = mesh.get_nN()
    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    n_eqs = NUM_DOFS * nN
    solver_data.r = np.zeros(n_eqs)
    solver_data.R_ext = np.zeros_like(solver_data.r)
    solver_data.R_int = np.zeros_like(solver_data.r)
    return solver_data


def unpack_solution(config: Config, field: np.ndarray):
    """Unpacks the generalized solution vector into separate fields for plane stress or plate problems.
    
    Example: Unpacking the displacement vector r into x and y components for plane stress problem.
        u,v = unpack_solution(config, r)
        
    Example: Unpacting the external force vector R_ext into z, thetax, and thetay components for plate problem.
        R_ext_z, R_ext_thetax, R_ext_thetay = unpack_solution(config, R_ext)
        
    """

    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    nN = len(field) // NUM_DOFS
    if nN * NUM_DOFS != len(field):
        raise ValueError(f"Field length {len(field)} is not a multiple of the number of degrees of freedom {NUM_DOFS} "
                         f"for problem type {problem_type_to_string[config.problem_type]}.")

    assert field is not None
    assert field.shape[0] == 2 * nN or field.shape[0] == 3 * nN
    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        _DOF_U_ = 0
        _DOF_V_ = 1
        field_x = np.zeros(nN)
        field_y = np.zeros(nN)
        for i in range(nN):
            field_x[i] = field[i * 2 + _DOF_U_]
            field_y[i] = field[i * 2 + _DOF_V_]
        return field_x, field_y

    else:
        assert config.problem_type == PROBLEM_TYPE_PLATE
        _DOF_W_ = 0
        _DOF_THETAX_ = 1
        _DOF_THETAY_ = 2
        field_z = np.zeros(nN)
        field_thetax = np.zeros(nN)
        field_thetay = np.zeros(nN)
        for i in range(nN):
            field_z[i] = field[i * 3 + _DOF_W_]
            field_thetax[i] = field[i * 3 + _DOF_THETAX_]
            field_thetay[i] = field[i * 3 + _DOF_THETAY_]
        return field_z, field_thetax, field_thetay
