import shape_functions
from utils import Config
from mesh import Mesh
from element_utils import *
from solver_data import SolverData
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import element_stiffness


def calc_Ke(config: Config, mesh: Mesh, solver_data: SolverData, e):
    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        return element_stiffness.calc_Ke_plane_stress(config, mesh, e)
    else:
        assert config.problem_type == PROBLEM_TYPE_PLATE
        assert False
    # return calc_Ke_plate(config, mesh, e)


def assemble_stiffness_matrix(mesh: Mesh, config: Config, solver_data: SolverData):
    element_type = config.element_type
    elements = mesh.elements
    nodes = mesh.nodes
    nNl = element_type_to_nNl[element_type]
    nN = mesh.get_nN()
    nE = mesh.get_nE()
    assert elements.shape[1] == nNl
    assert nodes.shape[1] == 2

    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    n_eqs = NUM_DOFS * nN

    K = lil_matrix((n_eqs, n_eqs),
                   dtype=np.float64)  #This is more efficient for incrementally assembling sparse matrices
    # R = np.zeros(n_eqs, dtype=np.float64)

    for e in range(nE):
        Ke = calc_Ke(config, mesh, solver_data, e)
        element = elements[e, :]
        for il in range(nNl):
            for jl in range(nNl):
                for dof_i in range(NUM_DOFS):
                    for dof_j in range(NUM_DOFS):
                        i = element[il] * NUM_DOFS + dof_i
                        j = element[jl] * NUM_DOFS + dof_j
                        K[i, j] += Ke[NUM_DOFS * il + dof_i, NUM_DOFS * jl + dof_j]

    solver_data.K = K

    print("Stiffness matrix assembly complete")


def assign_boundary_conditions(config: Config, mesh: Mesh, solver_data: SolverData):
    #====================================================================
    # We specify boundary condtions by zeroing out the row and column of
    # the system matrix and modify the right hand side appropriately.
    #====================================================================
    K = solver_data.K
    A = lil_matrix(K)
    b = np.array(solver_data.R)

    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    node_sets = mesh.node_sets
    bcs = config.bcs
    for bc in bcs:
        nodeIDs = node_sets[bc.node_set_name]
        dof = bc.dof
        val = bc.value
        for I in nodeIDs:
            eq = I * NUM_DOFS + dof
            #====================================================================
            # In the first sweep, we modify the right hand side vector (b) by
            # the forces created from the prescribed dofs
            #====================================================================
            b -= val * K[:, eq].toarray().ravel()

    for bc in bcs:
        nodeIDs = node_sets[bc.node_set_name]
        dof = bc.dof
        val = bc.value
        for I in nodeIDs:
            eq = I * NUM_DOFS + dof
            #====================================================================
            # In the second sweep, we modify the system matrix A and right hand
            # side vector b of the prescribed dofs so that the boundary condition
            # is obeyed. We do this by setting the rows and cols of the equation
            # in question to zero except the diagonal, which is one.
            # This is combined with and setting b to the prescribed value
            #====================================================================
            A[:, eq] = 0
            A[eq, :] = 0
            A[eq, eq] = 1
            b[eq] = val

    solver_data.A = A
    solver_data.b = b


def solve(config: Config, solver_data: SolverData, mesh: Mesh):

    if len(config.bcs) == 0:
        raise Exception("No bcs specified. System will be singular")

    start_time = time.time()

    assemble_stiffness_matrix(mesh, config, solver_data)

    assign_boundary_conditions(config, mesh, solver_data)

    solver_data.A = solver_data.A.tocsr()  #Convert to csr matrix for more efficient solving
    solver_data.r = spsolve(solver_data.A, solver_data.b)  #Solve the linear system

    elapsed_time = (time.time() - start_time)
    print(f"Solver (assembly and system solution) completed in {elapsed_time:.2f} seconds")
