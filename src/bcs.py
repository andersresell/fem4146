from scipy.sparse import lil_matrix
from src.mesh import Mesh
from src.utils import Config, BC
from src.solver_data import SolverData
from src.fem_utils import *


def add_boundary_condition(config: Config, mesh: Mesh, node_set_name, dof, value=0):
    """Adds a prescribed boundary condition for a certain dof to a node set.
        For instance, if node_set_name is "west", dof is DOF_U, and value is 0, 
        all nodes in the "west" node set will have their x-displacement (DOF_U) set to 0.
    """
    node_sets = mesh.node_sets
    bcs = config.bcs

    if node_set_name not in node_sets:
        all_node_set_names = ", ".join(
            node_sets.keys())  #Find all available node set names to display in the error message
        raise Exception(f"Node set '{node_set_name}' not found. Available: {all_node_set_names}")

    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        if dof != DOF_U and dof != DOF_V:
            raise Exception(f"dof {dof} is not valid for {problem_type_to_string[config.problem_type]} "
                            f"(valid dofs are: DOF_U={DOF_U}, DOF_V={DOF_V})")
    else:
        assert config.problem_type == PROBLEM_TYPE_PLATE
        if dof != DOF_W and dof != DOF_THETAX and dof != DOF_THETAY:
            raise Exception(f"dof {dof} is not valid for {problem_type_to_string[config.problem_type]} "
                            f"(valid dofs are: DOF_W={DOF_W}, DOF_THETAX={DOF_THETAX}, DOF_THETAY={DOF_THETAY})")

    bc = BC()
    bc.value = value
    bc.node_set_name = node_set_name
    bc.dof = dof
    bcs.append(bc)


# def assign_boundary_conditions(config: Config, mesh: Mesh, solver_data: SolverData):
#     #====================================================================
#     # We specify boundary condtions by zeroing out the row and column of
#     # the system matrix and modify the right hand side appropriately.
#     #====================================================================
#     K = solver_data.K
#     # Create a copy of the stiffness matrix in LIL format for efficient row/col modifications. K is copied
#     # to avoid modifying the original stiffness matrix directly. We want to keep the original stiffness matrix
#     # for computing internal forces later.
#     A = lil_matrix(K.copy())

#     #We also copy the external force vector to avoid modifying the original one, for later use
#     b = solver_data.R_ext.copy()
#     NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
#     node_sets = mesh.node_sets
#     bcs = config.bcs
#     for bc in bcs:
#         nodeIDs = node_sets[bc.node_set_name]
#         dof_id = get_dof_id(bc.dof, config.problem_type)
#         val = bc.value
#         for I in nodeIDs:
#             eq = I * NUM_DOFS + dof_id
#             #====================================================================
#             # In the first sweep, we modify the right hand side vector (b) by
#             # the forces created from thconst Mesh &mesh, const NodeFieldData &node_field_data, byte *bufe prescribed dofs
#             #====================================================================
#             b -= val * K[:, eq].toarray().ravel()

#     for bc in bcs:
#         nodeIDs = node_sets[bc.node_set_name]
#         dof_id = get_dof_id(bc.dof, config.problem_type)
#         val = bc.value
#         for I in nodeIDs:
#             eq = I * NUM_DOFS + dof_id
#             #====================================================================
#             # In the second sweep, we modify the system matrix A and right hand
#             # side vector b of the prescribed dofs so that the boundary condition
#             # is obeyed. We do this by setting the rows and cols of the equation
#             # in question to zero except the diagonal, which is one.
#             # This is combined with and setting b to the prescribed value
#             #====================================================================
#             A[:, eq] = 0
#             A[eq, :] = 0
#             A[eq, eq] = 1
#             b[eq] = val

#     solver_data.A = A
#     solver_data.b = b
#     print("Boundary conditions assigned")


def assign_boundary_conditions(config: Config, mesh: Mesh, solver_data: SolverData):
    #====================================================================
    # We specify boundary condtions by zeroing out the row and column of
    # the system matrix and modify the right hand side appropriately.
    #====================================================================
    K = solver_data.K
    # Create a copy of the stiffness matrix in LIL format for efficient row/col modifications. K is copied
    # to avoid modifying the original stiffness matrix directly. We want to keep the original stiffness matrix
    # for computing internal forces later.
    A = lil_matrix(K.copy())

    #We also copy the external force vector to avoid modifying the original one, for later use
    b = solver_data.R_ext.copy()
    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
    node_sets = mesh.node_sets
    bcs = config.bcs
    u_prescribed = np.zeros_like(b)
    for bc in bcs:
        nodeIDs = node_sets[bc.node_set_name]
        dof_id = get_dof_id(bc.dof, config.problem_type)
        val = bc.value
        for I in nodeIDs:
            eq = I * NUM_DOFS + dof_id
            u_prescribed[eq] = val

    #====================================================================
    # In the first sweep, we modify the right hand side vector (b) by
    # the forces created from the prescribed dofs
    #====================================================================
    b -= K @ u_prescribed

    for bc in bcs:
        nodeIDs = node_sets[bc.node_set_name]
        dof_id = get_dof_id(bc.dof, config.problem_type)
        val = bc.value

        for I in nodeIDs:
            eq = I * NUM_DOFS + dof_id
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
    print("Boundary conditions assigned")
