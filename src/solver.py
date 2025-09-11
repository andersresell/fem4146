from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
from src.utils import Config
from src.mesh import Mesh
from src.fem_utils import *
from src.solver_data import SolverData
import src.element_stiffness as element_stiffness
import src.element_stiffness_user as element_stiffness_user
import src.loads as loads
import src.bcs as bcs


def calc_Ke(config: Config, mesh: Mesh, e, calc_hourglass_stiffness_only=False):
    if not calc_hourglass_stiffness_only:
        #====================================================================
        # This is the normal case: Calculate normal stiffness and
        # hourglass stiffness
        #====================================================================
        if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
            if config.element_type == ELEMENT_TYPE_Q4_USER:
                Ke = element_stiffness_user.calc_Ke_plane_stress_user_Q4(config, mesh, e)
            else:
                Ke = element_stiffness.calc_Ke_plane_stress(config, mesh, e)
                if config.element_type == ELEMENT_TYPE_Q4R:
                    Ke += element_stiffness.calc_Ke_hourglass_plane_stress_Q4R(config, mesh, e)
        else:
            assert config.problem_type == PROBLEM_TYPE_PLATE
            Ke = element_stiffness.calc_Ke_mindlin_plate(config, mesh, e)
    else:
        #====================================================================
        # This case is used when we only want to assemble the hourglass
        # stiffness part of the stiffness matrix (for artificial energy check)
        #====================================================================
        if config.problem_type == PROBLEM_TYPE_PLANE_STRESS and config.element_type == ELEMENT_TYPE_Q4R:
            Ke = element_stiffness.calc_Ke_hourglass_plane_stress_Q4R(config, mesh, e)
        else:
            nNl = element_type_to_nNl[config.element_type]
            NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)
            Ke = np.zeros((NUM_DOFS * nNl, NUM_DOFS * nNl))  #Return zero matrix if no hourglass stabilization is used
    return Ke


def assemble_stiffness_matrix(mesh: Mesh, config: Config, solver_data: SolverData, calc_hourglass_stiffness_only=False):
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

    for e in range(nE):
        Ke = calc_Ke(config, mesh, e, calc_hourglass_stiffness_only)
        element = elements[e, :]
        for il in range(nNl):
            for jl in range(nNl):
                for dof_i in range(NUM_DOFS):
                    for dof_j in range(NUM_DOFS):
                        i = element[il] * NUM_DOFS + dof_i
                        j = element[jl] * NUM_DOFS + dof_j
                        K[i, j] += Ke[NUM_DOFS * il + dof_i, NUM_DOFS * jl + dof_j]
    if not calc_hourglass_stiffness_only:
        print("Stiffness matrix assembly complete")
    return K


def solve(config: Config, solver_data: SolverData, mesh: Mesh):
    """Assembles the stiffness matrix, applies boundary conditions, and solves the system K*r = R."""

    if len(config.bcs) == 0:
        raise Exception("No bcs specified. System will be singular")

    start_time = time.time()

    solver_data.K = assemble_stiffness_matrix(mesh, config, solver_data)

    loads.assemble_loads_consistent(config, mesh, solver_data)

    bcs.assign_boundary_conditions(config, mesh, solver_data)

    solver_data.A = solver_data.A.tocsr()  #Convert to csr matrix for more efficient solving
    solver_data.r = spsolve(solver_data.A, solver_data.b)  #Solve the linear system

    solver_data.R_int = solver_data.K @ solver_data.r  #Calculate internal forces

    if config.element_type == ELEMENT_TYPE_Q4R and config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        #====================================================================
        # Check the artificial hourglass energy
        #====================================================================
        K_hg = assemble_stiffness_matrix(mesh, config, solver_data, calc_hourglass_stiffness_only=True)
        U_hg = 0.5 * solver_data.r.T @ K_hg @ solver_data.r
        U = 0.5 * solver_data.r.T @ solver_data.K @ solver_data.r
        print(f"Hourglass energy: U_hg = {U_hg:.3e}, U = {U:.3e}, artficial energy percentage: {U_hg / U * 100:.2f} %")

    elapsed_time = (time.time() - start_time)
    print(f"Solver (assembly and system solution) completed in {elapsed_time:.2f} seconds")
    rel_res = np.linalg.norm(solver_data.A @ solver_data.r - solver_data.b) / np.linalg.norm(solver_data.b)
    print(f"Relative solver residual ||A*r - b||/||b|| = {rel_res:.1e} ")
