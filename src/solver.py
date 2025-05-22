import shape_functions
from utils import Config
from mesh import Mesh
from element_utils import *
from solver_data import SolverData
from scipy.sparse import lil_matrix, csr_matrix


def calc_Ke_plane_stress(config: Config, mesh: Mesh, e):
    assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
    assert e >= 0 and e < mesh.get_nE()
    element_type = config.element_type
    E = config.E
    nu = config.nu
    t = config.t

    elements = mesh.elements
    coord_x = mesh.nodes[elements[e, :], 0]
    coord_y = mesh.nodes[elements[e, :], 1]

    nNl = element_type_to_nNl[element_type]
    assert len(coord_x) == nNl and len(coord_y) == nNl

    #fmt: off
    D = E / (1 - nu**2) * np.array([[1,     nu,     0],
                                    [nu,    1,      0],
                                    [0,     0,  (1 - nu) / 2]])
    #fmt: on

    nGauss = shape_functions.element_type_to_nGauss_1D[element_type]
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)

    Ke = np.zeros((2 * nNl, 2 * nNl))

    for i in range(nGauss):
        for j in range(nGauss):
            xi = arr_xi[i]
            eta = arr_xi[j]
            weight = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, coord_x, coord_y, element_type)
            J = shape_functions.calc_J(xi, eta, coord_x, coord_y, element_type)
            detJ = np.linalg.det(J)
            assert len(N) == nNl and len(dNdx) == nNl and len(dNdy) == nNl
            B = np.zeros((3, 2 * nNl))
            for k in range(nNl):
                #fmt: off
                B[:, 2 * k:2 * k + 2] = np.array([[dNdx[k],     0],
                                                  [0,       dNdy[k]],
                                                  [dNdy[k], dNdx[k]]])
                #fmt: on

            Ke += B.T @ D @ B * detJ * weight * t
    return Ke


def calc_Ke(config: Config, mesh: Mesh, solver_data: SolverData, e):
    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        return calc_Ke_plane_stress(config, mesh, e)
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

    K = lil_matrix((n_eqs, n_eqs), dtype=np.float64)
    # R = np.zeros(n_eqs, dtype=np.float64)

    for e in range(nE):
        Ke = calc_Ke(config, mesh, solver_data, e)
        element = elements[e, :]
        K[element, element] += Ke

    print("Stiffness matrix assembly complete")
    return K
