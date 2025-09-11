import src.shape_functions as shape_functions
from src.utils import Config
from src.mesh import Mesh
from src.fem_utils import *


def calc_Ke_plane_stress(config: Config, mesh: Mesh, e):
    assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
    assert e >= 0 and e < mesh.get_nE()
    element_type = config.element_type
    E = config.E
    nu = config.nu
    thickness = config.h
    elements = mesh.elements
    x_l = mesh.nodes[elements[e, :], 0]
    y_l = mesh.nodes[elements[e, :], 1]
    nNl = element_type_to_nNl[element_type]
    assert len(x_l) == nNl and len(x_l) == nNl

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
            dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            assert len(N) == nNl and len(dNdx) == nNl and len(dNdy) == nNl
            B = np.zeros((3, 2 * nNl))
            for k in range(nNl):
                #fmt: off
                B[:, 2 * k:2 * k + 2] = np.array([[dNdx[k],     0],
                                                  [0,       dNdy[k]],
                                                  [dNdy[k], dNdx[k]]])
                #fmt: on

            Ke += B.T @ D @ B * detJ * weight * thickness

    return Ke


def calc_Ke_hourglass_plane_stress_Q4R(config: Config, mesh: Mesh, e):
    assert config.element_type == ELEMENT_TYPE_Q4R
    #====================================================================
    # Add hourglass stabilization term: See Page 522, eq. (8.7.11) in
    # the book Nonlinear Finite Elements for Continua and Structures, by
    # Belytschko, Liu and Moran
    #====================================================================

    #For reduced integration of Q4R, we only have one Gauss point at the centroid
    xi = 0
    eta = 0
    element_type = config.element_type
    E = config.E
    nu = config.nu
    thickness = config.h
    elements = mesh.elements
    x_l = mesh.nodes[elements[e, :], 0]
    y_l = mesh.nodes[elements[e, :], 1]
    nNl = element_type_to_nNl[element_type]
    assert len(x_l) == nNl and len(x_l) == nNl
    dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
    J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
    detJ = np.linalg.det(J)
    #Shape function derivatives at the element center
    b_x, b_y = dNdx, dNdy
    A = 4 * detJ  #Element area
    h = np.array([1, -1, 1, -1]).transpose()
    gamma = 1 / 4 * (h - (h.dot(x_l)) * b_x - (h.dot(y_l)) * b_y)
    #====================================================================
    # Need to do this since the formulation in the book orders the dofs as
    # [u1, u2, u3, u4, v1, v2, v3, v4] whereas we use
    # [u1, v1, u2, v2, u3, v3, u4, v4]
    #====================================================================
    gamma_x = np.array([gamma[0], 0, gamma[1], 0, gamma[2], 0, gamma[3], 0])
    gamma_y = np.array([0, gamma[0], 0, gamma[1], 0, gamma[2], 0, gamma[3]])

    mu = E / (2 * (1 + nu))
    #Need to check the scaling C_Q, as it's based on dynamic eigenvalue analysis
    C_Q = 0.5 * config.hourglass_scaling * mu * (b_x.dot(b_x) + b_y.dot(b_y))  #removed A

    #fmt: off
    K_hg = C_Q * thickness* A * (np.outer(gamma_x, gamma_x) + np.outer(gamma_y, gamma_y))
    #fmt: on

    return K_hg


def calc_Ke_mindlin_plate(config: Config, mesh: Mesh, e):
    assert config.problem_type == PROBLEM_TYPE_PLATE
    assert e >= 0 and e < mesh.get_nE()
    element_type = config.element_type
    E = config.E
    nu = config.nu
    h = config.h
    elements = mesh.elements
    x_l = mesh.nodes[elements[e, :], 0]
    y_l = mesh.nodes[elements[e, :], 1]
    nNl = element_type_to_nNl[element_type]
    assert len(x_l) == nNl and len(x_l) == nNl

    #fmt: off
    D = E * h**3 / (12 * (1 - nu**2)) * np.array([[1,   nu, 0],
                                                  [nu,  1,  0],
                                                  [0,   0,  (1 - nu) / 2]])
    #fmt: on
    Cs = E / (2 * (1 + nu)) * np.identity(2)

    hs = 5 / 6 * h

    nGauss = shape_functions.element_type_to_nGauss_1D[element_type]
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)

    Ke = np.zeros((3 * nNl, 3 * nNl))

    for i in range(nGauss):
        for j in range(nGauss):
            xi = arr_xi[i]
            eta = arr_xi[j]
            weight = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            assert len(N) == nNl and len(dNdx) == nNl and len(dNdy) == nNl
            Bb = np.zeros((3, 3 * nNl))
            Bs = np.zeros((2, 3 * nNl))
            for k in range(nNl):
                #fmt: off
                Bb[:, 3 * k:3 * k + 3] = np.array([[0,  0,       -dNdx[k]],
                                                   [0,  dNdy[k],  0      ],
                                                   [0,  dNdx[k], -dNdy[k]]])
                Bs[:, 3 * k:3 * k + 3] = np.array([[dNdy[k], -N[k], 0    ],
                                                   [dNdx[k],  0,    N[k] ]])
                #fmt: on
            Ke += (Bb.T @ D @ Bb + hs * Bs.T @ Cs @ Bs) * detJ * weight

    return Ke


def calc_stress_plane_stress_consistent(e, config: Config, mesh: Mesh, nodes, u, v, xi_eta: list[tuple]):
    assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
    assert e >= 0 and e < mesh.get_nE()
    elements = mesh.elements
    element_type = config.element_type
    nNl = element_type_to_nNl[element_type]
    E = config.E
    nu = config.nu
    n_sample_points = len(xi_eta)

    #fmt: off
    D = E / (1 - nu**2) * np.array([[1,     nu,     0],
                                    [nu,    1,      0],
                                    [0,     0,  (1 - nu) / 2]])
    #fmt: on
    arr_sigma = np.zeros((n_sample_points, 3))
    x_l = nodes[elements[e, :], 0]
    y_l = nodes[elements[e, :], 1]
    r_e = np.zeros(2 * nNl)
    r_e[0::2] = u[elements[e, :]]
    r_e[1::2] = v[elements[e, :]]

    assert len(x_l) == nNl and len(y_l) == nNl and len(r_e) == 2 * nNl
    for i in range(n_sample_points):
        xi, eta = xi_eta[i]
        dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
        B = np.zeros((3, 2 * nNl))
        for k in range(nNl):
            #fmt: off
            B[:, 2 * k:2 * k + 2] = np.array([[dNdx[k],     0],
                                                [0,       dNdy[k]],
                                                [dNdy[k], dNdx[k]]])
            #fmt: on
        arr_sigma[i, :] = D @ B @ r_e
    return arr_sigma


# def calc_stress_plane_stress_consistent_all_elements(config: Config, mesh: Mesh, nodes, u, v, xi_eta: list[tuple]):
#     #xi_eta is a list of local coords where the stress is sampled for each element. These
#     #need not correspond to the Gauss points

#     assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
#     nQ = len(xi_eta)
#     element_type = config.element_type
#     nNl = element_type_to_nNl[config.element_type]
#     elements = mesh.elements
#     nE = mesh.get_nE()
#     E = config.E
#     nu = config.nu
#     #fmt: off
#     D = E / (1 - nu**2) * np.array([[1,     nu,     0],
#                                     [nu,    1,      0],
#                                     [0,     0,  (1 - nu) / 2]])
#     #fmt: on
#     arr_sigma = np.zeros((nE * nQ, 3))

#     for e in range(nE):
#         x_l = nodes[elements[e, :], 0]
#         y_l = nodes[elements[e, :], 1]
#         r_e = np.zeros(2 * nNl)
#         r_e[0::2] = u[elements[e, :]]
#         r_e[1::2] = v[elements[e, :]]

#         assert len(x_l) == nNl and len(y_l) == nNl and len(r_e) == 2 * nNl
#         for i in range(nQ):
#             xi, eta = xi_eta[i]
#             dNdx, dNdy = shape_functions.calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type)
#             B = np.zeros((3, 2 * nNl))
#             for k in range(nNl):
#                 #fmt: off
#                 B[:, 2 * k:2 * k + 2] = np.array([[dNdx[k],     0],
#                                                   [0,       dNdy[k]],
#                                                   [dNdy[k], dNdx[k]]])
#                 #fmt: on
#             arr_sigma[e * nQ + i, :] = D @ B @ r_e
#     return arr_sigma
