from src.fem_utils import *


def calc_N_1D(xi, nNl_1d):
    N = np.zeros(nNl_1d)
    if nNl_1d == 2:
        N[0] = 0.5 * (1 - xi)
        N[1] = 0.5 * (1 + xi)
    elif nNl_1d == 3:
        N[0] = 0.5 * xi * (xi - 1)
        N[1] = 1 - xi**2
        N[2] = 0.5 * xi * (xi + 1)
    elif nNl_1d == 4:
        N[0] = -(9.0 / 16) * (xi + 1.0 / 3) * (xi - 1.0 / 3) * (xi - 1)
        N[1] = (27.0 / 16) * (xi + 1) * (xi - 1.0 / 3) * (xi - 1)
        N[2] = (-27.0 / 16) * (xi + 1) * (xi + 1.0 / 3) * (xi - 1)
        N[3] = (9.0 / 16) * (xi + 1) * (xi + 1.0 / 3) * (xi - 1.0 / 3)
    else:
        assert False
    return N


def calc_dNdxi_1D(xi, nNl_1d):
    dN = np.zeros(nNl_1d)
    if nNl_1d == 2:
        dN[0] = -0.5
        dN[1] = 0.5
    elif nNl_1d == 3:
        dN[0] = xi - 0.5
        dN[1] = -2 * xi
        dN[2] = xi + 0.5
    elif nNl_1d == 4:
        dN[0] = 1.0 / 16 * (-27 * xi * xi + 18 * xi + 1)
        dN[1] = 9.0 / 16 * (9 * xi * xi - 2 * xi - 3)
        dN[2] = -9.0 / 16 * (9 * xi * xi + 2 * xi - 3)
        dN[3] = 1.0 / 16 * (27 * xi * xi + 18 * xi - 1)
    else:
        assert False
    return dN


def calc_N_lagrangian(xi, eta, element_type):
    assert element_type_to_category(element_type) == ELEMENT_CATEGORY_LAGRANGIAN
    nNl = element_type_to_nNl[element_type]
    nNl_1d = element_type_to_nNl_1D[element_type]
    N = np.zeros(nNl)
    il_to_ij_loc = il_to_ij_loc_all[element_type]
    N1D_xi = calc_N_1D(xi, nNl_1d)
    N1D_eta = calc_N_1D(eta, nNl_1d)
    for il in range(nNl):
        i = il_to_ij_loc[il][0]
        j = il_to_ij_loc[il][1]
        N[il] = N1D_xi[i] * N1D_eta[j]
    return N


def calc_dNdxi_dNdeta_lagrangian(xi, eta, element_type):
    assert element_type_to_category(element_type) == ELEMENT_CATEGORY_LAGRANGIAN
    nNl = element_type_to_nNl[element_type]
    nNl_1d = element_type_to_nNl_1D[element_type]
    dNdxi = np.zeros(nNl)
    dNdeta = np.zeros(nNl)
    il_to_ij_loc = il_to_ij_loc_all[element_type]
    N1D_xi = calc_N_1D(xi, nNl_1d)
    N1D_eta = calc_N_1D(eta, nNl_1d)
    dN1D_dxi = calc_dNdxi_1D(xi, nNl_1d)
    dN1D_deta = calc_dNdxi_1D(eta, nNl_1d)
    for il in range(nNl):
        i = il_to_ij_loc[il][0]
        j = il_to_ij_loc[il][1]
        dNdxi[il] = dN1D_dxi[i] * N1D_eta[j]
        dNdeta[il] = N1D_xi[i] * dN1D_deta[j]
    return dNdxi, dNdeta


def calc_N_serendipity(xi, eta, element_type):
    assert element_type_to_category(element_type) == ELEMENT_CATEGORY_SERENDIPITY
    nNl = element_type_to_nNl[element_type]
    N = np.zeros(nNl)
    if element_type == ELEMENT_TYPE_Q8 or element_type == ELEMENT_TYPE_Q8R:
        N[0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
        N[1] = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
        N[2] = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
        N[3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
        N[4] = 0.5 * (1 - xi**2) * (1 - eta)
        N[5] = 0.5 * (1 + xi) * (1 - eta**2)
        N[6] = 0.5 * (1 - xi**2) * (1 + eta)
        N[7] = 0.5 * (1 - xi) * (1 - eta**2)
    else:
        assert False
    return N


def calc_dNdxi_dNdeta_serendipity(xi, eta, element_type):
    assert element_type_to_category(element_type) == ELEMENT_CATEGORY_SERENDIPITY
    nNl = element_type_to_nNl[element_type]
    dNdxi = np.zeros(nNl)
    dNdeta = np.zeros(nNl)
    if element_type == ELEMENT_TYPE_Q8 or element_type == ELEMENT_TYPE_Q8R:
        #Obtained by symbolic differentiation (sympy)
        dNdxi[0] = 0.25 * (-eta - 2 * xi) * (eta - 1)
        dNdxi[1] = 0.25 * (eta - 1) * (eta - 2 * xi)
        dNdxi[2] = 0.25 * (eta + 1) * (eta + 2 * xi)
        dNdxi[3] = 0.25 * (-eta + 2 * xi) * (eta + 1)
        dNdxi[4] = 1.0 * xi * (eta - 1)
        dNdxi[5] = 0.5 - 0.5 * eta**2
        dNdxi[6] = -1.0 * xi * (eta + 1)
        dNdxi[7] = 0.5 * eta**2 - 0.5

        dNdeta[0] = 0.25 * (-2 * eta - xi) * (xi - 1)
        dNdeta[1] = 0.25 * (2 * eta - xi) * (xi + 1)
        dNdeta[2] = 0.25 * (2 * eta + xi) * (xi + 1)
        dNdeta[3] = 0.25 * (-2 * eta + xi) * (xi - 1)
        dNdeta[4] = 0.5 * xi**2 - 0.5
        dNdeta[5] = -1.0 * eta * (xi + 1)
        dNdeta[6] = 0.5 - 0.5 * xi**2
        dNdeta[7] = eta * (xi - 1)
    else:
        assert False
    return dNdxi, dNdeta


def calc_N(xi, eta, element_type):
    if element_type_to_category(element_type) == ELEMENT_CATEGORY_LAGRANGIAN:
        return calc_N_lagrangian(xi, eta, element_type)
    else:
        assert element_type_to_category(element_type) == ELEMENT_CATEGORY_SERENDIPITY
        return calc_N_serendipity(xi, eta, element_type)


def calc_dNdxi_dNdeta(xi, eta, element_type):
    if element_type_to_category(element_type) == ELEMENT_CATEGORY_LAGRANGIAN:
        return calc_dNdxi_dNdeta_lagrangian(xi, eta, element_type)
    else:
        assert element_type_to_category(element_type) == ELEMENT_CATEGORY_SERENDIPITY
        return calc_dNdxi_dNdeta_serendipity(xi, eta, element_type)


def calc_J(xi, eta, x_l, y_l, element_type):
    dNdxi, dNdeta = calc_dNdxi_dNdeta(xi, eta, element_type)
    assert len(x_l) == len(y_l) and len(x_l) == len(dNdxi) and len(y_l) == len(dNdeta)
    J = np.array([[dNdxi.dot(x_l), dNdxi.dot(y_l)], [dNdeta.dot(x_l), dNdeta.dot(y_l)]])
    detJ = np.linalg.det(J)
    assert detJ > SMALL_VAL
    return J


def calc_dNdx_dNdy(xi, eta, x_l, y_l, element_type):
    dNdxi, dNdeta = calc_dNdxi_dNdeta(xi, eta, element_type)
    J = calc_J(xi, eta, x_l, y_l, element_type)
    dNdXI = np.vstack((dNdxi, dNdeta))
    dNdX = np.linalg.inv(J) @ dNdXI
    return dNdX[0, :], dNdX[1, :]


def get_arr_xi(nGauss_1D):
    """Returns the natural coordinates for the Gauss quadrature rule for 1D integration."""
    if nGauss_1D == 4:
        arr_xi = np.array([
            -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)), -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
            np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))
        ])
    elif nGauss_1D == 3:
        arr_xi = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
    elif nGauss_1D == 2:
        arr_xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    elif nGauss_1D == 1:
        arr_xi = np.array([0])
    else:
        assert False
    assert abs(np.sum(arr_xi)) < SMALL_VAL
    return arr_xi


def get_arr_w(nGauss_1D):
    """Returns the weights for the Gauss quadrature rule for 1D integration."""

    if nGauss_1D == 4:
        arr_w = np.array([
            (18 - np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 + np.sqrt(30)) / 36,
            (18 - np.sqrt(30)) / 36,
        ])
    elif nGauss_1D == 3:
        arr_w = np.array([5 / 9, 8 / 9, 5 / 9])
    elif nGauss_1D == 2:
        arr_w = np.array([1, 1])
    elif nGauss_1D == 1:
        arr_w = np.array([2])
    else:
        assert False
    assert abs(np.sum(arr_w) - 2) < SMALL_VAL
    return arr_w
