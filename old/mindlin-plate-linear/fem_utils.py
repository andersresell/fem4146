import numpy as np

SMALL_VAL = 1e-8

NO_EQ = -1    

DOF_W = 0
DOF_THETAX =1
DOF_THETAY =2

DOF_FREE = 0
DOF_SUPPRESSED = 1

TYPE_Q4 = 0
TYPE_Q4R = 1
TYPE_Q9 = 2
TYPE_Q9R = 3
TYPE_Q16 = 4 

element_type_to_str =   {TYPE_Q4:"Q4",
                         TYPE_Q4R:"Q4R",
                         TYPE_Q9:"Q9",
                         TYPE_Q9R:"Q9R",
                          TYPE_Q16:"Q16",}

element_type_to_nNl = {TYPE_Q4: 4,
                       TYPE_Q4R: 4,
                       TYPE_Q9: 9,
                       TYPE_Q9R: 9,
                       TYPE_Q16: 16}

element_type_to_nNl_1D = {TYPE_Q4: 2,
                          TYPE_Q4R: 2,
                       TYPE_Q9: 3,
                       TYPE_Q9R: 3,
                       TYPE_Q16: 4} 

element_type_to_nQ = {TYPE_Q4: 2,
                      TYPE_Q4R: 1,
                      TYPE_Q9: 3,
                      TYPE_Q9R: 2,
                      TYPE_Q16: 4}


il_to_ij_loc_all ={TYPE_Q4: [(0,0),(1,0),(1,1),(0,1)],
                   TYPE_Q4R: [(0,0),(1,0),(1,1),(0,1)],
                    TYPE_Q9: [(0,0),(2,0),(2,2),(0,2),(1,0),(2,1),(1,2),(0,1),(1,1)],
                    TYPE_Q9R: [(0,0),(2,0),(2,2),(0,2),(1,0),(2,1),(1,2),(0,1),(1,1)],
                    TYPE_Q16: [(0,0),(3,0),(3,3),(0,3),(1,0),(2,0),(3,1),(3,2),(2,3),(1,3),(0,2),(0,1),(1,1),(2,1),(2,2),(1,2)]}

def calc_N_1D(xi, nNl_1d):
    N = np.zeros(nNl_1d)
    if nNl_1d==2:
        N[0] = 0.5*(1-xi)
        N[1] = 0.5*(1+xi)
    elif nNl_1d==3:
        N[0] = 0.5 * xi * (xi - 1)
        N[1] = 1 - xi ** 2
        N[2] = 0.5 * xi * (xi + 1)
    elif nNl_1d==4:
        N[0] = -(9.0 / 16) * (xi + 1.0 / 3) * (xi - 1.0 / 3) * (xi - 1)
        N[1] = (27.0 / 16) * (xi + 1) * (xi - 1.0 / 3) * (xi - 1)
        N[2] = (-27.0 / 16) * (xi + 1) * (xi + 1.0 / 3) * (xi - 1)
        N[3] = (9.0 / 16) * (xi + 1) * (xi + 1.0 / 3) * (xi - 1.0 / 3)
    else: assert False
    return N

def calc_dNdxi_1D(xi, nNl_1d):
    dN = np.zeros(nNl_1d)
    if nNl_1d == 2:
        dN[0] = -0.5
        dN[1] = 0.5
    elif nNl_1d==3:
        dN[0] = xi - 0.5
        dN[1] = -2 * xi
        dN[2] = xi + 0.5
    elif nNl_1d==4:
        dN[0] = 1.0 / 16 * (-27 * xi * xi + 18 * xi + 1)
        dN[1] = 9.0 / 16 * (9 * xi * xi - 2 * xi - 3)
        dN[2] = -9.0 / 16 * (9 * xi * xi + 2 * xi - 3)
        dN[3] = 1.0 / 16 * (27 * xi * xi + 18 * xi - 1)
    else: assert False
    return dN


def calc_N(xi,eta, element_type):
    nNl = element_type_to_nNl[element_type]
    nNl_1d = element_type_to_nNl_1D[element_type]
    N = np.zeros(nNl)
    il_to_ij_loc = il_to_ij_loc_all[element_type]
    N1D_xi = calc_N_1D(xi,nNl_1d)
    N1D_eta = calc_N_1D(eta,nNl_1d)
    for il in range(nNl):
        i = il_to_ij_loc[il][0]
        j = il_to_ij_loc[il][1]
        N[il] = N1D_xi[i]*N1D_eta[j]
    return N


def calc_dNdxi_dNdeta(xi,eta, element_type):
    nNl = element_type_to_nNl[element_type]
    nNl_1d = element_type_to_nNl_1D[element_type]
    dNdxi = np.zeros(nNl)
    dNdeta = np.zeros(nNl)
    il_to_ij_loc = il_to_ij_loc_all[element_type]
    N1D_xi = calc_N_1D(xi,nNl_1d)
    N1D_eta = calc_N_1D(eta,nNl_1d)
    dN1D_dxi = calc_dNdxi_1D(xi,nNl_1d)
    dN1D_deta = calc_dNdxi_1D(eta,nNl_1d)
    for il in range(nNl):
        i = il_to_ij_loc[il][0]
        j = il_to_ij_loc[il][1]
        dNdxi[il] = dN1D_dxi[i]*N1D_eta[j]
        dNdeta[il] = N1D_xi[i]*dN1D_deta[j]
    
    return dNdxi,dNdeta


def calc_J(xi, eta, coord_x, coord_y, element_type):
    dNdxi,dNdeta = calc_dNdxi_dNdeta(xi,eta,element_type)
    assert len(coord_x)==len(coord_y) and len(coord_x)==len(dNdxi) and len(coord_y)==len(dNdeta)
    J = np.array([[dNdxi.dot(coord_x), dNdxi.dot(coord_y)],
                  [dNdeta.dot(coord_x), dNdeta.dot(coord_y)]])
    assert np.linalg.det(J)>SMALL_VAL
    return J

def calc_dNdx_dNdy(xi, eta, coord_x, coord_y, element_type):
    dNdxi,dNdeta = calc_dNdxi_dNdeta(xi,eta,element_type)
    J = calc_J(xi,eta,coord_x,coord_y,element_type)
    dNdXI = np.vstack((dNdxi,dNdeta))
    dNdX = np.linalg.inv(J) @ dNdXI
    return dNdX[0,:], dNdX[1,:]
    

def get_arr_xi(nQ):
    if nQ == 4:
        arr_xi = np.array([
                -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))])
    elif nQ == 3:
        arr_xi = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
    elif nQ == 2:
        arr_xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    elif nQ == 1:
        arr_xi = np.array([0])
    else:
        assert False
    assert abs(np.sum(arr_xi))<SMALL_VAL
    return arr_xi

def get_arr_w(nQ):

    if nQ == 4:
        arr_w = np.array([(18 - np.sqrt(30)) / 36,
                        (18 + np.sqrt(30)) / 36,
                        (18 + np.sqrt(30)) / 36,
                        (18 - np.sqrt(30)) / 36,])
    elif nQ == 3:
        arr_w = np.array([5/9, 8/9, 5/9])
    elif nQ == 2:
        arr_w = np.array([1, 1])
    elif nQ==1:
        arr_w = np.array([2])
    else:
        assert False
    assert abs(np.sum(arr_w)-2)<SMALL_VAL
    return arr_w
