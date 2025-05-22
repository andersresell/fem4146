import numpy as np

SMALL_VAL = 1e-8

NO_EQ = -1    

DOF_U = 0
DOF_V = 1
DOF_W = 2

DOF_FREE = 0
DOF_SUPPRESSED = 1

TYPE_HEX8 = 0
TYPE_HEX27 = 1
TYPE_HEX64 = 2

GAUSS_QUADRATURE=0
LOBOTTO_QUADRATURE = 1

element_type_to_str =   {TYPE_HEX8: "HEX8",
                         TYPE_HEX27: "HEX27",
                         TYPE_HEX64: "HEX64"}


element_type_to_nNl = {TYPE_HEX8: 8,
                       TYPE_HEX27: 27,
                       TYPE_HEX64: 64}

element_type_to_nNl_1D = {TYPE_HEX8: 2,
                          TYPE_HEX27: 3,
                          TYPE_HEX64: 4}

element_type_to_nQ = {TYPE_HEX8: 2,
                      TYPE_HEX27: 3,
                      TYPE_HEX64: 4}


il_to_ijk_loc_all ={
    TYPE_HEX8: [(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)],      
    TYPE_HEX27: [(0,0,0),(1,0,0),(2,0,0),(0,1,0),(1,1,0),(2,1,0),(0,2,0),(1,2,0),(2,2,0), #0-7
                 (0,0,1),(1,0,1),(2,0,1),(0,1,1),(1,1,1),(2,1,1),(0,2,1),(1,2,1),(2,2,1), #8-15
                 (0,0,2),(1,0,2),(2,0,2),(0,1,2),(1,1,2),(2,1,2),(0,2,2),(1,2,2),(2,2,2)],  #16-26
    TYPE_HEX64: [(0,0,0),(1,0,0),(2,0,0),(3,0,0),(0,1,0),(1,1,0),(2,1,0),(3,1,0),(0,2,0),(1,2,0),(2,2,0),(3,2,0),(0,3,0),(1,3,0),(2,3,0),(3,3,0), #0-15
                 (0,0,1),(1,0,1),(2,0,1),(3,0,1),(0,1,1),(1,1,1),(2,1,1),(3,1,1),(0,2,1),(1,2,1),(2,2,1),(3,2,1),(0,3,1),(1,3,1),(2,3,1),(3,3,1), #16-31
                 (0,0,2),(1,0,2),(2,0,2),(3,0,2),(0,1,2),(1,1,2),(2,1,2),(3,1,2),(0,2,2),(1,2,2),(2,2,2),(3,2,2),(0,3,2),(1,3,2),(2,3,2),(3,3,2), #32-47
                 (0,0,3),(1,0,3),(2,0,3),(3,0,3),(0,1,3),(1,1,3),(2,1,3),(3,1,3),(0,2,3),(1,2,3),(2,2,3),(3,2,3),(0,3,3),(1,3,3),(2,3,3),(3,3,3)] #48-63
    }



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


def calc_N(xi0,xi1,xi2, element_type):
    nNl = element_type_to_nNl[element_type]
    nNl_1d = element_type_to_nNl_1D[element_type]
    N = np.zeros(nNl)
    il_to_ijk_loc = il_to_ijk_loc_all[element_type]
    N1D_xi0 = calc_N_1D(xi0,nNl_1d)
    N1D_xi1 = calc_N_1D(xi1,nNl_1d)
    N1D_xi2 = calc_N_1D(xi2,nNl_1d)
    for il in range(nNl):
        i = il_to_ijk_loc[il][0]
        j = il_to_ijk_loc[il][1]
        k = il_to_ijk_loc[il][2]
        N[il] = N1D_xi0[i]*N1D_xi1[j]*N1D_xi2[k]
    return N

    

def calc_dNdxi0_dNdxi1_dNdxi2(xi0,xi1,xi2, element_type):
    nNl = element_type_to_nNl[element_type]
    nNl_1d = element_type_to_nNl_1D[element_type]
    dNdxi0 = np.zeros(nNl)
    dNdxi1 = np.zeros(nNl)
    dNdxi2 = np.zeros(nNl)
    il_to_ijk_loc = il_to_ijk_loc_all[element_type]
    N1D_xi0 = calc_N_1D(xi0,nNl_1d)
    N1D_xi1 = calc_N_1D(xi1,nNl_1d)
    N1D_xi2 = calc_N_1D(xi2,nNl_1d)
    dN1D_dxi0 = calc_dNdxi_1D(xi0,nNl_1d)
    dN1D_dxi1 = calc_dNdxi_1D(xi1,nNl_1d)
    dN1D_dxi2 = calc_dNdxi_1D(xi2,nNl_1d)
    for il in range(nNl):
        i = il_to_ijk_loc[il][0]
        j = il_to_ijk_loc[il][1]
        k = il_to_ijk_loc[il][2]
        dNdxi0[il] = dN1D_dxi0[i]*N1D_xi1[j]*N1D_xi2[k]
        dNdxi1[il] = N1D_xi0[i]*dN1D_dxi1[j]*N1D_xi2[k]
        dNdxi2[il] = N1D_xi0[i]*N1D_xi1[j]*dN1D_dxi2[k]
    
    return dNdxi0,dNdxi1,dNdxi2


def calc_J(xi0, xi1, xi2, coord_x, coord_y,coord_z, element_type):
    dNdxi0,dNdxi1,dNdxi2 = calc_dNdxi0_dNdxi1_dNdxi2(xi0,xi1,xi2,element_type)
    assert len(coord_x)==len(coord_y)==len(coord_z) and len(coord_x)==len(dNdxi0)==len(dNdxi1)==len(dNdxi2)
    J = np.array([[dNdxi0.dot(coord_x), dNdxi0.dot(coord_y),dNdxi0.dot(coord_z)],
                  [dNdxi1.dot(coord_x), dNdxi1.dot(coord_y),dNdxi1.dot(coord_z)],
                  [dNdxi2.dot(coord_x), dNdxi2.dot(coord_y),dNdxi2.dot(coord_z)]])
    assert np.linalg.det(J)>SMALL_VAL
    return J

def calc_dNdx_dNdy_dNdz(xi0, xi1, xi2, coord_x, coord_y, coord_z, element_type):
    dNdxi0,dNdxi1,dNdxi2 = calc_dNdxi0_dNdxi1_dNdxi2(xi0,xi1,xi2,element_type)
    J = calc_J(xi0,xi1,xi2,coord_x,coord_y,coord_z,element_type)
    dNdXI = np.vstack((dNdxi0,dNdxi1,dNdxi2))
    dNdX = np.linalg.inv(J) @ dNdXI
    return dNdX[0,:], dNdX[1,:], dNdX[2,:]
    

def get_gauss_rule(nQ):
    if nQ == 4:
        arr_xi = np.array([
                -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5))])
        arr_w = np.array([(18 - np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 + np.sqrt(30)) / 36,
                (18 - np.sqrt(30)) / 36,])
    elif nQ == 3:
        arr_xi = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)])
        arr_w = np.array([5/9, 8/9, 5/9])
    elif nQ == 2:
        arr_xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        arr_w = np.array([1, 1])
    elif nQ == 1:
        arr_xi = np.array([0])
        arr_w = np.array([2])
    else:
        assert False
    assert abs(np.sum(arr_w)-2)<SMALL_VAL
    assert abs(np.sum(arr_xi))<SMALL_VAL
    return arr_xi, arr_w


def get_lobatto_rule(nQ):
    if nQ == 2:
        arr_xi = np.array([-1, 1])
        arr_w = np.array([1, 1])
    elif nQ == 3:
        arr_xi = np.array([-1, 0, 1])
        arr_w = np.array([1/3, 4/3, 1/3])
    elif nQ == 4:
        arr_xi = np.array([-1, -np.sqrt(5)/5, np.sqrt(5)/5, 1])
        arr_w = np.array([1/6, 5/6, 5/6, 1/6])
    else: assert False
    assert abs(np.sum(arr_xi))<SMALL_VAL
    assert abs(np.sum(arr_w)-2)<SMALL_VAL
    return arr_xi,arr_w

def get_N_unique(element_type):
    N_unique = []
    nQ = element_type_to_nQ[element_type]
    arr_xi, _ = get_gauss_rule(nQ)
    for i in range(nQ):
        for j in range(nQ):
            for k in range(nQ):
                xi0 = arr_xi[i]
                xi1 = arr_xi[j]
                xi2 = arr_xi[k]
                N = calc_N(xi0,xi1,xi2,element_type)
                for il in range(len(N)):
                    if N_unique==[]:
                        N_unique.append(N[il])
                    equal_found = False
                    for jl in range(len(N_unique)):
                        if abs(N_unique[jl]-N[il])<=1e-16:
                            equal_found = True
                            break   
                    if not equal_found:
                        N_unique.append(N[il])
                    
    min_diff=1e10
    for i in range(len(N_unique)):
        for j in range(i+1,len(N_unique)):
            diff = abs(N_unique[i]-N_unique[j])
            if diff<min_diff:
                i_min = i
                j_min = j
                min_diff = diff
    print("min_diff",min_diff)
    print("N_unique[i_min]",N_unique[i_min])    
    print("N_unique[j_min]",N_unique[j_min])
    return N_unique
                    