import numpy as np
import matplotlib.pyplot as plt
import signal

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plot_tools
import fem_utils

# fmt: off
signal.signal(signal.SIGINT, signal.SIG_DFL)


def create_mesh_plate(nEx, nEy,Lx,Ly, element_type):
    nE = nEx*nEy
    nNl = fem_utils.element_type_to_nNl[element_type]
    nNl_1D = fem_utils.element_type_to_nNl_1D[element_type]
    nNx = (nNl_1D-1)*nEx+1 #this only makes sense for lagrangian shape funcs
    nNy = (nNl_1D-1)*nEy+1
    nN = nNx*nNy
    dx = Lx/(nNx-1)
    dy = Ly/(nNy-1)

    ind = np.zeros((nE,nNl),dtype=int)
    nodes = np.zeros((nN,2))

    il_to_ij_loc = fem_utils.il_to_ij_loc_all[element_type]
    for j in range(nNy):
        for i in range(nNx):
            x = i*dx
            y = j*dy
            nodes[j*nNx+i,0] = x
            nodes[j*nNx+i,1] = y
    for ex in range(nEx):
        for ey in range(nEy):
            e = ey*nEx + ex
            assert len(il_to_ij_loc)==nNl
            for il in range(nNl):
                i = (nNl_1D-1)*ex + il_to_ij_loc[il][0]
                j = (nNl_1D-1)*ey + il_to_ij_loc[il][1]
                ind[e,il] = j*nNx+i

    return nodes,ind

def count_nNx_nNy(nEx,nEy, element_type):
    nNl_1d = fem_utils.element_type_to_nNl_1D[element_type]
    nNx = (nNl_1d-1)*nEx+1
    nNy = (nNl_1d-1)*nEy+1
    return nNx,nNy

def create_dof_status(nN):
    return np.full(3*nN,fem_utils.DOF_FREE, dtype=int)

def suppress_boundary(dof_status, boundary_str, dof_loc, nEx,nEy, element_type):
    nNl = fem_utils.element_type_to_nNl[element_type]
    nNx,nNy = count_nNx_nNy(nEx,nEy,element_type)
    nN = nNx*nNy
    assert nN*3==len(dof_status)
    assert dof_loc==fem_utils.DOF_W or dof_loc==fem_utils.DOF_THETAX or dof_loc == fem_utils.DOF_THETAY

    nodes_bound = []
    if boundary_str=="west":
        i = 0
        for j in range(nNy):
            I = j*nNx+i
            nodes_bound.append(I)
    elif boundary_str=="east":
        i = nNx-1
        for j in range(nNy):
            I = j*nNx+i
            nodes_bound.append(I)
    elif boundary_str=="south":
        j = 0
        for i in range(nNx):
            I = j*nNx+i
            nodes_bound.append(I)
    else:
        assert boundary_str=="north"
        j = nNy-1
        for i in range(nNx):
            I = j*nNx+i
            nodes_bound.append(I)

    for I in nodes_bound:
        dof_status[3*I+dof_loc] = fem_utils.DOF_SUPPRESSED

def add_point_load(val,dof_loc, I, Rf,dof_to_eq_number):
    nN = int(len(dof_to_eq_number)/3)
    assert 3*nN == len(dof_to_eq_number)
    assert I<nN
    assert dof_loc>=0 and dof_loc<3
    dof = 3*I+dof_loc
    eq = dof_to_eq_number[dof]
    if eq != fem_utils.NO_EQ:
        Rf[eq] += val

def calc_Ke(coord_x, coord_y, element_type, h, nu, E):
    nNl = fem_utils.element_type_to_nNl[element_type]
    assert len(coord_x)==nNl and len(coord_y) == nNl

    D = E*h**3/(12*(1-nu**2))*np.array([[1,nu,0],
                                        [nu,1,0],
                                        [0,0,(1-nu)/2]])
    Cs = E/(2*(1+nu))*np.identity(2)
    hs = 5/6*h


    nQ = fem_utils.element_type_to_nQ[element_type]
    arr_xi = fem_utils.get_arr_xi(nQ)
    arr_w = fem_utils.get_arr_w(nQ)


    Ke = np.zeros((3*nNl, 3*nNl))

    for i in range(nQ):
        for j in range(nQ):
            xi = arr_xi[i]
            eta = arr_xi[j]
            weight = arr_w[i]*arr_w[j]
            N = fem_utils.calc_N(xi, eta, element_type)
            dNdx,dNdy = fem_utils.calc_dNdx_dNdy(xi, eta, coord_x,coord_y,element_type)
            J = fem_utils.calc_J(xi,eta,coord_x,coord_y,element_type)
            detJ = np.linalg.det(J)
            assert len(N) == nNl and len(dNdx)==nNl and len(dNdy) == nNl
            Bb = np.zeros((3,3*nNl))
            Bs = np.zeros((2,3*nNl))
            for k in range(nNl):
                Bb[:,3*k:3*k+3] = np.array([[0, 0,      -dNdx[k]],
                                            [0, dNdy[k], 0      ],
                                            [0, dNdx[k], -dNdy[k]]])
                Bs[:,3*k:3*k+3] = np.array([[dNdy[k], -N[k], 0],
                                            [dNdx[k], 0, N[k]]])
            Ke += (Bb.T @ D @ Bb + hs * Bs.T @ Cs @ Bs)* detJ * weight

# print(np.linalg.norm(Ke-Ke.T))
    assert np.linalg.norm(Ke-Ke.T)<1e-3 #check that Ke is symmetric

    return Ke


def assemble(nodes,ind,dof_status, h,E,nu,element_type):
    nNl = fem_utils.element_type_to_nNl[element_type]
    nE = ind.shape[0]
    nN = nodes.shape[0]
    assert ind.shape[1] == nNl
    assert nodes.shape[1] == 2
    assert len(dof_status) == 3*nN

    #count number of equations
    n_eqs = 0
    for status in dof_status:
        if status == fem_utils.DOF_FREE:
            n_eqs += 1


    dof_to_eq_number = np.full(len(dof_status),fem_utils.NO_EQ)
    eq_counter=0
    for i in range(len(dof_status)):
        if dof_status[i] == fem_utils.DOF_FREE:
            dof_to_eq_number[i] = eq_counter
            eq_counter+=1
    assert eq_counter==n_eqs

    Kff = np.zeros((n_eqs, n_eqs)) #should rather be sparse
    Rf = np.zeros(n_eqs)

    for e in range(nE):
        coord_x = np.zeros(nNl)
        coord_y = np.zeros(nNl)
        for i in range(nNl):
            I = ind[e,i]
            coord_x[i] = nodes[I,0]
            coord_y[i] = nodes[I,1]

        Ke = calc_Ke(coord_x,coord_y,element_type,h,nu,E)

        for i in range(nNl):
            for j in range(nNl):
                I = ind[e,i]
                J = ind[e,j]
                for dof_i in range(3):
                    for dof_j in range(3):
                        dof_I = 3*I+dof_i
                        dof_J = 3*J+dof_j
                        EQ_I = dof_to_eq_number[dof_I]
                        EQ_J = dof_to_eq_number[dof_J]
                        if EQ_I != fem_utils.NO_EQ and EQ_J != fem_utils.NO_EQ:
                            assert dof_status[dof_I] == fem_utils.DOF_FREE and dof_status[dof_J] == fem_utils.DOF_FREE
                            Kff[EQ_I, EQ_J] += Ke[3*i+dof_i, 3*j+dof_j]
    print("assembly complete")
    return Kff,Rf, dof_to_eq_number

def solve(Kff, Rf, dof_status, dof_to_eq_number):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    Kff_sparse = csr_matrix(Kff)
    rf = spsolve(Kff_sparse,Rf)

    # rf = np.linalg.solve(Kff,Rf)

    r = np.zeros(len(dof_status))
    for i in range(len(dof_status)):
        eq_num = dof_to_eq_number[i]
        if eq_num != fem_utils.NO_EQ:
            r[i] = rf[eq_num]
    print("System solved")
    return r

def get_disp_vectors_from_r(r):
    nN = int(len(r)/3)
    assert 3*nN== len(r)
    w = r[0::3]
    thetax = r[1::3]
    thetay = r[2::3]
    assert len(w)==nN
    return w,thetax,thetay


def get_applied_forces_and_moments_from_Rf(Rf, dof_to_eq_number,nN):
    assert(len(dof_to_eq_number))==3*nN
    R=np.zeros(3*nN)
    for i in range(3*nN):
        eq_num = dof_to_eq_number[i]
        if eq_num != fem_utils.NO_EQ:
            R[i] = Rf[eq_num]

    Rw = R[0::3]
    Rthetax = R[1::3]
    Rthetay = R[2::3]
    return Rw,Rthetax,Rthetay



def calc_Re_ext_consistent_tip_shitty( coord_y, element_type,  t_uniform):
    nNl = fem_utils.element_type_to_nNl[element_type]


    nQ = fem_utils.element_type_to_nQ[element_type]
    arr_xi = fem_utils.get_arr_xi(nQ)
    arr_w = fem_utils.get_arr_w(nQ)
    dy = np.max(coord_y)-np.min(coord_y)
    Re = np.zeros(3*nNl)
    for j in range(nQ):
        xi0 = 1.0
        xi1 = arr_xi[j]
        weight = arr_w[j]
        N = fem_utils.calc_N(xi0,xi1,element_type)
        J_surf = dy/2
        for l in range(nNl):
            Re[3*l] += N[l]*weight*J_surf*t_uniform
    return Re


def assemble_R_consistent(nodes,Rf, dof_to_eq_number,dof_status, ind,element_type, elements, t_uniform):
    nNl = fem_utils.element_type_to_nNl[element_type]
    for e in elements:
        coord_x = np.zeros(nNl)
        coord_y = np.zeros(nNl)
        for il in range(nNl):
            I = ind[e,il]
            coord_x[il] = nodes[I,0]
            coord_y[il] = nodes[I,1]

        Re = calc_Re_ext_consistent_tip_shitty(coord_y, element_type, t_uniform)
        for i in range(nNl):
            I = ind[e,i]
            for dof_i in range(3):
                dof_I = 3*I+dof_i
                EQ_I = dof_to_eq_number[dof_I]
                if EQ_I != fem_utils.NO_EQ:
                    assert dof_status[dof_I] == fem_utils.DOF_FREE
                    Rf[EQ_I] += Re[3*i+dof_i]
