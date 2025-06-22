import numpy as np
import matplotlib.pyplot as plt
import signal

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plot_tools
import fem_utils

# fmt: off
signal.signal(signal.SIGINT, signal.SIG_DFL)

def IX_box(i,j,k, nNx,nNy,nNz):
    assert i>=0 and j>=0 and k>=0
    assert i<nNx and j<nNy and k<nNz
    ID = k*nNx*nNy + j*nNx + i
    return ID

def create_mesh_box(nEx, nEy,nEz, Lx,Ly,Lz, element_type):
    nE = nEx*nEy*nEz
    nNl = fem_utils.element_type_to_nNl[element_type]
    nNl_1D = fem_utils.element_type_to_nNl_1D[element_type]
    nNx = (nNl_1D-1)*nEx+1 #this only makes sense for lagrangian shape funcs
    nNy = (nNl_1D-1)*nEy+1
    nNz = (nNl_1D-1)*nEz+1

    nN = nNx*nNy*nNz
    dx = Lx/(nNx-1)
    dy = Ly/(nNy-1)
    dz = Lz/(nNz-1)

    ind = np.zeros((nE,nNl),dtype=int)
    nodes = np.zeros((nN,3))

    il_to_ijk_loc = fem_utils.il_to_ijk_loc_all[element_type]
    for k in range(nNz):
        for j in range(nNy):
            for i in range(nNx):
                x = i*dx
                y = j*dy
                z = k*dz
                I = IX_box(i,j,k,nNx,nNy,nNz)
                nodes[I,:] = np.array([x,y,z])
    for ez in range(nEz):
        for ey in range(nEy):
            for ex in range(nEx):
                e = ez*nEx*nEy + ey*nEx + ex
                assert len(il_to_ijk_loc)==nNl
                for il in range(nNl):
                    i = (nNl_1D-1)*ex + il_to_ijk_loc[il][0]
                    j = (nNl_1D-1)*ey + il_to_ijk_loc[il][1]
                    k = (nNl_1D-1)*ez + il_to_ijk_loc[il][2]
                    ind[e,il] = IX_box(i,j,k,nNx,nNy,nNz)
    return nodes,ind

def count_nNx_nNy_nNz(nEx,nEy,nEz, element_type):
    nNl_1d = fem_utils.element_type_to_nNl_1D[element_type]
    nNx = (nNl_1d-1)*nEx+1
    nNy = (nNl_1d-1)*nEy+1
    nNz = (nNl_1d-1)*nEz+1
    return nNx,nNy,nNz

def create_dof_status(nN):
    return np.full(3*nN,fem_utils.DOF_FREE, dtype=int)

def get_nodesIDs_set_box(boundary_str, nEx,nEy,nEz, element_type):
    nNx,nNy,nNz = count_nNx_nNy_nNz(nEx,nEy,nEz, element_type)
    nN = nNx*nNy*nNz
    nodes_bound = []
    if boundary_str=="west":
        i = 0
        for j in range(nNy):
            for k in range(nNz):
                I = IX_box(i,j,k,nNx,nNy,nNz)
                nodes_bound.append(I)
    elif boundary_str=="east":
        i = nNx-1
        for j in range(nNy):
            for k in range(nNz):
                I = IX_box(i,j,k,nNx,nNy,nNz)
                nodes_bound.append(I)
    elif boundary_str=="south":
        j = 0
        for i in range(nNx):
            for k in range(nNz):
                I = IX_box(i,j,k,nNx,nNy,nNz)
                nodes_bound.append(I)
    else:
        assert boundary_str=="north"
        j = nNy-1
        for i in range(nNx):
            for k in range(nNz):
                I = IX_box(i,j,k,nNx,nNy,nNz)
                nodes_bound.append(I)
    return np.array(nodes_bound)

def suppress_boundary_box(nodeIDs, dof_status, dof_loc,nN):
    assert nN*3==len(dof_status)
    assert dof_loc==fem_utils.DOF_U or dof_loc==fem_utils.DOF_V or dof_loc == fem_utils.DOF_W
    for I in nodeIDs:
        assert I<nN
        dof_status[3*I+dof_loc] = fem_utils.DOF_SUPPRESSED

def add_point_loads_set(val,dof_loc, nodeIDs, Rf,dof_to_eq_number):
    nN = int(len(dof_to_eq_number)/3)
    assert 3*nN == len(dof_to_eq_number)
    assert dof_loc>=0 and dof_loc<3
    for I in nodeIDs:
        assert I<nN
        dof = 3*I+dof_loc
        eq = dof_to_eq_number[dof]
        if eq != fem_utils.NO_EQ:
            Rf[eq] += val

def calc_Ke(coord_x, coord_y, coord_z, element_type, nu, E):
    nNl = fem_utils.element_type_to_nNl[element_type]
    assert len(coord_x)==nNl and len(coord_y) == nNl

    G = E/((2*(1+nu)))
    Lambda = nu*E/((1+nu)*(1-2*nu))
    C = np.array([[Lambda+2*G, Lambda, Lambda, 0, 0, 0],
                  [Lambda, Lambda+2*G, Lambda, 0, 0, 0],
                  [Lambda, Lambda, Lambda+2*G, 0, 0, 0],
                  [0, 0, 0, G, 0, 0],
                  [0, 0, 0, 0, G, 0],
                  [0, 0, 0, 0, 0, G]])

    # print("C\n",C)

    nQ = fem_utils.element_type_to_nQ[element_type]
    arr_xi, arr_w = fem_utils.get_gauss_rule(nQ)

    Ke = np.zeros((3*nNl, 3*nNl))

    for i in range(nQ):
        for j in range(nQ):
            for k in range(nQ):
                xi0 = arr_xi[i]
                xi1 = arr_xi[j]
                xi2 = arr_xi[k]
                weight = arr_w[i]*arr_w[j]*arr_w[k]
                dNdx,dNdy,dNdz = fem_utils.calc_dNdx_dNdy_dNdz(xi0, xi1, xi2, coord_x,coord_y,coord_z, element_type)
                J = fem_utils.calc_J(xi0,xi1,xi2,coord_x,coord_y,coord_z, element_type)
                detJ = np.linalg.det(J)
                # print("dNdx",dNdx)
                # print("detJ",detJ)
                assert len(dNdx)==nNl and len(dNdy) == nNl and len(dNdz) == nNl
                B = np.zeros((6,3*nNl))

                for l in range(nNl):
                    B[:,3*l:3*l+3] = np.array([[dNdx[l], 0, 0],
                                               [0, dNdy[l], 0],
                                               [0, 0, dNdz[l]],
                                               [dNdy[l],dNdx[l],0],
                                               [0,dNdz[l],dNdy[l]],
                                               [dNdz[l],0,dNdx[l]]])
                Ke += B.T @ C @ B* detJ * weight
    # print("KE\n",Ke[0:6,0:6])
    max_value = np.max(np.abs(Ke))
    scaled_tolerance = max_value * 1e-6
    assert np.allclose(Ke,Ke.T, atol=scaled_tolerance)
    return Ke

def calc_Me_consistent(coord_x,coord_y,coord_z,rho, element_type, integration_type):
    nNl = fem_utils.element_type_to_nNl[element_type]
    nQ = fem_utils.element_type_to_nQ[element_type]
    if integration_type==fem_utils.GAUSS_QUADRATURE:
        arr_xi,arr_w = fem_utils.get_gauss_rule(nQ)
    elif integration_type==fem_utils.LOBOTTO_QUADRATURE:
        arr_xi,arr_w = fem_utils.get_lobatto_rule(nQ)
    else: assert False
    Me = np.zeros((nNl,nNl))

    for i in range(nQ):
        for j in range(nQ):
            for k in range(nQ):
                xi0 = arr_xi[i]
                xi1 = arr_xi[j]
                xi2 = arr_xi[k]
                weigth = arr_w[i]*arr_w[j]*arr_w[k]
                J = fem_utils.calc_J(xi0,xi1,xi2,coord_x,coord_y,coord_z,element_type)
                detJ = np.linalg.det(J)
                Ncompact = fem_utils.calc_N(xi0,xi1,xi2,element_type)
                # N = np.zeros(3*nNl)
                # for il in range(nNl):
                #     N[3*il:3*il+3] = Ncompact[il] * np.ones(3)
                # Me += N.T@N * rho* detJ * weigth
                Me += Ncompact.T@Ncompact * rho* detJ * weigth
    return Me

def calc_Me_lumped_row_sum(coord_x,coord_y,coord_z,rho, element_type, integration_type):
    Me = calc_Me_consistent(coord_x,coord_y,coord_z,rho,element_type,integration_type)
    nNl = fem_utils.element_type_to_nNl[element_type]
    Me_lumped = np.zeros(nNl)
    for i in range(nNl):
        for j in range(nNl):
            Me_lumped[i] += Me[i,j]
    assert np.abs(Me_lumped.all())>fem_utils.SMALL_VAL
    return Me_lumped


def calc_Re_ext_consistent_shitty(coord_x, coord_y, coord_z,element_type,xi_comp,  t_uniform):
    nNl = fem_utils.element_type_to_nNl[element_type]

    assert xi_comp==0 or xi_comp==1 or xi_comp==2
    assert len(t_uniform)==3

    assert xi_comp==0 #only testing for this now
    nQ = fem_utils.element_type_to_nQ[element_type]
    arr_xi,arr_w= fem_utils.get_gauss_rule(nQ)
    dy = np.max(coord_y)-np.min(coord_y)
    dz = np.max(coord_z) - np.min(coord_z)
    Re = np.zeros(3*nNl)
    for j in range(nQ):
        for k in range(nQ):
            xi1 = arr_xi[j]
            xi2 = arr_xi[k]
            weight = arr_w[j]*arr_w[k]
            xi0 = 1.0
            N = fem_utils.calc_N(xi0,xi1,xi2,element_type)
            J_surf = dz*dy/4
            for l in range(nNl):
                Re[3*l:3*l+3] += N[l]*weight*J_surf*t_uniform
    return Re

def assemble_R_consistent(nodes,Rf, dof_to_eq_number,dof_status, ind,element_type, elements, t_uniform):
    nNl = fem_utils.element_type_to_nNl[element_type]
    for e in elements:
        coord_x = np.zeros(nNl)
        coord_y = np.zeros(nNl)
        coord_z = np.zeros(nNl)
        for il in range(nNl):
            I = ind[e,il]
            coord_x[il] = nodes[I,0]
            coord_y[il] = nodes[I,1]
            coord_z[il] = nodes[I,2]

        Re = calc_Re_ext_consistent_shitty(coord_x,coord_y,coord_z,element_type,0,t_uniform)
        for i in range(nNl):
            I = ind[e,i]
            for dof_i in range(3):
                dof_I = 3*I+dof_i
                EQ_I = dof_to_eq_number[dof_I]
                if EQ_I != fem_utils.NO_EQ:
                    assert dof_status[dof_I] == fem_utils.DOF_FREE
                    Rf[EQ_I] += Re[3*i+dof_i]

def assemble(nodes,ind,dof_status,E,nu,element_type, rho=0,mass_integration_type=0):
    nNl = fem_utils.element_type_to_nNl[element_type]
    nE = ind.shape[0]
    nN = nodes.shape[0]
    assert ind.shape[1] == nNl
    assert nodes.shape[1] == 3
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
    calc_mass = True if rho > 0 else False

    consistent_mass=True

    if consistent_mass:
        Mff = np.zeros((n_eqs,n_eqs))
    else:
        Mff = np.zeros(n_eqs)

    for e in range(nE):
        coord_x = np.zeros(nNl)
        coord_y = np.zeros(nNl)
        coord_z = np.zeros(nNl)
        for i in range(nNl):
            I = ind[e,i]
            coord_x[i] = nodes[I,0]
            coord_y[i] = nodes[I,1]
            coord_z[i] = nodes[I,2]

        Ke = calc_Ke(coord_x,coord_y,coord_z, element_type,nu,E)
        if calc_mass:
            if consistent_mass:
                Me = calc_Me_consistent(coord_x,coord_y,coord_z,rho,element_type,mass_integration_type)
            else:
                Me = calc_Me_lumped_row_sum(coord_x,coord_y,coord_z,rho,element_type,mass_integration_type)
        for i in range(nNl):
            I = ind[e,i]
            for j in range(nNl):
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

                        if calc_mass:
                            if consistent_mass:
                                Mff[EQ_I,EQ_I] += Me[i,j]
                            else:
                                if J==0 and dof_j==0:
                                    Mff[EQ_I] += Me[i]



    print("assembly complete")
    return Kff,Rf, Mff, dof_to_eq_number

def solve(Kff, Rf, dof_status, dof_to_eq_number):
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve
    Kff_sparse = csr_matrix(Kff)
    rf = spsolve(Kff_sparse,Rf)

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
    u = r[0::3]
    v = r[1::3]
    w = r[2::3]
    assert len(w)==nN
    return u,v,w


def get_R_from_Rf(Rf, dof_to_eq_number,nN):
    assert(len(dof_to_eq_number))==3*nN
    R=np.zeros(3*nN)
    for i in range(3*nN):
        eq_num = dof_to_eq_number[i]
        if eq_num != fem_utils.NO_EQ:
            R[i] = Rf[eq_num]

    return R
