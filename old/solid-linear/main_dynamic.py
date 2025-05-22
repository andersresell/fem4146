import numpy as np
import matplotlib.pyplot as plt
import signal

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plot_tools
import element_utils
from solid_linear import *

# fmt: off
signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == "__main__":
    nEx = 5
    nEy = 2
    nEz = 2
    E = 200e9
    nu = 0.3
    rho = 7000
    Lx = 10
    Ly = 1
    Lz = 1
    p = 100000e3 #distributed load [N/m²]
    case = "cantilever tip loaded"
    dx = Lx/nEx
    dy = Ly/nEy
    dz = Lz/nEz
    CFL = 0.9
    plot_scale = 1

    element_type = element_utils.TYPE_HEX8
    l_crit = min(dx,dy,dz)
    nNl_1D = element_utils.element_type_to_nNl_1D[element_type]
    l_crit/=(nNl_1D-1)

    nodes, ind = create_mesh_box(nEx, nEy, nEz,Lx,Ly,Lz,element_type)
    nN = nodes.shape[0]

    #supress end to model a cantilever beam
    dof_status = create_dof_status(nN)
    nodeIDs_west = get_nodesIDs_set_box("west",nEx,nEy,nEz,element_type)
    suppress_boundary_box(nodeIDs_west,dof_status,element_utils.DOF_U,nN)
    suppress_boundary_box(nodeIDs_west,dof_status,element_utils.DOF_V,nN)
    suppress_boundary_box(nodeIDs_west,dof_status,element_utils.DOF_W,nN)


    Kff,Rf, Mff, dof_to_eq_number = assemble(nodes,ind,dof_status,E,nu,element_type,rho)

    elements_bound = []
    for ez in range(nEz):
        for ey in range(nEy):
            ex=nEx-1
            e = ez*nEx*nEy + ey*nEx + ex
            elements_bound.append(e)
    if case=="cantilever tip loaded":
        t_uniform = np.array([0,0,p])
    elif case=="uniaxial tension":
        t_uniform = np.array([p,0,0])
    else: assert False
    assemble_R_consistent(nodes,Rf,dof_to_eq_number,dof_status,ind,element_type,elements_bound,t_uniform)

    nNl_1D=element_utils.element_type_to_nNl_1D[element_type]
    nNx,nNy,nNz = count_nNx_nNy_nNz(nEx,nEy,nEz,element_type)

    nodeIDs_east = get_nodesIDs_set_box("east",nEx,nEy,nEz,element_type)

    # P_i = P/len(nodeIDs_east)
    # print("P",P,"P_i",P_i)
    # if case=="cantilever tip loaded":
    #     add_point_loads_set(P_i,fem_utils.DOF_W,nodeIDs_east,Rf,dof_to_eq_number)
    # elif case=="uniaxial tension":
    #     add_point_loads_set(P_i,fem_utils.DOF_U,nodeIDs_east,Rf,dof_to_eq_number)
    # else: assert False

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    Kff_sparse = csr_matrix(Kff)
    vf = np.zeros(Kff.shape[0])
    rf = np.zeros_like(vf)


    R = get_R_from_Rf(Rf,dof_to_eq_number,nN)

    triangles,outlines = plot_tools.triangulate_quad_mesh(ind,element_type)


    c = np.sqrt(E*(1-nu)/(rho*(1+nu)*(1-2*nu)))
    dt = CFL * l_crit/ c
    t_max = 0.5
    n_timesteps = int(t_max/dt)
    #print(Mff)
    for n in range(n_timesteps):
        t = n*dt
        print("n =",n, "t =",t)

        if len(Mff.shape)==2:
            af = np.linalg.solve(Mff,-Kff@rf + Rf)
        else:
            af = (-Kff@rf + Rf)/Mff
        vf += dt*af
        rf += dt*vf


        if n%1000==0:
            r = np.zeros(len(dof_status)) #obtain the full r from rf
            for i in range(len(dof_status)):
                eq_num = dof_to_eq_number[i]
                if eq_num != element_utils.NO_EQ:
                    r[i] = rf[eq_num]
            plt.cla()
            plot_tools.plot_3d_mesh(nodes,r,triangles,outlines,plot_scale,R,False,element_type)
            plt.pause(0.01)
            #plt.show()
    print("done")
    plt.show()
