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

    # N_unique = fem_utils.get_N_unique(fem_utils.TYPE_HEX64)
    # print("N_unique")
    # for i in range(len(N_unique)):
    #     print(i,N_unique[i])
    # print("num N_unique",len(N_unique))
    # exit(0)

    nEx = 1
    nEy = 1
    nEz = 1
    E = 1 # 200e9
    nu = 0.3
    Lx = 1
    Ly = 1 # 0.05
    Lz = 1 # 0.05
    P = 10000
    p = P/(Ly*Lz)

    case = "uniaxial tension"
    case = "cantilever tip loaded"
    element_type = element_utils.TYPE_HEX8
    plot_scale = 1
    show_nodes=False

    nodes, ind = create_mesh_box(nEx, nEy, nEz,Lx,Ly,Lz,element_type)
    nN = nodes.shape[0]

    #supress end to model a cantilever beam
    dof_status = create_dof_status(nN)
    nodeIDs_west = get_nodesIDs_set_box("west",nEx,nEy,nEz,element_type)
    suppress_boundary_box(nodeIDs_west,dof_status,element_utils.DOF_U,nN)
    suppress_boundary_box(nodeIDs_west,dof_status,element_utils.DOF_V,nN)
    suppress_boundary_box(nodeIDs_west,dof_status,element_utils.DOF_W,nN)
    # suppress_boundary_box(np.array([0]),dof_status,fem_utils.DOF_V,nN)
    # suppress_boundary_box(np.array([0]),dof_status,fem_utils.DOF_W,nN)


    Kff,Rf, _, dof_to_eq_number = assemble(nodes,ind,dof_status,E,nu,element_type)

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

    r = solve(Kff,Rf,dof_status,dof_to_eq_number)
    u,_,w = get_disp_vectors_from_r(r)
    R = get_R_from_Rf(Rf,dof_to_eq_number,nN)

    if case=="cantilever tip loaded":
        I = 1/12*Ly*Lz**3
        w_tip = np.mean(w[nodeIDs_east])
        w_ana = P*Lx**3/(3*E*I)
        error_percent = 100*np.abs((w_tip-w_ana)/w_ana)
        print("Comparing solution. case: "+case)
        print("w max = "+str(w_tip)+" [m]")
        print("w analytical = "+str(w_ana)+" [m]")
        print("error = "+str(error_percent) +" %")

    elif case=="uniaxial tension":
        A = Ly*Lz
        print("u max=",np.max(u),"u_min=",np.min(u))

        u_tip = np.mean(u[nodeIDs_east])
        u_ana = P*Lx/(E*A)
        error_percent = 100*np.abs((u_tip-u_ana)/u_ana)
        print("Comparing solution. case: "+case)
        print("u max = "+str(u_tip)+" [m]")
        print("u analytical = "+str(u_ana)+" [m]")
        print("error = "+str(error_percent) +" %")
    else: assert False

    triangles,outlines = plot_tools.triangulate_quad_mesh(ind,element_type)
    plot_tools.plot_3d_mesh(nodes,r,triangles,outlines,plot_scale,R,show_nodes,element_type)

    plt.show()
