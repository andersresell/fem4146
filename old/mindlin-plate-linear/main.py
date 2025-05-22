import numpy as np
import matplotlib.pyplot as plt
import signal

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import plot_tools
import element_utils
from mindlin_plate_linear import *

# fmt: off
signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == "__main__":
    nEx = 8
    nEy = 1
    E = 200e9
    nu =  0.3
    h = 0.01
    Lx = 10
    Ly = 1
    p = 100 #distributed load
    m = 100 #distributed moment
    P = p*Ly
    M = m*Ly
    case = "tip moment"
    case = "tip force"
    element_type = element_utils.TYPE_Q16
    plot_scale = 1
    show_node_labels=True

    nodes, ind = create_mesh_plate(nEx, nEy, Lx, Ly, element_type)


    #supress end to model a cantilever beam
    dof_status = create_dof_status(nN= nodes.shape[0])
    suppress_boundary(dof_status, "west", element_utils.DOF_W, nEx, nEy, element_type)
    suppress_boundary(dof_status, "west", element_utils.DOF_THETAY, nEx, nEy, element_type)
    suppress_boundary(dof_status, "west", element_utils.DOF_THETAX, nEx, nEy, element_type)

    # for i in range(nodes.shape[0]):
    #     dof_status[3*i+fem_utils.DOF_THETAX] = fem_utils.DOF_SUPPRESSED

    Kff,Rf, dof_to_eq_number = assemble(nodes, ind, dof_status, h,E,nu,element_type)

    #set "point load" at end to model a tip loaded cantilever
    nNl_1D=element_utils.element_type_to_nNl_1D[element_type]
    nNy = nEy*(nNl_1D-1)+1
    nNx = nEx*(nNl_1D-1)+1

    load_type = "lumped"
    load_type = "consistent"
    if load_type == "lumped":
        for j in range(nNy):
            I = nNx*j + nNx-1
            if case == "tip force":
                add_point_load(P/nNy,element_utils.DOF_W,I, Rf,dof_to_eq_number)
            elif case == "tip moment":
                add_point_load(M/nNy,element_utils.DOF_THETAY,I, Rf,dof_to_eq_number)
            else: assert False
    else:
        assert load_type== "consistent"
        elements_bound = []
        for ey in range(nEy):
            ex=nEx-1
            e = ey*nEx + ex
            elements_bound.append(e)
        if case=="tip force":
            t_uniform = p
        else: assert False
        assemble_R_consistent(nodes,Rf,dof_to_eq_number,dof_status,ind,element_type,elements_bound,t_uniform)

    print("Rf sum =", np.sum(Rf))
    print("p*Ly =", p*Ly)

    r = solve(Kff,Rf,dof_status,dof_to_eq_number)
    w,thetax,thetay = get_disp_vectors_from_r(r)

    print("thetax\n", thetax)

    print("thetay\n", thetay)


    Rw,_,_ = get_applied_forces_and_moments_from_Rf(Rf,dof_to_eq_number,nN=nodes.shape[0])



    I = 1/12*Ly*h**3
    w_tip = np.max(w)
    if case == "tip force":
        w_ana = P*Lx**3/(3*E*I)
    elif case == "tip moment":
        w_ana = M*Lx**2/(2*E*I)
    else: assert False
    error_percent = 100*np.abs((w_tip-w_ana)/w_ana)

    print("Comparing beam solution. case: "+case)
    print("w max = "+str(w_tip)+" [m]")
    print("w analytical = "+str(w_ana)+" [m]")
    print("error = "+str(error_percent) +" %")

    triangles,outlines = plot_tools.triangulate_quad_mesh(ind,element_type)
    plot_tools.plot_3d_mesh(nodes,w,triangles,outlines, scale=plot_scale, Rw=Rw, show_node_labels=show_node_labels,element_type=element_type)

    plt.show()
