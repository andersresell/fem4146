import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mesh import *
from solver_data import *
import plot_tools

if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    t = 0.01  # Plate thickness in m
    element_type = ELEMENT_TYPE_Q9
    problem_type = PROBLEM_TYPE_PLANE_STRESS

    config = create_config(E, nu, t, element_type, problem_type)

    config.disp_scaling = 0.1

    mesh = create_structured_quad_mesh(config, Lx=2, Ly=1, nEx=10, nEy=5)

    solver_data = create_solver_data(config, mesh)

    # solver_data.R[:] = 1

    # plot_tools.plot_3d_mesh(mesh.nodes,
    #                         mesh.nodes[:, 1],
    #                         triangles,
    #                         outlines,
    #                         scale=1,
    #                         show_node_labels=True,
    #                         element_type=element_type)

    plot_tools.plot_2d(config, mesh, solver_data)

    plt.show()
