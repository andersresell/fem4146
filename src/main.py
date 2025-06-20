import numpy as np
import matplotlib.pyplot as plt
from utils import *
from mesh import *
from solver_data import *
from solver import solve
import plot_tools
from bcs import add_boundary_condition

if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 0.01  # Plate thickness in m
    element_type = ELEMENT_TYPE_Q8
    problem_type = PROBLEM_TYPE_PLANE_STRESS

    config = create_config(E, nu, h, element_type, problem_type)

    mesh = create_structured_quad_mesh(config, Lx=5, Ly=1, nEx=40, nEy=10)

    solver_data = create_solver_data(config, mesh)

    solver_data.R_ext[:] = 1000

    add_boundary_condition(config, mesh, "west", DOF_U, 0)
    add_boundary_condition(config, mesh, "south_west", DOF_V, 0)

    solve(config, solver_data, mesh)

    plot_tools.Plot(config, mesh, solver_data)

    plt.show()
