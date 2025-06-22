from src.useful_imports import *

if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 0.01  # Plate thickness in m
    element_type = ELEMENT_TYPE_Q9
    problem_type = PROBLEM_TYPE_PLANE_STRESS
    Lx = 10.0
    Ly = 0.5

    config = create_config(E, nu, h, element_type, problem_type)

    mesh = create_structured_quad_mesh(config, Lx=Lx, Ly=Ly, nEx=20, nEy=1)

    solver_data = create_solver_data(config, mesh)

    # for i in range(3 * mesh.get_nN()):
    #     if i % 3 == 0:
    #         solver_data.R_ext[i] = 1000.0  #Set a const force in z direction

    add_boundary_condition(config, mesh, "west", DOF_U, 0)
    add_boundary_condition(config, mesh, "west", DOF_V, 0)
    add_boundary_condition(config, mesh, "south_west", DOF_V, 0)

    F = 1000
    traction_tip = F / (Ly * h)
    tip_load_func = lambda x, y: (0, -traction_tip)
    add_load(config, mesh, "east", LOAD_TYPE_TRACTION, tip_load_func)

    # add_boundary_condition(config, mesh, "west", DOF_W, 0)
    # add_boundary_condition(config, mesh, "west", DOF_THETAX, 0)
    # add_boundary_condition(config, mesh, "west", DOF_THETAY, 0)

    solve(config, solver_data, mesh)

    I_beam = Ly**3 * h / 12
    v_tip_ana = abs(F) * Lx**3 / (3 * E * I_beam)
    _, v = unpack_solution(config, solver_data.r)
    v_tip = np.max(np.abs(v))
    print(
        f"Tip deflection (numerical): {v_tip:.6f}, analytical: {v_tip_ana:.6f}, relative error: {100 * (v_tip - v_tip_ana) / v_tip_ana:.4f}%"
    )

    Plot(config, mesh, solver_data)
