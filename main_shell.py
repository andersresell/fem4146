from src.useful_imports import *  #import required functions

if __name__ == "__main__":

    E = 200e9
    nu = 0.3
    h = 0.001
    Lx = 1.0
    Ly = 0.001
    P = 10
    element_type = ELEMENT_TYPE_Q16
    problem_type = PROBLEM_TYPE_PLATE
    nEx = 10
    nEy = 1

    config = create_config(E, nu, h, element_type, problem_type)
    mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)

    add_boundary_condition(config, mesh, "west", DOF_W, 0)
    add_boundary_condition(config, mesh, "south_west", DOF_THETAX, 0)
    add_boundary_condition(config, mesh, "west", DOF_THETAY, 0)

    load_func = lambda x, y: P / (h * Ly)
    add_load(config, mesh, "east", LOAD_TYPE_TRACTION, load_func)

    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    Rz, Rthetax, Rthetay = unpack_solution(config, solver_data.R_ext)
    # print("Rthtax\n", Rthetax)
    # print("Rthtay\n", Rthetay)
    # print("Rz\n", Rz)

    # print("K\n", solver_data.K.toarray())
    # print("A\n", solver_data.A.toarray())

    w, _, _ = unpack_solution(config, solver_data.r)
    w_east = w[mesh.node_sets["east"]]
    w_tip = np.max(w)  # np.mean(w_east)

    I = h**3 * Ly / 12
    print("P", P)
    print("R_ext", np.sum(solver_data.R_ext[0::3]))
    w_tip_theory = P * Lx**3 / (3 * E * I)
    print("w_tip: ", w_tip)
    print("w_tip_theory: ", w_tip_theory)
    print("rel error percent: ", abs((w_tip - w_tip_theory) * 100 / w_tip_theory))
    # nodeIDs_east = mesh.node_sets["east"]
    # y_east = mesh.nodes[mesh.node_sets["east"], 1]
    # nodeIDs_east_ordered = nodeIDs_east[np.argsort(y_east)]
    # w_tip = w[nodeIDs_east_ordered]
    # y_east_ordered = mesh.nodes[nodeIDs_east_ordered, 1]

    Plot(config, mesh, solver_data)
