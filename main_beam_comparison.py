from src.useful_imports import *  #import required functions

if __name__ == "__main__":

    E = 210e9  # Young's modulus
    nu = 0.3  # Poisson's ratio
    Lx = 10  #Length in x-direction
    Ly = 1  #Length in y-direction
    h = 1  # Length into the third dimension (out of plane thickness)
    F_tip = -10000  #Total force applied at the right edge. Will be applied as a vertical traction of magnitude F_tip/(h*Ly)
    problem_type = PROBLEM_TYPE_PLANE_STRESS  #Specify that a plane stress problem is solved
    # nEx = 10  #Number of elements in x-direction
    # nEy = 5  #Number of elements in y-direction

    I = Ly**3 * h / 12  #Second moment of inertia for a rectangular cross-section
    x_theory = np.linspace(0, Lx)
    v_theory = F_tip * x_theory**2 / (6 * E * I) * (3 * Lx - x_theory)  #Beam theory solution

    v_theory_tip = v_theory[-1]

    hg_factor = 0.0000001
    hg_factor = 0.01

    nEys = [4, 8]  # [3, 6, 10]  # 10, 1]  #number of elements in y-direction
    for nEy in nEys:
        if nEy == 1:
            tmp = "element"
        else:
            tmp = "elements"
        name = f"{nEy} {tmp} over the thickness"
        plt.figure(name)
        plt.title(f"Lateral displacement\n{nEy} {tmp} over the thickness, L/h = {Lx/Ly:.1f}")
        plt.plot(x_theory, v_theory, label="Beam theory solution", linestyle='-', linewidth=2, color='black')
        nEx = int(nEy * Lx / Ly)
        element_types = [ELEMENT_TYPE_Q4, ELEMENT_TYPE_Q4R]  #, ELEMENT_TYPE_Q8, ELEMENT_TYPE_Q9, ELEMENT_TYPE_Q16]
        tip_disps = []
        for element_type in element_types:
            #====================================================================
            # Group problem settings in an object called config
            #====================================================================
            config = create_config(E, nu, h, element_type, problem_type)
            config.hourglass_scaling = hg_factor

            #====================================================================
            # Create a rectangular structured mesh
            #====================================================================
            mesh = create_structured_quad_mesh(config, Lx, Ly, nEx, nEy)

            #====================================================================
            # Add fixed boundary condition  to the left edge called "west"
            #====================================================================
            add_boundary_condition(config, mesh, "west", DOF_U, 0)  #set u to 0
            add_boundary_condition(config, mesh, "south_west", DOF_V, 0)  #set v to 0

            #====================================================================
            # Assign the tip load as a traction on the right edge named "east"
            #====================================================================
            load_func = lambda x, y: (0, F_tip / (h * Ly))
            add_load(config, mesh, "east", LOAD_TYPE_TRACTION, load_func)

            #====================================================================
            # Create objects holding system matrices and vectors. Then assemble
            # the stiffness matrix, integrate loads and assign boundary conditions
            # before finally solving the system K*r = R_ext
            #====================================================================
            solver_data = create_solver_data(config, mesh)
            solve(config, solver_data, mesh)

            #====================================================================
            # Take out the displacements along the top and bottom edges and
            # average them to get a more accurate estimate of the tip displacement
            #====================================================================
            u, v = unpack_solution(config, solver_data.r)
            nodeIDs_top = mesh.node_sets["north"]
            x_top = mesh.nodes[nodeIDs_top, 0]
            nodeIDs_top_ordered = nodeIDs_top[np.argsort(x_top)]
            x_top_ordered = mesh.nodes[nodeIDs_top_ordered, 0]
            v_top_ordered = v[nodeIDs_top_ordered]

            nodeIDs_bottom = mesh.node_sets["south"]
            x_bottom = mesh.nodes[nodeIDs_bottom, 0]
            nodeIDs_bottom_ordered = nodeIDs_bottom[np.argsort(x_bottom)]
            x_bottom_ordered = mesh.nodes[nodeIDs_bottom_ordered, 0]
            assert np.allclose(x_top_ordered, x_bottom_ordered)
            v_bottom_ordered = v[nodeIDs_bottom_ordered]
            v_mid_ordered = 0.5 * (v_top_ordered + v_bottom_ordered)

            label = element_type_to_str[element_type]
            label += f", % error = {((v_mid_ordered[-1] - v_theory_tip) * 100 / v_theory_tip):.2f}%"
            plt.plot(x_top_ordered, v_mid_ordered, label=label, linestyle="--")
            tip_disps.append(v_mid_ordered[-1])

        print("\nTip displacements and relative errors:")
        print("Tip displacement theory:", v_theory_tip)
        for i, element_type in enumerate(element_types):
            v_tip = tip_disps[i]
            print(
                f"{element_type_to_str[element_type]}: Tip displacement = {v_tip:.3e}, error percent: {((v_tip - v_theory_tip) * 100 / v_theory_tip):.4f}%"
            )
        plt.xlabel("x")
        plt.ylabel("v")
        plt.legend()
        plt.tight_layout()

        #increase dpi for better quality

        plt.savefig(name + ".png", dpi=600)

    #====================================================================
    # Start the GUI to visualize the results
    #====================================================================
    Plot(config, mesh, solver_data)

    plt.show()
