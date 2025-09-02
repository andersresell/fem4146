import inspect
from src.mesh import Mesh, QuadElementTraits
from src.utils import Config, BC, Load
from src.solver_data import SolverData
from src.fem_utils import *
import src.shape_functions as shape_functions


def validate_load_func(f):
    if not callable(f):
        raise TypeError("Load must be a callable. Likely something wrong with the load function")
    sig = inspect.signature(f)
    if len(sig.parameters) != 2:
        raise TypeError("Load function must take exactly two arguments (x, y)")
    try:
        result = f(0.0, 0.0)
    except Exception as e:
        raise ValueError(f"Load function raised an error when called with (0.0, 0.0): {e}")

    if isinstance(result, (int, float)):
        return 1
    elif isinstance(result, tuple) and len(result) == 2:
        if all(isinstance(v, (int, float)) for v in result):
            return 2

    raise TypeError("""Load function must return a float or a tuple of two floats.\n  
        Example of valid load functions are: 
            Two outputs (eg. traction): f_load = lambda x,y: (10,y)
            One output (eg. pressure): f_load = lambda x,y: 10
            May also write a function like:
            def f_load(x, y):
                return 10, y
    """)


def check_load_validity(config: Config, mesh: Mesh, element_set_name, load_type, load_function):

    element_sets = mesh.element_sets

    #====================================================================
    # First some checks to ensure that the element set exists
    #====================================================================
    if element_set_name not in element_sets:
        all_element_set_names = ", ".join(
            element_sets.keys())  #Find all available element set names to display in the error message
        raise Exception(f"Element set '{element_set_name}' not found. Available: {all_element_set_names}")

    #====================================================================
    # Then some checks to ensure that the load type is valid for
    # the problem type and the type of element set
    #====================================================================
    element_set = element_sets[element_set_name]
    element_entity = element_set.element_entity
    geometry_type = element_entity_to_geometry_type[element_entity]
    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        if geometry_type == ELEMENT_GEOMETRY_TYPE_EDGE:
            if load_type not in [LOAD_TYPE_TRACTION, LOAD_TYPE_PRESSURE]:
                raise Exception(
                    f"Load type '{load_type_to_string[load_type]}' is not valid for {problem_type_to_string[config.problem_type]} "
                    f"when the element set defines an edge.")
        else:
            assert geometry_type == ELEMENT_GEOMETRY_TYPE_FACE
            if load_type not in [LOAD_TYPE_TRACTION, LOAD_TYPE_BODY_FORCE]:
                raise Exception(
                    f"Load type '{load_type_to_string[load_type]}' is not valid for {problem_type_to_string[config.problem_type]} "
                    f"when the element set defines a face.")
    else:
        assert config.problem_type == PROBLEM_TYPE_PLATE
        if geometry_type == ELEMENT_GEOMETRY_TYPE_EDGE:
            if load_type != LOAD_TYPE_TRACTION:
                raise Exception(f"Only LOAD_TYPE_TRACTION is valid for {problem_type_to_string[config.problem_type]} "
                                f"when the element set defines an edge.")
        else:
            assert geometry_type == ELEMENT_GEOMETRY_TYPE_FACE
            if load_type not in [LOAD_TYPE_PRESSURE, LOAD_TYPE_BODY_FORCE]:
                raise Exception(
                    f"Only LOAD_TYPE_PRESSURE and LOAD_TYPE_BODY_FORCE are valid for {problem_type_to_string[config.problem_type]} "
                    f"when the element set defines a face.")

    #====================================================================
    # Then a bunch of checks to ensure the load function is valid
    #====================================================================
    num_output_components = validate_load_func(load_function)
    assert num_output_components in [1, 2]
    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        if load_type == LOAD_TYPE_TRACTION or load_type == LOAD_TYPE_BODY_FORCE:
            if num_output_components == 1:
                raise Exception(
                    f"A load function that returns a single value cannot be used with LOAD_TYPE_TRACTION or LOAD_TYPE_BODY_FORCE in "
                    f"{problem_type_to_string[config.problem_type]}. Use LOAD_TYPE_PRESSURE instead.")
            elif num_output_components != 2:
                raise Exception(f"Load function must return either a single value or a tuple of two values. "
                                f"Got {num_output_components} components instead.")
        else:
            assert load_type == LOAD_TYPE_PRESSURE
            if num_output_components != 1:
                raise Exception(
                    f"Load function must return a single value for LOAD_TYPE_PRESSURE in {problem_type_to_string[config.problem_type]}. "
                    f"Got {num_output_components} components instead.")
    else:
        assert config.problem_type == PROBLEM_TYPE_PLATE
        if num_output_components != 1:
            raise Exception(
                f"Load type '{load_type_to_string[load_type]}' is not valid for a load function that returns "
                f"{num_output_components} components in {problem_type_to_string[config.problem_type]}. "
                f"Use a single-valued load function for LOAD_TYPE_TRACTION, LOAD_TYPE_PRESSURE or LOAD_TYPE_BODY_FORCE."
            )


def add_load(config: Config, mesh: Mesh, element_set_name, load_type, load_function):
    """Adds a load to the specified element set with the given load function.
       The load function is a lambda or function that takes (x, y) coordinates and
       returns either a single value (for LOAD_TYPE_PRESSURE) or a tuple of two values
       (for LOAD_TYPE_TRACTION or LOAD_TYPE_BODY_FORCE). 
       Examples:
       load_function = lambda x, y: (10, x) # For LOAD_TYPE_TRACTION or LOAD_TYPE_BODY_FORCE
       load_function = lambda x, y: 10 # For LOAD_TYPE_PRESSURE
    """

    try:
        check_load_validity(config, mesh, element_set_name, load_type, load_function)
    except Exception as e:
        raise ValueError(f"Invalid load configuration for element set '{element_set_name}' "
                         f"and load type '{load_type_to_string[load_type]}':\n{e}")

    loads = config.loads
    load = Load()
    load.element_set_name = element_set_name
    load.load_type = load_type
    load.load_function = load_function
    loads.append(load)


def integrate_element_face_load_plane_stress(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
    element_type = config.element_type
    h = config.h
    nNl_face = len(nodeIDs_l)
    assert nNl_face == element_type_to_nNl[config.element_type]
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_l = mesh.nodes[nodeIDs, 0]
    y_l = mesh.nodes[nodeIDs, 1]
    #We integrate loads with full integration even if the element stiffness matrix uses reduced integration.
    nGauss_1D = shape_functions.element_type_to_nGauss_1D[element_type] + element_uses_reduced_integration[element_type]
    arr_xi = shape_functions.get_arr_xi(nGauss_1D)
    arr_w = shape_functions.get_arr_w(nGauss_1D)
    R_load = np.zeros(2 * nNl_face)
    for i in range(nGauss_1D):
        for j in range(nGauss_1D):
            xi = arr_xi[i]
            eta = arr_xi[j]
            w = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            if detJ < SMALL_VAL:
                raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                                 f"Check the element geometry and node ordering.")

            #Used to evaluate the load function at the Gauss points
            x_Gauss = N @ x_l
            y_Gauss = N @ y_l

            if load.load_type == LOAD_TYPE_TRACTION:
                t = load.load_function(x_Gauss, y_Gauss)
                assert len(t) == 2
            else:
                assert load.load_type == LOAD_TYPE_BODY_FORCE
                b = load.load_function(x_Gauss, y_Gauss)
                assert len(b) == 2
                t = h * b  #convert body force to a in plane traction by multiplying by the plate thickness

            R_load[0::2] += t[0] * N * detJ * w
            R_load[1::2] += t[1] * N * detJ * w
    return R_load


def integrate_element_edge_load_plane_stress(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == PROBLEM_TYPE_PLANE_STRESS
    element_type = config.element_type
    h = config.h
    nNl_edge = len(nodeIDs_l)
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_edge = mesh.nodes[nodeIDs, 0]
    y_edge = mesh.nodes[nodeIDs, 1]
    #We integrate loads with full integration even if the element uses reduced integration.
    nGauss = shape_functions.element_type_to_nGauss_1D[element_type] + element_uses_reduced_integration[element_type]
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)
    R_load = np.zeros(2 * nNl_edge)
    for i in range(nGauss):
        xi = arr_xi[i]
        w = arr_w[i]
        N = shape_functions.calc_N_1D(xi, nNl_edge)
        dNdxi = shape_functions.calc_dNdxi_1D(xi, nNl_edge)
        dxdxi = dNdxi @ x_edge
        dydxi = dNdxi @ y_edge
        detJ = np.sqrt(dxdxi**2 + dydxi**2)
        if detJ < SMALL_VAL:
            raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                             f"Check the element geometry and node ordering.")

        # Normal vector to the edge (the edge is assumed to run counter-clockwise around the element)
        normal = np.array([dydxi, -dxdxi]) / detJ

        #Used to evaluate the load function at the Gauss points
        x_Gauss = N @ x_edge
        y_Gauss = N @ y_edge
        if load.load_type == LOAD_TYPE_TRACTION:
            t = load.load_function(x_Gauss, y_Gauss)
        else:
            assert load.load_type == LOAD_TYPE_PRESSURE
            p = load.load_function(x_Gauss, y_Gauss)
            t = -p * normal

        R_load[0::2] += t[0] * N * detJ * w * h
        R_load[1::2] += t[1] * N * detJ * w * h
    return R_load


def integrate_element_face_load_plate(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == PROBLEM_TYPE_PLATE
    element_type = config.element_type
    h = config.h
    nNl_face = len(nodeIDs_l)
    assert nNl_face == element_type_to_nNl[config.element_type]
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_l = mesh.nodes[nodeIDs, 0]
    y_l = mesh.nodes[nodeIDs, 1]
    #We integrate loads with full integration even if the element stiffness matrix uses reduced integration.
    nGauss_1D = shape_functions.element_type_to_nGauss_1D[element_type] + element_uses_reduced_integration[element_type]
    arr_xi = shape_functions.get_arr_xi(nGauss_1D)
    arr_w = shape_functions.get_arr_w(nGauss_1D)
    #We make the load vector contain all 3 dofs w,thetax, thetay, even though we only fill the w dofs
    #This is because we don't want special code in the assembly procedure dealing with a reduced form of the shape functions
    R_load = np.zeros(3 * nNl_face)
    for i in range(nGauss_1D):
        for j in range(nGauss_1D):
            xi = arr_xi[i]
            eta = arr_xi[j]
            w = arr_w[i] * arr_w[j]
            N = shape_functions.calc_N(xi, eta, element_type)
            J = shape_functions.calc_J(xi, eta, x_l, y_l, element_type)
            detJ = np.linalg.det(J)
            if detJ < SMALL_VAL:
                raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                                 f"Check the element geometry and node ordering.")

            #Used to evaluate the load function at the Gauss points
            x_Gauss = N @ x_l
            y_Gauss = N @ y_l

            if load.load_type == LOAD_TYPE_PRESSURE:
                p = load.load_function(x_Gauss, y_Gauss)
                assert isinstance(p, (int, float))
            else:
                assert load.load_type == LOAD_TYPE_BODY_FORCE
                b = load.load_function(x_Gauss, y_Gauss)
                assert isinstance(b, (int, float))
                p = h * b  #We convert the body force to a pressure by multiplying with thickness

            R_load[0::3] += -p * N * detJ * w  #The pressure force only contributes to the w-dofs

    return R_load


def integrate_element_edge_load_plate(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e):
    assert config.problem_type == PROBLEM_TYPE_PLATE
    element_type = config.element_type
    h = config.h
    nNl_edge = len(nodeIDs_l)
    nodeIDs = mesh.elements[e, nodeIDs_l]
    x_edge = mesh.nodes[nodeIDs, 0]
    y_edge = mesh.nodes[nodeIDs, 1]
    #We integrate loads with full integration even if the element uses reduced integration.
    nGauss = shape_functions.element_type_to_nGauss_1D[element_type] + element_uses_reduced_integration[element_type]
    arr_xi = shape_functions.get_arr_xi(nGauss)
    arr_w = shape_functions.get_arr_w(nGauss)
    #We specify an empty load vector containing all dofs w,thetax,thetaz, allthough the two latter will remain zero
    R_load = np.zeros(3 * nNl_edge)
    for i in range(nGauss):
        xi = arr_xi[i]
        w = arr_w[i]
        N = shape_functions.calc_N_1D(xi, nNl_edge)
        dNdxi = shape_functions.calc_dNdxi_1D(xi, nNl_edge)
        dxdxi = dNdxi @ x_edge
        dydxi = dNdxi @ y_edge
        detJ = np.sqrt(dxdxi**2 + dydxi**2)
        if detJ < SMALL_VAL:
            raise ValueError(f"Element {e} has a zero or negative Jacobian determinant: {detJ}. "
                             f"Check the element geometry and node ordering.")

        #Used to evaluate the load function at the Gauss points
        x_Gauss = N @ x_edge
        y_Gauss = N @ y_edge

        #We only accept traction loads for the edge loads of the plates.
        #However this traction function should return a single value.
        #In global 3D space we assume the traction has the form t = t_z * [0, 0, 1]^T, and we only provide t_z
        assert load.load_type == LOAD_TYPE_TRACTION
        t_z = load.load_function(x_Gauss, y_Gauss)
        assert isinstance(t_z, (int, float))  #Check that t_z only is a single value

        R_load[0::3] += t_z * N * detJ * w * h  #This traction load only acts in the z direction
    return R_load


def integrate_element_load(config: Config, mesh: Mesh, load: Load, nodeIDs_l, e, geometry_type):
    if config.problem_type == PROBLEM_TYPE_PLANE_STRESS:
        if geometry_type == ELEMENT_GEOMETRY_TYPE_FACE:
            R_load = integrate_element_face_load_plane_stress(config, mesh, load, nodeIDs_l, e)
        else:
            assert geometry_type == ELEMENT_GEOMETRY_TYPE_EDGE
            R_load = integrate_element_edge_load_plane_stress(config, mesh, load, nodeIDs_l, e)
    else:
        assert config.problem_type == PROBLEM_TYPE_PLATE
        if geometry_type == ELEMENT_GEOMETRY_TYPE_FACE:
            R_load = integrate_element_face_load_plate(config, mesh, load, nodeIDs_l, e)
        else:
            assert geometry_type == ELEMENT_GEOMETRY_TYPE_EDGE
            R_load = integrate_element_edge_load_plate(config, mesh, load, nodeIDs_l, e)

    return R_load


def assemble_loads_consistent(config: Config, mesh: Mesh, solver_data: SolverData):

    element_sets = mesh.element_sets
    elements = mesh.elements
    element_traits = QuadElementTraits(config.element_type)

    for load in config.loads:
        element_set = element_sets[load.element_set_name]
        element_entity = element_set.element_entity
        geometry_type = element_entity_to_geometry_type[element_entity]
        nodeIDs_l = element_traits.get_nodeIDs_from_element_entity(element_entity)
        nNl_ent = len(nodeIDs_l)
        NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)

        for e in element_set.elementIDs:

            #====================================================================
            # Integrate consistent load for the element in question
            #====================================================================
            R_load = integrate_element_load(config, mesh, load, nodeIDs_l, e, geometry_type)

            #====================================================================
            # Assemble contribution into the external force vector R_ext
            #====================================================================
            assert len(R_load) == nNl_ent * NUM_DOFS
            nodeIDs = elements[e, nodeIDs_l]
            for il in range(nNl_ent):
                for dof in range(NUM_DOFS):
                    I = nodeIDs[il] * NUM_DOFS + dof
                    solver_data.R_ext[I] += R_load[NUM_DOFS * il + dof]

    print("Consistent load integration complete")
