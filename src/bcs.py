from mesh import Mesh
from utils import Config, BC
from element_utils import *

# def add_boundary_condition(config: Config, mesh: Mesh, node_set_name, dof, value=0):
#     node_sets = mesh.node_sets
#     bcs = config.bcs

#     try:
#         if not node_set_name in node_sets:
#             all_node_set_names = []
#             for name, _ in node_sets.items():
#                 all_node_set_names.append(name)
#             all_node_set_names = str(", ".join(all_node_set_names))
#             raise Exception("Node set name \'" + node_set_name + "\' is not among the defined node sets: " +
#                             all_node_set_names)

#         NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)

#         if dof < 0:
#             raise Exception("Tried to assign boundary condition to dof " + str(dof) + ", dof can't be negative")
#         elif dof >= NUM_DOFS:
#             raise Exception("Tried to assign boundary condition to dof " + str(dof) + " for a " +
#                             problem_type_to_string[config.problem_type] + " problem, which only has " + str(NUM_DOFS) +
#                             " dofs")

#         bc = BC()
#         bc.value = value
#         bc.node_set_name = node_set_name
#         bc.dof = dof
#         bcs.append()

#     except Exception as e:
#         print("Error adding boundary condition:", e)
#         exit(1)


def add_boundary_condition(config: Config, mesh: Mesh, node_set_name, dof, value=0):
    """Adds a prescribed boundary condition for a certain dof to a node set.
        For instance, if node_set_name is "west", dof is DOF_U, and value is 0, 
        all nodes in the "west" node set will have their x-displacement (DOF_U) set to 0.
    """
    node_sets = mesh.node_sets
    bcs = config.bcs

    if node_set_name not in node_sets:
        all_node_set_names = ", ".join(node_sets.keys())
        raise Exception(f"Node set '{node_set_name}' not found. Available: {all_node_set_names}")

    NUM_DOFS = get_num_dofs_from_problem_type(config.problem_type)

    if dof < 0:
        raise Exception(f"Invalid dof {dof}, must be >= 0")
    elif dof >= NUM_DOFS:
        raise Exception(f"dof {dof} out of bounds for {problem_type_to_string[config.problem_type]} "
                        f"(only {NUM_DOFS} dofs)")

    bc = BC()
    bc.value = value
    bc.node_set_name = node_set_name
    bc.dof = dof
    bcs.append(bc)
