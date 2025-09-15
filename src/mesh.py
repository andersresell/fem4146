from typing import Dict
import numpy as np
from src.utils import *
from src.fem_utils import *
import src.shape_functions as shape_functions


class ElementSet:

    def __init__(self, element_entity=None):
        self.elementIDs = np.ndarray([], dtype=int)
        self.element_entity = element_entity


class Mesh:

    def __init__(self):
        self.nodes = np.ndarray([])
        self.elements = np.ndarray([])
        self.node_sets = {}
        self.element_sets: Dict[str, ElementSet] = {}

    def get_nN(self):
        """Get number of nodes"""
        return self.nodes.shape[0]

    def get_nE(self):
        """Get number of elements"""
        return self.elements.shape[0]


class QuadElementTraits:
    """Containing logic for creating higher order quad elements from Q4 elements
    as well as natural coordinates of the nodes and local node indices for load integration"""

    Q4_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def __init__(self, element_type):
        if element_type == ELEMENT_TYPE_Q4 or element_type == ELEMENT_TYPE_Q4R or element_type == ELEMENT_TYPE_Q4_USER:
            self.nNl = 4
            self.xi_eta = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            self.internal_edge_nodes = []
            self.internal_nodes = []
            self.corner_nodes = [0, 1, 2, 3]
            self.entity_nodes = {
                EDGE0_QUAD: [0, 1],
                EDGE1_QUAD: [1, 2],
                EDGE2_QUAD: [2, 3],
                EDGE3_QUAD: [3, 0],
                FACE_QUAD: [0, 1, 2, 3]
            }
        elif element_type == ELEMENT_TYPE_Q8 or element_type == ELEMENT_TYPE_Q8R:
            self.nNl = 8
            self.xi_eta = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0)]
            self.internal_edge_nodes = [[4], [5], [6], [7]]
            self.internal_nodes = []
            self.corner_nodes = [0, 1, 2, 3]
            self.entity_nodes = {
                EDGE0_QUAD: [0, 4, 1],
                EDGE1_QUAD: [1, 5, 2],
                EDGE2_QUAD: [2, 6, 3],
                EDGE3_QUAD: [3, 7, 0],
                FACE_QUAD: [0, 1, 2, 3, 4, 5, 6, 7]
            }

        elif element_type == ELEMENT_TYPE_Q9 or element_type == ELEMENT_TYPE_Q9R:
            self.nNl = 9
            self.xi_eta = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
            self.internal_edge_nodes = [[4], [5], [6], [7]]
            self.internal_nodes = [8]
            self.corner_nodes = [0, 1, 2, 3]
            self.entity_nodes = {
                EDGE0_QUAD: [0, 4, 1],
                EDGE1_QUAD: [1, 5, 2],
                EDGE2_QUAD: [2, 6, 3],
                EDGE3_QUAD: [3, 7, 0],
                FACE_QUAD: [0, 1, 2, 3, 4, 5, 6, 7, 8]
            }
        elif element_type == ELEMENT_TYPE_Q16:
            self.nNl = 16
            #fmt: off
            self.xi_eta = [(-1, -1), (1, -1), (1, 1), (-1, 1),
                           (-1/3, -1), (1/3, -1), (1, -1/3), (1, 1/3),
                           (1/3, 1), (-1/3, 1), (-1, 1/3), (-1, -1/3),
                           (-1/3, -1/3), (1/3, -1/3), (1/3, 1/3), (-1/3, 1/3)]
            #fmt: on
            self.internal_edge_nodes = [[4, 5], [6, 7], [8, 9], [10, 11]]
            self.internal_nodes = [12, 13, 14, 15]
            self.corner_nodes = [0, 1, 2, 3]
            self.entity_nodes = {
                EDGE0_QUAD: [0, 4, 5, 1],
                EDGE1_QUAD: [1, 6, 7, 2],
                EDGE2_QUAD: [2, 8, 9, 3],
                EDGE3_QUAD: [3, 10, 11, 0],
                FACE_QUAD: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            }
        else:
            assert False

        #Test that the shape functions are correct (1 at the node and 0 at all other nodes)
        for il in range(self.nNl):
            xi, eta = self.xi_eta[il]
            N = shape_functions.calc_N(xi, eta, element_type)
            for i in range(self.nNl):
                if i == il:
                    if not np.isclose(N[i], 1.0):
                        print(
                            f"Shape function test failed: Shape function for node {i} at xi={xi}, eta={eta} is {N[i]}")
                        exit(1)
                else:
                    if not np.isclose(N[i], 0.0):
                        print(
                            f"Shape function test failed: Shape function for node {i} at xi={xi}, eta={eta} is {N[i]}")
                        exit(1)

    def is_corner_node(self, il):
        return il in self.corner_nodes

    def is_internal_node(self, il):
        return il in self.internal_nodes

    def is_edge_node(self, il):
        return il in self.internal_nodes or il in self.corner_nodes

    def is_internal_edge_node(self, il):
        for edge in self.internal_edge_nodes:
            if il in edge:
                return True
        return False

    def get_edge_id(self, il):
        assert self.is_internal_edge_node(il)
        for edge_id in range(len(self.internal_edge_nodes)):
            if il in self.internal_edge_nodes[edge_id]:
                return edge_id
        assert False

    def get_interal_edge_local_index(self, il, edge_id):
        assert il in self.internal_edge_nodes[edge_id]
        return self.internal_edge_nodes[edge_id].index(il)

    def get_nodeIDs_from_element_entity(self, element_entity):
        return self.entity_nodes[element_entity]


def check_if_internal_edge_is_created(existing_edges_Q4, I_corner_a, I_corner_b):
    exists = False
    reversed = None
    if (I_corner_a, I_corner_b) in existing_edges_Q4:
        reversed = False
        exists = True
    elif (I_corner_b, I_corner_a) in existing_edges_Q4:
        reversed = True
        exists = True
    return exists, reversed


def create_higher_order_quad_mesh_from_Q4_mesh(mesh_Q4: Mesh, config: Config):
    element_type = config.element_type

    element_traits = QuadElementTraits(element_type)
    nNl = element_traits.nNl

    mesh = Mesh()
    elements_Q4 = mesh_Q4.elements
    nodes_Q4 = mesh_Q4.nodes
    assert elements_Q4.shape[1] == 4
    nE = elements_Q4.shape[0]

    nodes = nodes_Q4.copy()
    elements = np.full((nE, nNl), fill_value=-1, dtype=int)
    existing_edges_Q4 = {}
    new_node_counter = nodes_Q4.shape[0]
    for e in range(nE):

        #Retrieve the Q4 corner nodes
        x_Q4 = nodes_Q4[elements_Q4[e, :], 0]
        y_Q4 = nodes_Q4[elements_Q4[e, :], 1]

        for il in range(nNl):

            add_new_node = False
            I = -1
            if element_traits.is_corner_node(il):
                #If the node is a corner node, retrieve the Q4 corner node number
                I = elements_Q4[e, element_traits.corner_nodes.index(il)]
            elif element_traits.is_internal_node(il):
                #If the node is an internal node, we always add a new node
                add_new_node = True
            else:
                #Now we know that the node is an internal edge node
                edge_id = element_traits.get_edge_id(il)

                I_corner_a = elements_Q4[e, element_traits.Q4_edges[edge_id][0]]
                I_corner_b = elements_Q4[e, element_traits.Q4_edges[edge_id][1]]

                exists, reversed = check_if_internal_edge_is_created(existing_edges_Q4, I_corner_a, I_corner_b)
                if exists:
                    if reversed:
                        I_corner_tuple = (I_corner_b, I_corner_a)
                    else:
                        I_corner_tuple = (I_corner_a, I_corner_b)
                    #Internal edge node already exist
                    num_internal_edge_nodes = len(element_traits.internal_edge_nodes[edge_id])
                    assert num_internal_edge_nodes == len(existing_edges_Q4[I_corner_tuple])
                    local_index = element_traits.get_interal_edge_local_index(il, edge_id)
                    if reversed:
                        local_index = num_internal_edge_nodes - 1 - local_index
                    I = existing_edges_Q4[I_corner_tuple][local_index]
                else:
                    #Internal edge node doesn't exist, so we must add a new
                    add_new_node = True

            if add_new_node:
                I = new_node_counter
                new_node_counter += 1

                xi, eta = element_traits.xi_eta[il]
                N = shape_functions.calc_N(xi, eta, ELEMENT_TYPE_Q4)
                #Calculate the new node coordinates
                x, y = N @ x_Q4, N @ y_Q4
                nodes = np.append(nodes, [[x, y]], axis=0)

            assert I not in elements[e, :]
            elements[e, il] = I

        #Cache all edges that wasn't previously created for this element
        for edge_id in range(4):
            I_corner_a = elements[e, element_traits.Q4_edges[edge_id][0]]
            I_corner_b = elements[e, element_traits.Q4_edges[edge_id][1]]
            if (I_corner_a, I_corner_b) not in existing_edges_Q4:  # and len(element_traits.internal_edge_nodes) > 0:
                existing_edges_Q4[(I_corner_a, I_corner_b)] = []
                if len(element_traits.internal_edge_nodes) > 0:
                    for il in element_traits.internal_edge_nodes[edge_id]:
                        existing_edges_Q4[(I_corner_a, I_corner_b)].append(elements[e, il])

    #====================================================================
    # Create node sets from the previously created element sets
    #====================================================================

    node_sets = mesh_Q4.node_sets.copy()
    for set_name, elem_set in mesh_Q4.element_sets.items():
        nodeIDs = set()  #use a set to avoid duplicate nodeIDs
        elementIDs = elem_set.elementIDs
        nE_set = len(elementIDs)
        element_entity = elem_set.element_entity
        for e_l in range(nE_set):
            e = elementIDs[e_l]
            nodeIDs_l = element_traits.get_nodeIDs_from_element_entity(element_entity)
            for il in nodeIDs_l:
                I = elements[e, il]
                nodeIDs.add(I)
        node_sets[set_name] = np.array(sorted(nodeIDs), dtype=int)

    mesh.nodes = nodes
    mesh.elements = elements
    mesh.element_sets = mesh_Q4.element_sets.copy()
    mesh.node_sets = node_sets
    return mesh


def create_structured_Q4_mesh(Lx, Ly, nEx, nEy) -> Mesh:

    #====================================================================
    # Create nodes and elements
    #====================================================================
    nE = nEx * nEy
    nNl = 4
    nNx = nEx + 1
    nNy = nEy + 1
    nN = nNx * nNy

    dx = Lx / nEx
    dy = Ly / nEy

    elements = np.zeros((nE, nNl), dtype=int)
    nodes = np.zeros((nN, 2))

    def IX_node(i, j):
        #maps 2D index to node number
        assert i >= 0 and j >= 0
        assert i < nNx and j < nNy
        I = j * nNx + i
        return I

    def IX_element(ex, ey):
        #maps 2D index to node number
        assert ex >= 0 and ey >= 0
        assert ex < nEx and ey < nEy
        e = ey * nEx + ex
        return e

    for j in range(nNy):
        for i in range(nNx):
            x = i * dx
            y = j * dy
            I = IX_node(i, j)
            nodes[I, :] = np.array([x, y])

    il_to_ij_loc = il_to_ij_loc_all[ELEMENT_TYPE_Q4]
    nNl_1D = 2
    for ey in range(nEy):
        for ex in range(nEx):
            e = IX_element(ex, ey)
            for il in range(nNl):
                i = (nNl_1D - 1) * ex + il_to_ij_loc[il][0]
                j = (nNl_1D - 1) * ey + il_to_ij_loc[il][1]
                elements[e, il] = IX_node(i, j)
    """Create predefined element sets
        Domain is shown below:
   
                          north
 north_west .______________________________.north_east
            |                              |
            |                              |
            |                              |
            |                              |
    west    |            domain            |east
            |                              | 
            |                              |
            |                              |
 south_west .______________________________.south_east
                          south 
                          
    Each quad element is marked with local element entities as follows
    
        ______EDGE2_____
        |               |                    
        |               |
   EDGE3|      FACE     |EDGE1
        |               |
        |               |
        |_______________|
              EDGE0
    Each element set is marked with a name and contains a
    list of tuples (element_index, element_entity)
    """
    element_sets = {}
    element_sets["domain"] = ElementSet(FACE_QUAD)
    element_sets["west"] = ElementSet(EDGE3_QUAD)
    element_sets["east"] = ElementSet(EDGE1_QUAD)
    element_sets["south"] = ElementSet(EDGE0_QUAD)
    element_sets["north"] = ElementSet(EDGE2_QUAD)

    element_sets_tmp = {}
    for set_name in element_sets.keys():
        element_sets_tmp[set_name] = set()  #using sets temporarily to have

    for ey in range(nEy):
        for ex in range(nEx):
            e = IX_element(ex, ey)
            for il in range(nNl):
                i = (nNl_1D - 1) * ex + il_to_ij_loc[il][0]
                j = (nNl_1D - 1) * ey + il_to_ij_loc[il][1]
                element_sets_tmp["domain"].add(e)
                if i == 0:
                    element_sets_tmp["west"].add(e)
                if i == nNx - 1:
                    element_sets_tmp["east"].add(e)
                if j == 0:
                    element_sets_tmp["south"].add(e)
                if j == nNy - 1:
                    element_sets_tmp["north"].add(e)

    #Convert the sets to np arrays and store them in the element sets
    for set_name, elem_set in element_sets_tmp.items():
        element_sets[set_name].elementIDs = np.array(sorted(elem_set), dtype=int)

    #Add the corner node sets
    node_sets = {}
    node_sets["south_west"] = np.array([IX_node(0, 0)], dtype=int)
    node_sets["south_east"] = np.array([IX_node(nNx - 1, 0)], dtype=int)
    node_sets["north_west"] = np.array([IX_node(0, nNy - 1)], dtype=int)
    node_sets["north_east"] = np.array([IX_node(nNx - 1, nNy - 1)], dtype=int)

    mesh = Mesh()
    mesh.nodes = nodes
    mesh.elements = elements
    mesh.element_sets = element_sets
    mesh.node_sets = node_sets

    return mesh


def perturb_mesh_nodes_of_structured_Q4_mesh(mesh_Q4: Mesh, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor):
    """Perturb the nodes of the structured mesh randomly to create badly shaped elements.
    This is useful to illustrate the effects of badly shaped elements.
    Note that the border nodes are constrained to stay on the original border."""
    SAFETY_FACTOR = 0.6
    if perturb_mesh_nodes_factor <= 0:
        return
    else:
        if perturb_mesh_nodes_factor > 1:
            print("Error: perturb_mesh_nodes_factor must be between 0 and 1")
            exit(1)

    nodes = mesh_Q4.nodes
    dx = Lx / nEx
    dy = Ly / nEy
    max_perturbation_dist_x = SAFETY_FACTOR * perturb_mesh_nodes_factor * 0.5 * dx
    max_perturbation_dist_y = SAFETY_FACTOR * perturb_mesh_nodes_factor * 0.5 * dy

    for i in range(nodes.shape[0]):
        factor = np.random.uniform(-1.0, 1.0)
        x, y = nodes[i, :]
        x_perturbed = x
        y_perturbed = y
        if SMALL_VAL < x < Lx - SMALL_VAL:
            #Node is neither on the left nor right border
            x_perturbed = x + factor * max_perturbation_dist_x
            assert 0 <= x_perturbed <= Lx
        if SMALL_VAL < y < Ly - SMALL_VAL:
            #Node is neither on the top nor bottom border
            y_perturbed = y + factor * max_perturbation_dist_y
            assert 0 <= y_perturbed <= Ly
        nodes[i, :] = np.array([x_perturbed, y_perturbed])


# #====================================================================
# # Max allowed value for perturb_mesh_nodes_factor (This value is found by trial and error, by checking
# # when the Jacobian determinant becomes negative for some elements)
# #====================================================================
# PERTURB_MESH_NODES_FACTOR_SAFETY = 0.4
# def perturb_mesh_nodes_of_structured_mesh(config: Config, mesh_Q4: Mesh, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor):
#     """Perturb the nodes of the structured mesh randomly to create badly shaped elements.
#     This is useful to illustrate the effects of badly shaped elements.
#     Note that the border nodes are constrained to stay on the original border."""
#     if perturb_mesh_nodes_factor <= 0:
#         return
#     else:
#         if perturb_mesh_nodes_factor > 1:
#             print(f"Error: perturb_mesh_nodes_factor must be between 0 and 1")
#             exit(1)
#     np.random.seed(0)  #For reproducibility. Running the same simulation multiple times should give the same result

#     nodes = mesh_Q4.nodes
#     nNl_1D = element_type_to_nNl_1D[config.element_type]
#     dx_min = Lx / (nEx * (nNl_1D - 1))
#     dy_min = Ly / (nEy * (nNl_1D - 1))
#     max_perturbation_dist_x = PERTURB_MESH_NODES_FACTOR_SAFETY * perturb_mesh_nodes_factor * 0.5 * dx_min
#     max_perturbation_dist_y = PERTURB_MESH_NODES_FACTOR_SAFETY * perturb_mesh_nodes_factor * 0.5 * dy_min

#     for i in range(nodes.shape[0]):
#         factor = np.random.uniform(-1.0, 1.0)
#         x, y = nodes[i, :]
#         x_perturbed = x
#         y_perturbed = y
#         if SMALL_VAL < x < Lx - SMALL_VAL:
#             #Node is neither on the left nor right border
#             x_perturbed = x + factor * max_perturbation_dist_x
#             assert 0 <= x_perturbed <= Lx
#         if SMALL_VAL < y < Ly - SMALL_VAL:
#             #Node is neither on the top nor bottom border
#             y_perturbed = y + factor * max_perturbation_dist_y
#             assert 0 <= y_perturbed <= Ly
#         nodes[i, :] = np.array([x_perturbed, y_perturbed])


def create_structured_quad_mesh(config: Config, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor=0) -> Mesh:
    """Creates a structured rectangular quad mesh with the specified parameters.
    Args:
        config (Config): Configuration object containing problem parameters.
        Lx (float): Length of the mesh in the x-direction.
        Ly (float): Length of the mesh in the y-direction.
        nEx (int): Number of elements in the x-direction.
        nEy (int): Number of elements in the y-direction.
    Returns:
        Mesh: A Mesh object containing the structured quad mesh.
    """
    try:
        if nEx < 1:
            raise ValueError("nEx must be at least 1")
        elif nEy < 1:
            raise ValueError("nEy must be at least 1")
        elif Lx < SMALL_VAL:
            raise ValueError("Lx must be greater than 0")
        elif Ly < SMALL_VAL:
            raise ValueError("Ly must be greater than 0")
    except ValueError as e:
        print(f"Error in create_structured_quad_mesh: {e}")
        exit(1)

    mesh_Q4 = create_structured_Q4_mesh(Lx, Ly, nEx, nEy)
    perturb_mesh_nodes_of_structured_Q4_mesh(mesh_Q4, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor)
    mesh = create_higher_order_quad_mesh_from_Q4_mesh(mesh_Q4, config)
    # perturb_mesh_nodes_of_structured_mesh(config, mesh, Lx, Ly, nEx, nEy, perturb_mesh_nodes_factor)
    return mesh
