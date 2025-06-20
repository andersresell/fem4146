import numpy as np
from utils import *
from element_utils import *
import shape_functions


class Mesh:
    nodes = np.ndarray([])
    elements = np.ndarray([])
    node_sets = {}
    element_sets = {}

    def get_nN(self):
        """Get number of nodes"""
        return self.nodes.shape[0]

    def get_nE(self):
        """Get number of elements"""
        return self.elements.shape[0]


class QuadElementTraits:
    """Containing logic for creating higher order quad elements from Q4 elements"""

    Q4_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def __init__(self, element_type):
        if element_type == ELEMENT_TYPE_Q4 or element_type == ELEMENT_TYPE_Q4R or element_type == ELEMENT_TYPE_Q4_USER:
            self.nNl = 4
            self.xi_eta = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
            self.internal_edge_nodes = []
            self.internal_nodes = []
            self.corner_nodes = [0, 1, 2, 3]
        elif element_type == ELEMENT_TYPE_Q8:
            self.nNl = 8
            self.xi_eta = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0)]
            self.internal_edge_nodes = [[4], [5], [6], [7]]
            self.internal_nodes = []
            self.corner_nodes = [0, 1, 2, 3]
        elif element_type == ELEMENT_TYPE_Q9:
            self.nNl = 9
            self.xi_eta = [(-1, -1), (1, -1), (1, 1), (-1, 1), (0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
            self.internal_edge_nodes = [[4], [5], [6], [7]]
            self.internal_nodes = [8]
            self.corner_nodes = [0, 1, 2, 3]
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

        else:
            assert False

        #Test that the shape functions are correct
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

    def get_nodeIDs_from_Q4_entity(self, Q4_entity):
        nodeIDs = []
        if Q4_entity >= EDGE0_Q4 and Q4_entity < FACE_Q4:
            for k in range(2):
                nodeIDs.append(self.corner_nodes[self.Q4_edges[Q4_entity][k]])
            if len(self.internal_edge_nodes) > 0:
                nodeIDs_internal = self.internal_edge_nodes[Q4_entity]
                for il in nodeIDs_internal:
                    nodeIDs.append(il)
        else:
            assert Q4_entity == FACE_Q4
            nodeIDs = [il for il in range(self.nNl)]
        return nodeIDs


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
    # elements = np.zeros((nE, nNl), dtype=int)
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
    for set_name, data in mesh_Q4.element_sets.items():
        nodeIDs = set()  #use a set to avoid duplicate nodeIDs
        nE_set = len(data)
        for e_set in range(nE_set):
            e, Q4_entity = data[e_set]
            nodeIDs_loc = element_traits.get_nodeIDs_from_Q4_entity(Q4_entity)
            for il in nodeIDs_loc:
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
    element_sets["domain"] = set()
    element_sets["west"] = set()
    element_sets["east"] = set()
    element_sets["north"] = set()
    element_sets["south"] = set()

    for ey in range(nEy):
        for ex in range(nEx):
            e = IX_element(ex, ey)
            for il in range(nNl):
                i = (nNl_1D - 1) * ex + il_to_ij_loc[il][0]
                j = (nNl_1D - 1) * ey + il_to_ij_loc[il][1]
                element_sets["domain"].add((e, FACE_Q4))
                if i == 0:
                    element_sets["west"].add((e, EDGE3_Q4))
                if i == nNx - 1:
                    element_sets["east"].add((e, EDGE1_Q4))
                if j == 0:
                    element_sets["south"].add((e, EDGE0_Q4))
                if j == nNy - 1:
                    element_sets["north"].add((e, EDGE2_Q4))

    for name, data in element_sets.items():
        element_sets[name] = list(data)

    #Add the corner node sets
    node_sets = {}
    node_sets["south_west"] = np.array([IX_node(0, 0)], dtype=int)
    node_sets["south_east"] = np.array([IX_node(nNx - 1, 0)], dtype=int)
    node_sets["north_west"] = np.array([IX_node(0, nNy - 1)], dtype=int)
    node_sets["north_east"] = np.array([IX_node(nNx - 1, nNy - 1)], dtype=int)

    # node_sets = {}
    # node_sets["domain"] = np.arange(0, nN, dtype=int)
    # node_sets["west"] = np.array([], dtype=int)
    # node_sets["east"] = np.array([], dtype=int)
    # node_sets["north"] = np.array([], dtype=int)
    # node_sets["south"] = np.array([], dtype=int)

    # for j in range(nNy):
    #     for i in range(nNx):
    #         if i == 0:
    #             node_sets["west"] = np.append(node_sets["west"], IX(i, j))
    #         if i == nNx - 1:
    #             node_sets["east"] = np.append(node_sets["east"], IX(i, j))
    #         if j == 0:
    #             node_sets["south"] = np.append(node_sets["south"], IX(i, j))
    #         if j == nNy - 1:
    #             node_sets["north"] = np.append(node_sets["north"], IX(i, j))
    mesh = Mesh()
    mesh.nodes = nodes
    mesh.elements = elements
    mesh.element_sets = element_sets
    mesh.node_sets = node_sets

    return mesh


def create_structured_quad_mesh(config: Config, Lx, Ly, nEx, nEy) -> Mesh:
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
    mesh_Q4 = create_structured_Q4_mesh(Lx, Ly, nEx, nEy)
    mesh = create_higher_order_quad_mesh_from_Q4_mesh(mesh_Q4, config)
    return mesh
