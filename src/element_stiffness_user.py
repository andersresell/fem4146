import src.shape_functions
from src.utils import Config
from src.mesh import Mesh
from src.fem_utils import *
from src.solver_data import SolverData


def calc_Ke_plane_stress_user_Q4(config: Config, mesh: Mesh, e):
    elements = mesh.elements  # A matrix of shape (nE, nNl) where nE is the number of elements and nNl is the number of nodes per element. Stores the node indices for each element.
    nodeIDs_e = elements[e, :]  # Get the node indices for the element e
    x_e = mesh.nodes[nodeIDs_e, 0]  # x-coordinates of the nodes of the element
    y_e = mesh.nodes[nodeIDs_e, 1]  # y-coordinates of the nodes of the element
    E = config.E  # Young's modulus
    nu = config.nu  # Poisson's ratio
    h = config.h  # (Uniform) thickness of the plate
    nNl = 4  # Number of nodes in a Q4 element
    Ke = np.zeros((2 * nNl, 2 * nNl))  # Initialize the element stiffness matrix for the Q4 element

    #====================================================================
    # IMPLEMENT USER-DEFINED Q4 ELEMENT STIFFNESS CALCULATION BELOW:
    #====================================================================
    print("User-defined Q4 element stiffness calculation not implemented")

    return Ke
