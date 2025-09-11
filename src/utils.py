from src.fem_utils import *
from typing import List


class BC:
    """
    Defines a boundary condition:
    - node_set_name: Name of the node set to which the boundary condition applies
    - value: The value to which the degree of freedom is set (e.g., displacement)
    - dof: The degree of freedom that is prescribed (e.g., DOF_U, DOF_V for plane 
    stress or DOF_W, DOF_THETAX, DOF_THETAY for plate problems)
    """

    def __init__(self):
        self.node_set_name = None
        self.value = None
        self.dof = None


class Load:
    """
    Defines a load that can be applied to the mesh. Can be used both for plane stress and plate problems and it
    can define both surface loads (like traction or pressure) and body forces (like gravity). The element set will define
    if the load is applied to a surface or a volume.
    - element_set_name: Name of the element set to which the load applies
    - load_type: Type of load, e.g., "traction", "pressure", "gravity".
    - load_function: A lambda function that calculates the load as a function of position.
                     For example, a traction (2D) or pressure (scalar) for plane stress or a pressure (scalar) for 
                     plate problems.
    
    Example when the problem type is plane stress:
    load = Load()
    load.element_set_name = "west"
    load.load_function = lambda x, y: np.array([10, -1000*x])  # Traction in x and y direction
    """

    def __call__(self):
        self.element_set_name = None
        self.load_type = None
        self.load_function = None


class Config:
    """Configuration struct that holds all the parameters and settings for the simulation and plotting."""

    def __init__(self):

        #====================================================================
        # Problem settings
        #====================================================================
        self.E = None  #Young's modulus
        self.nu = None  #Poisson's ratio
        self.h = None  #Plate thickness
        self.element_type = None  #Element type
        self.problem_type = None  #Plane stress or plate problem
        self.hourglass_scaling = 0.1  #Scaling factor for hourglass stabilization term in Q4R element

        self.bcs: List[BC] = []  #List of boundary conditions
        self.loads: List[Load] = []  #List of loads

        #====================================================================
        # Plot settings
        #====================================================================
        self.show_mesh = True
        self.disp_scaling = 1  #Scaling factor for displacements
        self.show_node_labels = False
        self.show_bcs = False
        self.contour_type = "disp"  #Displacement or stress
        self.disp_component = "mag"
        self.stress_component = "mises"
        self.arrow_type = "external forces"  #external forces, reaction forces, disp, none
        self.node_scale = 1  #Scales nodes and node labels
        self.arrow_scale = 1
        self.specify_contour_limits = False
        self.contour_min = 0
        self.contour_max = 1


def create_config(E, nu, h, element_type, problem_type) -> Config:
    config = Config()
    config.E = E
    config.nu = nu
    config.h = h
    config.element_type = element_type
    config.problem_type = problem_type

    return config
