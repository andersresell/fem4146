from element_utils import *
from typing import List


#====================================================================
# Boundary condition
#====================================================================
class BC:

    def __init__(self):
        self.node_set_name = None
        self.value = None
        self.dof = None


#====================================================================
# File that contains most parameters and settings for the simulation
#====================================================================
class Config:

    def __init__(self):
        self.E = None  #Young's modulus
        self.nu = None  #Poisson's ratio
        self.h = None  #Plate thickness
        self.element_type = None  #Element type
        self.problem_type = None  #Plane stress or plate problem

        self.bcs: List[BC] = []  #List of boundary conditions

        #====================================================================
        # Plot settings
        #====================================================================
        self.disp_scaling = 1  #Scaling factor for displacements
        self.show_node_labels = False
        self.plot_external_forces = True
        self.contour_type = "disp"  #Displacement or stress
        self.disp_component = "mag"
        self.stress_component = "mises"
        self.node_scale = 1  #Scales nodes and node labels


def create_config(E, nu, h, element_type, problem_type) -> Config:
    config = Config()
    config.E = E
    config.nu = nu
    config.h = h
    config.element_type = element_type
    config.problem_type = problem_type

    #====================================================================
    # Plot settings
    #====================================================================

    return config
