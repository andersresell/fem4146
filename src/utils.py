from element_utils import *


#====================================================================
# File that contains most parameters and settings for the simulation
#====================================================================
class Config:

    def __init__(self):
        self.E = None  #Young's modulus
        self.nu = None  #Poisson's ratio
        self.t = None  #Plate thickness
        self.element_type = None  #Element type
        self.problem_type = None  #Plane stress or plate problem

        #====================================================================
        # Plot settings
        #====================================================================
        self.disp_scaling = 1  #Scaling factor for displacements
        self.show_node_labels = True
        self.plot_external_forces = True


def create_config(E, nu, t, element_type, problem_type) -> Config:
    config = Config()
    config.E = E
    config.nu = nu
    config.t = t
    config.element_type = element_type
    config.problem_type = problem_type

    #====================================================================
    # Plot settings
    #====================================================================

    return config
