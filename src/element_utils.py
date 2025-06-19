import numpy as np
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)
#fmt: off

SMALL_VAL = 1e-8

PROBLEM_TYPE_PLANE_STRESS = 0
PROBLEM_TYPE_PLATE = 1

problem_type_to_string = {PROBLEM_TYPE_PLANE_STRESS: "plane_stress",
                          PROBLEM_TYPE_PLATE: "plate"}

def get_num_dofs_from_problem_type(problem_type):
    if problem_type == PROBLEM_TYPE_PLANE_STRESS:
        return 2
    else:
        assert problem_type == PROBLEM_TYPE_PLATE
        return 3

DOF_U = 0
DOF_V = 1

DOF_W = 0
DOF_THETAX =1
DOF_THETAY =2

# DOF_FREE = 0
# DOF_SUPPRESSED = 1

EDGE0_Q4 = 0
EDGE1_Q4 = 1
EDGE2_Q4 = 2
EDGE3_Q4 = 3
FACE_Q4 = 4

ELEMENT_TYPE_Q4 = 0
ELEMENT_TYPE_Q4R = 1
ELEMENT_TYPE_Q8 = 2
ELEMENT_TYPE_Q8R = 3
ELEMENT_TYPE_Q9 = 4
ELEMENT_TYPE_Q9R = 5
ELEMENT_TYPE_Q16 = 6

ELEMENT_CATEGORY_LAGRANGIAN = 0
ELEMENT_CATEGORY_SERENDIPITY = 1

def element_type_to_category(element_type):
    if element_type == ELEMENT_TYPE_Q4 or element_type == ELEMENT_TYPE_Q4R or \
        element_type == ELEMENT_TYPE_Q9 or element_type == ELEMENT_TYPE_Q9R or \
        element_type == ELEMENT_TYPE_Q16:
        return ELEMENT_CATEGORY_LAGRANGIAN
    elif element_type == ELEMENT_TYPE_Q8 or element_type == ELEMENT_TYPE_Q8R:
        return ELEMENT_CATEGORY_SERENDIPITY
    else:
        assert False

element_type_to_str =   {ELEMENT_TYPE_Q4:"Q4",
                         ELEMENT_TYPE_Q4R:"Q4R",
                         ELEMENT_TYPE_Q8:"Q8",
                         ELEMENT_TYPE_Q8R: "Q8R",
                         ELEMENT_TYPE_Q9:"Q9",
                         ELEMENT_TYPE_Q9R:"Q9R",
                         ELEMENT_TYPE_Q16:"Q16",}

element_type_to_nNl = {ELEMENT_TYPE_Q4: 4,
                       ELEMENT_TYPE_Q4R: 4,
                       ELEMENT_TYPE_Q8: 8,
                       ELEMENT_TYPE_Q8R: 8,
                       ELEMENT_TYPE_Q9: 9,
                       ELEMENT_TYPE_Q9R: 9,
                       ELEMENT_TYPE_Q16: 16}

element_type_to_nNl_1D = {ELEMENT_TYPE_Q4: 2,
                          ELEMENT_TYPE_Q4R: 2,
                          ELEMENT_TYPE_Q9: 3,
                          ELEMENT_TYPE_Q9R: 3,
                          ELEMENT_TYPE_Q16: 4}

element_type_to_nGauss_1D = {ELEMENT_TYPE_Q4: 2,
                             ELEMENT_TYPE_Q4R: 1,
                             ELEMENT_TYPE_Q8: 3,
                             ELEMENT_TYPE_Q8R: 2,
                             ELEMENT_TYPE_Q9: 3,
                             ELEMENT_TYPE_Q9R: 2,
                             ELEMENT_TYPE_Q16: 4}


il_to_ij_loc_all ={ELEMENT_TYPE_Q4: [(0,0),(1,0),(1,1),(0,1)],
                   ELEMENT_TYPE_Q4R: [(0,0),(1,0),(1,1),(0,1)],
                   ELEMENT_TYPE_Q9: [(0,0),(2,0),(2,2),(0,2),(1,0),(2,1),(1,2),(0,1),(1,1)],
                   ELEMENT_TYPE_Q9R: [(0,0),(2,0),(2,2),(0,2),(1,0),(2,1),(1,2),(0,1),(1,1)],
                   ELEMENT_TYPE_Q16: [(0,0),(3,0),(3,3),(0,3),(1,0),(2,0),(3,1),(3,2),(2,3),(1,3),(0,2),(0,1),(1,1),(2,1),(2,2),(1,2)]}
