import numpy as np
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)
#fmt: off

SMALL_VAL = 1e-8

PROBLEM_TYPE_PLANE_STRESS = 0
PROBLEM_TYPE_PLATE = 1

problem_type_to_string = {PROBLEM_TYPE_PLANE_STRESS: "PROBLEM_TYPE_PLANE_STRESS",
                          PROBLEM_TYPE_PLATE: "PROBLEM_TYPE_PLATE"}

def get_num_dofs_from_problem_type(problem_type):
    if problem_type == PROBLEM_TYPE_PLANE_STRESS:
        return 2
    else:
        assert problem_type == PROBLEM_TYPE_PLATE
        return 3

DOF_U = 0
DOF_V = 1

DOF_W = 2
DOF_THETAX = 3
DOF_THETAY = 4
def get_dof_id(dof, problem_type):
    if problem_type==PROBLEM_TYPE_PLANE_STRESS:
        dof_id = dof - DOF_U
        assert dof_id >=0 and dof_id < 2
    else:
        assert problem_type==PROBLEM_TYPE_PLATE
        dof_id = dof - DOF_W
        assert dof_id >= 0 and dof_id < 3
    return dof_id

LOAD_TYPE_TRACTION = 0
LOAD_TYPE_PRESSURE = 1
LOAD_TYPE_BODY_FORCE = 2

load_type_to_string = {LOAD_TYPE_TRACTION: "LOAD_TYPE_TRACTION",
                       LOAD_TYPE_PRESSURE: "LOAD_TYPE_PRESSURE",
                       LOAD_TYPE_BODY_FORCE: "LOAD_TYPE_BODY_FORCE"}

ELEMENT_GEOMETRY_TYPE_EDGE = 0
ELEMENT_GEOMETRY_TYPE_FACE = 1

EDGE0_QUAD = 0
EDGE1_QUAD = 1
EDGE2_QUAD = 2
EDGE3_QUAD = 3
FACE_QUAD = 4
element_entity_to_geometry_type = {EDGE0_QUAD: ELEMENT_GEOMETRY_TYPE_EDGE,
                                   EDGE1_QUAD: ELEMENT_GEOMETRY_TYPE_EDGE,
                                   EDGE2_QUAD: ELEMENT_GEOMETRY_TYPE_EDGE,
                                   EDGE3_QUAD: ELEMENT_GEOMETRY_TYPE_EDGE,
                                   FACE_QUAD: ELEMENT_GEOMETRY_TYPE_FACE}


ELEMENT_TYPE_Q4 = 0
ELEMENT_TYPE_Q4R = 1
ELEMENT_TYPE_Q8 = 2
ELEMENT_TYPE_Q8R = 3
ELEMENT_TYPE_Q9 = 4
ELEMENT_TYPE_Q9R = 5
ELEMENT_TYPE_Q16 = 6
ELEMENT_TYPE_Q4_USER = 7 #Used for creating tasks for students to implement an element

ELEMENT_CATEGORY_LAGRANGIAN = 0
ELEMENT_CATEGORY_SERENDIPITY = 1

def element_type_to_category(element_type):
    if element_type == ELEMENT_TYPE_Q4 or element_type == ELEMENT_TYPE_Q4R or element_type==ELEMENT_TYPE_Q4_USER or \
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
                         ELEMENT_TYPE_Q16:"Q16",
                         ELEMENT_TYPE_Q4_USER:"Q4_USER"}

element_type_to_nNl = {ELEMENT_TYPE_Q4: 4,
                       ELEMENT_TYPE_Q4R: 4,
                       ELEMENT_TYPE_Q4_USER: 4,
                       ELEMENT_TYPE_Q8: 8,
                       ELEMENT_TYPE_Q8R: 8,
                       ELEMENT_TYPE_Q9: 9,
                       ELEMENT_TYPE_Q9R: 9,
                       ELEMENT_TYPE_Q16: 16,
                       ELEMENT_TYPE_Q4_USER: 4}

element_type_to_nNl_1D = {ELEMENT_TYPE_Q4: 2,
                          ELEMENT_TYPE_Q4R: 2,
                          ELEMENT_TYPE_Q4_USER: 2,
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
element_uses_reduced_integration = {ELEMENT_TYPE_Q4: False,
                                 ELEMENT_TYPE_Q4R: True,
                                 ELEMENT_TYPE_Q4_USER: False,
                                 ELEMENT_TYPE_Q8: False,
                                 ELEMENT_TYPE_Q8R: True,
                                 ELEMENT_TYPE_Q9: False,
                                 ELEMENT_TYPE_Q9R: True,
                                 ELEMENT_TYPE_Q16: False}


il_to_ij_loc_all ={ELEMENT_TYPE_Q4: [(0,0),(1,0),(1,1),(0,1)],
                   ELEMENT_TYPE_Q4_USER: [(0,0),(1,0),(1,1),(0,1)],
                   ELEMENT_TYPE_Q4R: [(0,0),(1,0),(1,1),(0,1)],
                   ELEMENT_TYPE_Q9: [(0,0),(2,0),(2,2),(0,2),(1,0),(2,1),(1,2),(0,1),(1,1)],
                   ELEMENT_TYPE_Q9R: [(0,0),(2,0),(2,2),(0,2),(1,0),(2,1),(1,2),(0,1),(1,1)],
                   ELEMENT_TYPE_Q16: [(0,0),(3,0),(3,3),(0,3),(1,0),(2,0),(3,1),(3,2),(2,3),(1,3),(0,2),(0,1),(1,1),(2,1),(2,2),(1,2)]}
