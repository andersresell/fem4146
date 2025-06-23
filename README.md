# Educational 2D FEM tool for static linear analysis written in Python

---

## Current features:
- 2D structured mesh generator for Q4, Q8, Q9, and Q12 quadrilateral elements
- Plane stress elements and plate elements (Mindlin)
- Consistent load integration either on faces or edges, where the load is specified as a function of x, y
- Zero or prescribed displacement boundary conditions
- Simple GUI (see images below)

![](fem-node-labels.png)  
*The current problem has a linearly varying pressure load p(x) = (L_x - x)p_0 applied to the top edge.  
The arrows shown are the external nodal forces arising from consistent load integration of this pressure.  
The brown features on the left edge mark the fixed boundary condition.*

![](fem-stress.png)  
*Von Mises stress (no nodal averaging) computed for each element. Reaction forces that counteract the external forces are also displayed.  
Note that this mesh is finer than the mesh shown above.*

---

The following script was used to specify the displayed problem:

```python
from src.useful_imports import *  #import required functions

if __name__ == "__main__":

    E = 210e9  # Young's modulus in Pa
    nu = 0.3  # Poisson's ratio
    h = 0.01  # Plate thickness in m
    Lx = 10.0  # Length in x-direction
    Ly = 1.5  # Length in y-direction
    p0 = 1000000  # Pressure applied to the top edge
    element_type = ELEMENT_TYPE_Q16  # Use 16 node quadrilateral element
    problem_type = PROBLEM_TYPE_PLANE_STRESS  # Specify plane stress problem
    nEx = 20  # Number of elements in x-direction
    nEy = 5  # Number of elements in y-direction

    #====================================================================
    # Group problem settings in an object called config
    #====================================================================
    config = create_config(E, nu, h, element_type, problem_type)

    #====================================================================
    # Create a rectangular structured mesh. The mesh contains nodal
    # coordinates, element connectivity and predefined "node sets" 
    # and "element sets". The sets are used to assign boundary condtions
    # and perform load integraion.
    #====================================================================
    mesh = create_structured_quad_mesh(config, Lx=Lx, Ly=Ly, nEx=nEx, nEy=nEy)

    #====================================================================
    # Add fixed boundary condition to the left edge called "west"
    #====================================================================
    add_boundary_condition(config, mesh, "west", DOF_U, 0)  # set u to 0
    add_boundary_condition(config, mesh, "west", DOF_V, 0)  # set v to 0

    #====================================================================
    # Assign a linearly varying load on the top edge named "north"
    #====================================================================
    load_func = lambda x, y: p0 * (Lx - x) 
    add_load(config, mesh, "north", LOAD_TYPE_PRESSURE, load_func)

    #====================================================================
    # Create objects holding system matrices and vectors. Then assemble
    # stiffness matrix, integrate loads and assign boundary conditions
    #====================================================================
    solver_data = create_solver_data(config, mesh)
    solve(config, solver_data, mesh)

    Plot(config, mesh, solver_data)


```

## Usage

Make sure you have Python 3.8+ with **numpy**, **matplotlib**, and **scipy** installed.

To test the program, clone the project and run:

```bash
python main.py
```

No installation is currently supported.
