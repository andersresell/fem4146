Educational 2D fem tool for linear static analysis
---
#Current features:
- 2D structured mesh generator for Q4, Q8, Q9 and Q12 quadrilateral elements
- Plane stress elements and plate elements (Mindlin)
- Consistent load integraion either on faces or edges where the load is specified as a function of x,y
- Zero or prescribed boundary conditions
- Simple GUI (see images below)


![The current problem has a linearly varying pressure load $p(x)=(L_x-x)p_0$](fem-node-labels.png)


![](fem-stress.png)

## Usage

Make sure you have Python 3.8+ with numpy, scipy and matplotlib installed.
To test the program, simply clone the project and run: python main.py from the project root directory.
No installation is currently supported.
